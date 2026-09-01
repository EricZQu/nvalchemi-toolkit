# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration of the on-policy generate-label-train segment loop."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nvalchemi._serialization import (
    _callable_signature,
    _cls_path_of,
    _constructor_signature,
    _extract_init_kwargs_from_attrs,
    _import_callable,
)
from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.training.distillation.replay import ReplayEviction
from nvalchemi.training.distillation.scoring import (
    InProcessTeacherScorer,
    TeacherScorer,
)

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["OnPolicyConfig"]

_SPEC_SCALARS = (bool, int, float, str, torch.dtype, torch.device)
"""Propagator constructor argument types a spec can carry verbatim."""

_RUNTIME_DYNAMICS_ARGS = frozenset(
    {"model", "hooks", "convergence_hook", "sinks", "sampler", "active_batch"}
)
"""Propagator constructor arguments held by the runtime rather than the spec."""

_RECORDED_SPEC_ATTR = "_recipe_spec"
"""Attribute a recipe-built propagator remembers its own spec under."""

_RECIPE_OBJECT_KEYS = frozenset({"dynamics", "teacher_scorer", "seed_dataset"})
"""Recipe entries that reference an object rather than carrying a scalar knob."""


def _dataset_spec_dict(dataset: BatchDatasetProtocol, field: str) -> dict[str, Any]:
    """Return the store reference a path-backed dataset round-trips as.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Dataset to reference. Only a dataset reading a filesystem or URI store
        can be named in a recipe; one holding its samples in memory cannot.
    field : str
        Name of the recipe field being serialized, quoted in the error.

    Returns
    -------
    dict[str, Any]
        ``{"path": ..., "device": ...}`` reference the rebuild reopens.

    Raises
    ------
    ValueError
        If *dataset* is not backed by a store a path names.
    """
    store = getattr(getattr(dataset, "reader", None), "_store", None)
    if not isinstance(store, (str, Path)):
        raise ValueError(
            f"{field} is a {type(dataset).__name__} holding its samples in "
            "memory, which no recipe can name: a spec references a dataset by "
            "the store it reads. Write the samples to a store with "
            "nvalchemi.training.distillation.label_dataset (or an "
            "AtomicDataZarrWriter) and point the recipe at that path, or "
            f"re-supply {field} at construction."
        )
    return {"path": str(store), "device": str(getattr(dataset, "target_device", "cpu"))}


def _dataset_from_spec_dict(spec: Mapping[str, Any]) -> BatchDatasetProtocol:
    """Reopen the dataset :func:`_dataset_spec_dict` referenced.

    Parameters
    ----------
    spec : Mapping[str, Any]
        Reference produced by :func:`_dataset_spec_dict`.

    Returns
    -------
    BatchDatasetProtocol
        Dataset over the referenced store. The reader it opens stays open for
        the caller to close.
    """
    from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset

    return Dataset(AtomicDataZarrReader(spec["path"]), device=spec.get("device", "cpu"))


def _dynamics_spec_dict(dynamics: BaseDynamics) -> dict[str, Any]:
    """Return the ``{"cls_path", "kwargs"}`` reference a propagator rebuilds from.

    A propagator a recipe built remembers the reference it was built from,
    which is the one it round-trips as. Any other one is introspected: its
    constructor arguments are read back off matching attributes, which works
    for a propagator that keeps them and fails for one that stores them as
    private internals instead — a timestep normalized into internal units, say,
    which rebuilding from would convert a second time — or whose constructor
    annotations name a type only a type checker imports.

    An argument travels as itself when JSON can carry it; a ``torch.dtype`` and
    a ``torch.device`` travel as their names — ``"float64"``, ``"cuda:0"`` — and
    are read back into objects for a constructor annotated to take one.

    The reference is a dotted path and keyword arguments rather than a
    :class:`~nvalchemi.training._spec.BaseSpec` because building one of those
    resolves the target's annotations, which a dynamics constructor's
    ``BaseModelMixin`` annotation does not survive: it is imported under
    ``TYPE_CHECKING`` throughout :mod:`nvalchemi.dynamics`. Rebuilding calls
    the constructor directly and needs no annotation at all.

    The student is left out either way and rebound at rebuild time, and so is
    every other live collaborator: hooks, a convergence hook, sinks, a sampler.
    Those are runtime objects the caller re-registers, exactly as
    :meth:`~nvalchemi.training.TrainingStrategy.to_spec_dict` leaves the
    strategy's own hooks out, and a propagator carrying one is reported rather
    than silently rebuilt without it.

    Raises
    ------
    ValueError
        If the propagator neither remembers a reference nor exposes the
        constructor arguments it was built with.
    """
    recorded = getattr(dynamics, _RECORDED_SPEC_ATTR, None)
    if isinstance(recorded, Mapping):
        return dict(recorded)
    kwargs, omitted = _introspected_dynamics_kwargs(dynamics)
    if omitted:
        warnings.warn(
            f"The propagator's {omitted!r} hold runtime objects no recipe "
            "describes, so they are omitted and a rebuilt propagator starts "
            "without them. Re-register them on the rebuilt dynamics, or "
            "re-supply the whole propagator at construction.",
            UserWarning,
            stacklevel=3,
        )
    return {"cls_path": _cls_path_of(type(dynamics)), "kwargs": kwargs}


def _spec_scalar(value: Any) -> Any:
    """Return the JSON-ready form of a constructor argument a spec carries."""
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    return value


def _introspected_dynamics_kwargs(
    dynamics: BaseDynamics,
) -> tuple[dict[str, Any], list[str]]:
    """Return a propagator's serializable constructor arguments and what was dropped."""
    try:
        signature = _constructor_signature(type(dynamics))
        attributes = _extract_init_kwargs_from_attrs(dynamics)
    except Exception as exc:
        raise ValueError(
            f"OnPolicyConfig.dynamics is a {type(dynamics).__name__} whose "
            f"constructor cannot be read back ({exc}), so no recipe describes "
            "it. Build the propagator from a recipe — OnPolicyConfig."
            "from_spec_dict keeps the reference it built from — or re-supply "
            "dynamics at construction."
        ) from exc
    kwargs: dict[str, Any] = {}
    omitted: list[str] = []
    for name, value in attributes.items():
        if name in _RUNTIME_DYNAMICS_ARGS:
            # The student is the one collaborator a rebuild rebinds itself.
            if value and name != "model":
                omitted.append(name)
        elif value is None or isinstance(value, _SPEC_SCALARS):
            kwargs[name] = _spec_scalar(value)
        elif value:
            omitted.append(name)
    missing = sorted(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in {"self", "model"}
        and name not in kwargs
        and (
            parameter.default is inspect.Parameter.empty
            or (name not in attributes and name not in _RUNTIME_DYNAMICS_ARGS)
        )
    )
    if missing:
        raise ValueError(
            f"OnPolicyConfig.dynamics is a {type(dynamics).__name__} that does "
            f"not expose its {missing!r} as attributes, so no recipe describes "
            "it: rebuilding it would fall back to the constructor's own "
            "defaults for arguments this propagator was not built with. Build "
            "the propagator from a recipe — OnPolicyConfig.from_spec_dict "
            "keeps the reference it built from — or re-supply dynamics at "
            "construction."
        )
    return kwargs, sorted(omitted)


def _decoded_dynamics_kwargs(
    target: Callable[..., Any], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Return recipe kwargs with the torch scalars a spec stringified read back."""
    try:
        parameters = _callable_signature(target).parameters
    except Exception:
        return dict(kwargs)
    decoded = dict(kwargs)
    for name, value in kwargs.items():
        annotation = getattr(parameters.get(name), "annotation", None)
        if not isinstance(value, str):
            continue
        if annotation is torch.dtype:
            decoded[name] = getattr(torch, value)
        elif annotation is torch.device:
            decoded[name] = torch.device(value)
    return decoded


def _dynamics_from_spec_dict(
    spec: Mapping[str, Any], student: BaseModelMixin
) -> BaseDynamics:
    """Rebuild the propagator around *student* and record the reference on it."""
    target = _import_callable(spec["cls_path"])
    dynamics = target(
        model=student, **_decoded_dynamics_kwargs(target, spec.get("kwargs", {}))
    )
    if not isinstance(dynamics, BaseDynamics):
        raise ValueError(
            f"OnPolicyConfig.dynamics rebuilt a {type(dynamics).__name__} from "
            f"{spec['cls_path']!r}; expected a BaseDynamics propagator."
        )
    object.__setattr__(dynamics, _RECORDED_SPEC_ATTR, dict(spec))
    return dynamics


def _scorer_spec_dict(
    scorer: TeacherScorer, teacher: BaseModelMixin | None
) -> dict[str, Any]:
    """Return the signals, cast dtype, and teacher reference of an in-process scorer."""
    if not isinstance(scorer, InProcessTeacherScorer):
        raise ValueError(
            f"OnPolicyConfig.teacher_scorer is a {type(scorer).__name__}, which "
            "no recipe describes: only an InProcessTeacherScorer round-trips, as "
            "a signal set over the strategy's own teacher. Re-supply the scorer "
            "at construction."
        )
    if teacher is not None and scorer.teacher is not teacher:
        raise ValueError(
            "OnPolicyConfig.teacher_scorer scores with a "
            f"{type(scorer.teacher).__name__} that is not the strategy's "
            "models['teacher'], and a recipe references the teacher by that "
            "name rather than serializing a second model. Score with the "
            "strategy's teacher, or re-supply the scorer at construction."
        )
    cast_to = scorer.cast_to
    return {
        "teacher": "teacher",
        "signals": sorted(scorer.signals),
        "cast_to": None if cast_to is None else str(cast_to).removeprefix("torch."),
    }


class _OnPolicyKnobs(BaseModel):
    """Scalar knobs of a segment loop, validatable without building a model.

    Split out of :class:`OnPolicyConfig` so a recipe's scalars can be checked
    against the constraints the config itself enforces — by the CLI's
    pre-flight, before a teacher is loaded onto a device — rather than against
    a second, drifting copy of them.
    """

    replay_ratio: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Fraction of every training batch drawn from the replay buffer; "
                "the rest comes from the reference dataset."
            ),
        ),
    ]
    steps_per_segment: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "Training batches drawn from each segment's mixture, one "
                "optimizer step each unless an update hook vetoes the step."
            ),
        ),
    ]
    batch_size: Annotated[
        int,
        Field(
            default=8,
            gt=0,
            description=(
                "Samples per training batch, split between the reference "
                "dataset and the replay buffer at replay_ratio."
            ),
        ),
    ] = 8
    segment_steps: Annotated[
        int,
        Field(
            default=100,
            gt=0,
            description="Propagator steps generated per segment.",
        ),
    ] = 100
    label_frequency: Annotated[
        int,
        Field(
            default=100,
            gt=0,
            description=(
                "Propagator steps between teacher labelings. Larger values trade "
                "label density for generation throughput."
            ),
        ),
    ] = 100
    replay_capacity: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Frames the replay buffer keeps; None leaves it unbounded.",
        ),
    ] = None
    replay_eviction: Annotated[
        ReplayEviction,
        Field(
            default="fifo",
            description=(
                "Policy retiring frames from a full replay buffer. 'uncertainty' "
                "is reserved and not implemented yet."
            ),
        ),
    ] = "fifo"
    replay_device: Annotated[
        torch.device | str | None,
        Field(
            default=None,
            description=(
                "Device the replay buffer holds frames on. Generated frames "
                "reach it from a host-memory sink, so None stages them where "
                "the reference dataset emits its own batches — the mixture is "
                "collated before training moves it — and leaves them in host "
                "memory when the run has no reference dataset. Set it only to "
                "override that, and load the reference dataset there too."
            ),
        ),
    ] = None
    seed: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description=(
                "Base seed of every segment's mixture sampler, combined with the "
                "segment index so consecutive segments draw different reference "
                "samples and replicate runs can be made independent."
            ),
        ),
    ] = 0
    weight_sync_frequency: Annotated[
        int,
        Field(
            default=1,
            gt=0,
            description=(
                "Segments between weight syncs to the propagator. Reserved: "
                "must be 1 while the propagator shares the student module."
            ),
        ),
    ] = 1

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_replay_eviction(self) -> _OnPolicyKnobs:
        """Hold the reserved eviction policy until committee scoring lands."""
        if self.replay_eviction == "uncertainty":
            raise ValueError(
                "replay_eviction='uncertainty' is reserved for committee-based "
                "frame selection and is not implemented yet; use 'fifo'."
            )
        return self

    @model_validator(mode="after")
    def _validate_weight_sync(self) -> _OnPolicyKnobs:
        """Hold the reserved sync knob at 1 until the decoupled paths land."""
        if self.weight_sync_frequency != 1:
            raise ValueError(
                "weight_sync_frequency must be 1: the propagator holds the same "
                "student module the trainer updates, so an eager run is never out "
                f"of sync; got {self.weight_sync_frequency!r}. Larger values are "
                "reserved for the compiled and asynchronous teacher paths."
            )
        return self


def _on_policy_knobs(recipe: Mapping[str, Any]) -> _OnPolicyKnobs:
    """Validate a segment-loop recipe's scalar knobs, ignoring its object entries.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Recipe produced by :meth:`OnPolicyConfig.to_spec_dict`, or the
        ``on_policy`` block of a distillation job spec.

    Returns
    -------
    _OnPolicyKnobs
        The knobs the recipe sets, with the config's own defaults filled in.

    Raises
    ------
    pydantic.ValidationError
        If a knob is out of range, of the wrong type, or unknown.
    """
    return _OnPolicyKnobs.model_validate(
        {key: value for key, value in recipe.items() if key not in _RECIPE_OBJECT_KEYS}
    )


class OnPolicyConfig(_OnPolicyKnobs):
    """Knobs of one on-policy distillation segment loop.

    On-policy distillation alternates two phases. A *generation* phase runs the
    student's own propagator for ``segment_steps`` steps from the seeded state,
    labeling frames with the teacher as it goes; a *training* phase then takes
    ``steps_per_segment`` optimizer steps on batches mixed from the reference
    dataset and the replay buffer at ``replay_ratio``. The student the
    propagator holds is the module the trainer updates, so each segment
    generates from a fresher policy than the last.

    The propagator is deliberately typed as
    :class:`~nvalchemi.dynamics.base.BaseDynamics` and named ``dynamics``, not
    ``integrator``: a relaxation optimizer such as
    :class:`~nvalchemi.dynamics.optimizers.FIRE` drives the loop exactly as a
    thermostat does, and nothing downstream of this config reads a velocity or
    a temperature. Seed structures must still carry whatever the chosen
    propagator declares in ``__needs_keys__`` — ``velocities`` and
    ``atomic_masses`` for the integrators and the optimizers alike.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator generating on-policy frames, holding the student module.
    teacher_scorer : TeacherScorer
        Scorer labeling generated frames.
    seed_dataset : BatchDatasetProtocol | None, optional
        Structures the generated trajectories start from, propagated as one
        batch. Default ``None``, which requires a ``sampler`` instead.
    replay_ratio : float
        Fraction of every training batch drawn from the replay buffer.
    steps_per_segment : int
        Training batches taken per segment.
    batch_size : int, optional
        Samples per training batch, across both mixture sources. Default ``8``.
    segment_steps : int, optional
        Propagator steps taken per segment. Default ``100``.
    label_frequency : int, optional
        Label every this many propagator steps. Default ``100``.
    replay_capacity : int | None, optional
        Frame capacity of the replay buffer. Default ``None`` (unbounded).
    replay_eviction : {"fifo", "uncertainty"}, optional
        Eviction policy of the replay buffer. Default ``"fifo"``.
    replay_device : torch.device | str | None, optional
        Device the replay buffer keeps frames on. Default ``None`` (wherever
        the reference dataset emits its own batches, and host memory without
        one).
    seed : int, optional
        Base seed of every segment's mixture sampler. Default ``0``.
    sampler : SizeAwareSampler | None, optional
        Size-aware sampler bin-packing the initial batch, in place of
        ``seed_dataset``. Default ``None``.
    weight_sync_frequency : int, optional
        Segments between pushing student weights to the propagator. Default
        ``1``, currently the only accepted value.

    Raises
    ------
    ValueError
        If a count is not positive, if ``replay_ratio`` falls outside
        ``[0, 1]``, if neither or both of ``seed_dataset`` and ``sampler`` are
        given, if ``replay_eviction`` is the reserved ``"uncertainty"``, or if
        ``weight_sync_frequency`` is not ``1``.

    Examples
    --------
    >>> from nvalchemi.training.distillation import InProcessTeacherScorer, OnPolicyConfig
    >>> config = OnPolicyConfig(  # doctest: +SKIP
    ...     dynamics=NVTLangevin(student, dt=0.5, temperature=300.0),
    ...     teacher_scorer=InProcessTeacherScorer(teacher, ["energy", "forces"]),
    ...     seed_dataset=seed_dataset,
    ...     replay_ratio=0.25,
    ...     steps_per_segment=32,
    ...     batch_size=16,
    ...     segment_steps=50,
    ...     label_frequency=10,
    ...     replay_capacity=8192,
    ... )

    Notes
    -----
    ``label_frequency`` is the throughput knob: the teacher is the expensive
    model, and a segment that labels every tenth frame costs a tenth of the
    teacher passes while still generating every frame at student speed.
    Frequencies are counted against the propagator's cumulative ``step_count``,
    which chunked runs carry across segments, so the labeling cadence does not
    restart at each segment boundary.

    ``steps_per_segment`` is spent as a budget of training batches, which is a
    budget of optimizer steps only while every batch takes one. Under an update
    orchestrator that vetoes the optimizer step on accumulation micro-batches,
    a segment lands proportionally fewer steps and the run takes
    proportionally more segments — and so proportionally more generation and
    teacher passes — to reach ``num_steps``.

    ``seed`` is the mixture's only source of randomness the loop owns. The
    segment loader is rebuilt every segment and its sampler seeds itself from
    ``seed`` plus the segment index, so the reference draw is reproducible
    across runs without repeating within one — and replicate runs meant to be
    independent, an ensemble or a seed-sensitivity sweep, need distinct values
    here rather than a distinct global ``torch`` seed, which the sampler's own
    generator never reads.

    ``weight_sync_frequency`` is reserved and must be ``1`` for now. Eager runs
    need no sync at all — the propagator and the trainer share one module
    object, so an optimizer step is visible to the next generated frame
    immediately — and the knob only becomes meaningful once the propagator
    holds a compiled or remote copy of the student.
    """

    dynamics: Annotated[
        BaseDynamics,
        Field(
            description=(
                "Propagator generating on-policy frames from the student. Any "
                "BaseDynamics: an integrator for trajectories, an optimizer for "
                "relaxation paths."
            )
        ),
    ]
    teacher_scorer: Annotated[
        TeacherScorer,
        Field(description="Scorer producing the teacher signals for generated frames."),
    ]
    seed_dataset: Annotated[
        BatchDatasetProtocol | None,
        Field(
            default=None,
            description=(
                "Structures the generated trajectories are seeded from, "
                "propagated as one batch. Mutually exclusive with sampler."
            ),
        ),
    ] = None
    sampler: Annotated[
        SizeAwareSampler | None,
        Field(
            default=None,
            description=(
                "Size-aware sampler bin-packing the initial batch under its own "
                "size budget, in place of seed_dataset. It seeds the run and "
                "nothing more: the loop drives no refill, so converged "
                "structures are not graduated and no fresh seed is backfilled."
            ),
        ),
    ] = None

    @model_validator(mode="after")
    def _validate_seed_source(self) -> OnPolicyConfig:
        """Require exactly one of the two ways to build the initial batch."""
        if (self.seed_dataset is None) == (self.sampler is None):
            given = [
                name
                for name, value in (
                    ("seed_dataset", self.seed_dataset),
                    ("sampler", self.sampler),
                )
                if value is not None
            ]
            raise ValueError(
                "Exactly one of seed_dataset or sampler must be set: a sampler "
                "builds the initial batch from its own dataset under its own "
                "size budget, so a seed_dataset alongside it would never be "
                f"read. Got {given!r}."
            )
        return self

    def to_spec_dict(self, *, teacher: BaseModelMixin | None = None) -> dict[str, Any]:
        """Serialize the segment loop to a JSON-ready recipe.

        Every scalar knob round-trips as itself. The three structured fields
        round-trip as references instead: the propagator as the spec it
        rebuilds from, with the student rebound at construction; the scorer as
        its signal set, its cast dtype, and the name of the strategy model it
        scores with; and ``seed_dataset`` as the store it reads.

        What stays runtime-only is ``sampler`` — it owns a live dataset and a
        size budget that a recipe cannot name — along with the hooks, sinks,
        and convergence hook a propagator may carry, and the in-flight state of
        a run, which travels in a checkpoint rather than in a spec. A
        ``sampler``-seeded config therefore round-trips only if the sampler is
        supplied again at rebuild.

        Parameters
        ----------
        teacher : BaseModelMixin | None, optional
            Model the recipe's ``"teacher"`` reference resolves to, checked
            against the scorer's own. Default ``None`` (unchecked).

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.

        Raises
        ------
        ValueError
            If the propagator cannot be described by a spec, if the scorer is
            not an
            :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`
            over *teacher*, or if ``seed_dataset`` holds its samples in memory.

        Warns
        -----
        UserWarning
            If the propagator carries hooks or other live collaborators, which
            a rebuilt one starts without.

        Notes
        -----
        Knobs the epic's sibling branches add to this config — a convergence
        lifecycle for relaxation propagators among them — gain their own spec
        entries as those branches integrate; this recipe describes the fields
        the class declares here.
        """
        spec: dict[str, Any] = {
            "dynamics": _dynamics_spec_dict(self.dynamics),
            "teacher_scorer": _scorer_spec_dict(self.teacher_scorer, teacher),
            "seed_dataset": (
                None
                if self.seed_dataset is None
                else _dataset_spec_dict(
                    self.seed_dataset, "OnPolicyConfig.seed_dataset"
                )
            ),
            "replay_ratio": self.replay_ratio,
            "steps_per_segment": self.steps_per_segment,
            "batch_size": self.batch_size,
            "segment_steps": self.segment_steps,
            "label_frequency": self.label_frequency,
            "replay_capacity": self.replay_capacity,
            "replay_eviction": self.replay_eviction,
            "replay_device": (
                None if self.replay_device is None else str(self.replay_device)
            ),
            "seed": self.seed,
            "weight_sync_frequency": self.weight_sync_frequency,
        }
        if self.sampler is not None:
            warnings.warn(
                "OnPolicyConfig.sampler owns a live dataset and a size budget "
                "that no recipe names, so it is omitted and a rebuilt config "
                "needs it supplied again — or a seed_dataset in its place.",
                UserWarning,
                stacklevel=2,
            )
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        student: BaseModelMixin,
        teacher: BaseModelMixin,
        sampler: SizeAwareSampler | None = None,
    ) -> OnPolicyConfig:
        """Rebuild a segment loop from a :meth:`to_spec_dict` recipe.

        Parameters
        ----------
        spec : Mapping[str, Any]
            Recipe produced by :meth:`to_spec_dict`, optionally after a JSON
            round trip.
        student : BaseModelMixin
            Model the rebuilt propagator generates with. It must be the very
            module the strategy trains, which is what makes the data
            on-policy.
        teacher : BaseModelMixin
            Model the rebuilt scorer labels with.
        sampler : SizeAwareSampler | None, optional
            Runtime sampler for a recipe that seeds from one rather than from a
            dataset. Default ``None``.

        Returns
        -------
        OnPolicyConfig
            Config equal to the serialized one on every field a recipe carries.

        Raises
        ------
        ValueError
            If the propagator spec builds something that is not a
            :class:`~nvalchemi.dynamics.base.BaseDynamics`, or if the rebuilt
            config is invalid.
        """
        scorer_spec = spec["teacher_scorer"]
        cast_to = scorer_spec.get("cast_to")
        seed_spec = spec.get("seed_dataset")
        knobs = _on_policy_knobs(spec)
        return cls(
            dynamics=_dynamics_from_spec_dict(spec["dynamics"], student),
            teacher_scorer=InProcessTeacherScorer(
                teacher,
                scorer_spec["signals"],
                cast_to=None if cast_to is None else getattr(torch, cast_to),
            ),
            seed_dataset=(
                None if seed_spec is None else _dataset_from_spec_dict(seed_spec)
            ),
            sampler=sampler,
            **{name: getattr(knobs, name) for name in _OnPolicyKnobs.model_fields},
        )
