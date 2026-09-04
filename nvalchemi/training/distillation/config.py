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
from typing import TYPE_CHECKING, Annotated, Any, get_args

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nvalchemi._serialization import _cls_path_of, _import_callable
from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.training.distillation.hooks import TeacherLabelHook
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

_LIVE_COLLABORATOR_ARGS = _RUNTIME_DYNAMICS_ARGS - {"model", "active_batch"}
"""Runtime propagator arguments a rebuild neither rebinds nor restores."""

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
    which is the one it round-trips as, so a knob mutated on the live object
    afterwards does not travel — latent rather than live, since the segment
    loop passes ``n_steps`` explicitly to every
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` call it makes, and a
    shipped propagator normalizes its physics knobs into private internals.

    Any other propagator is introspected: its constructor arguments are read
    back off matching attributes, which works for one that keeps them and
    fails for one that stores them as private internals instead — a timestep
    normalized into internal units, say, which rebuilding from would convert a
    second time. Every shipped integrator and optimizer is of that second kind,
    so a hand-built one of those is refused and only a recipe-built propagator
    round-trips.

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
    than silently rebuilt without it — a propagator that remembers a reference
    included, since the reference records what it was built with rather than
    what it now holds.

    Raises
    ------
    ValueError
        If the propagator neither remembers a reference nor exposes the
        constructor arguments it was built with.
    """
    live = _live_collaborators(dynamics)
    recorded = getattr(dynamics, _RECORDED_SPEC_ATTR, None)
    if isinstance(recorded, Mapping):
        _warn_live_collaborators(live)
        return dict(recorded)
    kwargs, unserializable = _introspected_dynamics_kwargs(dynamics)
    _warn_live_collaborators(sorted(set(live) | set(unserializable)))
    return {"cls_path": _cls_path_of(type(dynamics)), "kwargs": kwargs}


def _live_collaborators(dynamics: BaseDynamics) -> list[str]:
    """Return the collaborators a propagator holds that a rebuilt one would not.

    Read off the live propagator rather than off the constructor arguments it
    can be introspected for, so that one registered after construction counts
    and so that a propagator built from a recipe — which is never introspected
    at all — is checked too.

    Two collaborators are left out. The student is rebound at rebuild time. And
    so is the :class:`~nvalchemi.training.distillation.TeacherLabelHook` the
    segment loop registers for the length of a run and removes afterwards,
    which a rebuilt loop registers for itself, exactly as
    :class:`~nvalchemi.training.distillation.DistillationStrategy` keeps its own
    internal hooks out of its spec. Reporting it would fire at every
    mid-segment checkpoint and say nothing.
    """
    held = [
        name
        for name in _LIVE_COLLABORATOR_ARGS
        if name != "hooks" and getattr(dynamics, name, None)
    ]
    if any(
        not isinstance(hook, TeacherLabelHook)
        for hook in getattr(dynamics, "hooks", None) or ()
    ):
        held.append("hooks")
    return sorted(held)


def _warn_live_collaborators(omitted: list[str]) -> None:
    """Report the collaborators a rebuilt propagator starts without."""
    if not omitted:
        return
    warnings.warn(
        f"The propagator's {omitted!r} hold runtime objects no recipe "
        "describes, so they are omitted and a rebuilt propagator starts "
        "without them. Re-register them on the rebuilt dynamics, or "
        "re-supply the whole propagator at construction.",
        UserWarning,
        stacklevel=4,
    )


def _spec_scalar(value: Any) -> Any:
    """Return the JSON-ready form of a constructor argument a spec carries."""
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    return value


def _dynamics_signature(target: Callable[..., Any]) -> inspect.Signature:
    """Return *target*'s signature, leaving annotations unresolved when they must be.

    A propagator constructor annotates its model as ``BaseModelMixin``, a name
    :mod:`nvalchemi.dynamics` imports under ``TYPE_CHECKING`` alone, so
    resolving the string annotations of any propagator written in that style
    raises :exc:`NameError`. Falling back to the unresolved signature keeps the
    parameter names and defaults a recipe reads, and leaves the annotations as
    the strings the source wrote — which
    :func:`_decoded_dynamics_kwargs` matches alongside the resolved objects.

    Core's :func:`~nvalchemi._serialization._callable_signature` stays strict
    on purpose: :mod:`nvalchemi.training._spec` turns the annotations it
    resolves into pydantic field types, and a string there would build the
    wrong spec rather than a lenient one. Rebuilding a propagator calls its
    constructor directly and needs no annotation object at all.
    """
    try:
        return inspect.signature(target, eval_str=True)
    except NameError:
        return inspect.signature(target)


def _init_kwargs_from_attrs(dynamics: BaseDynamics) -> dict[str, Any]:
    """Read a propagator's constructor arguments back off its own attributes."""
    kwargs: dict[str, Any] = {}
    for name, parameter in _dynamics_signature(type(dynamics)).parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        try:
            kwargs[name] = getattr(dynamics, name)
        except AttributeError:
            continue
    return kwargs


def _introspected_dynamics_kwargs(
    dynamics: BaseDynamics,
) -> tuple[dict[str, Any], list[str]]:
    """Return a propagator's serializable constructor arguments and what was dropped."""
    try:
        signature = _dynamics_signature(type(dynamics))
        attributes = _init_kwargs_from_attrs(dynamics)
    except (TypeError, ValueError) as exc:
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
            continue
        if value is None or isinstance(value, _SPEC_SCALARS):
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


def _annotation_accepts(annotation: Any, scalar: type) -> bool:
    """Return whether a constructor annotation takes *scalar*, on its own or in a union.

    Both forms an annotation reaches this in are matched: the object
    :func:`_dynamics_signature` resolves it to, and the source string it leaves
    when a propagator's module hides an import behind ``TYPE_CHECKING``.
    """
    named = f"torch.{scalar.__name__}"
    if isinstance(annotation, str):
        return named in {part.strip() for part in annotation.split("|")}
    return scalar in (get_args(annotation) or (annotation,))


def _decoded_dynamics_kwargs(
    target: Callable[..., Any], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Return recipe kwargs with the torch scalars a spec stringified read back."""
    try:
        parameters = _dynamics_signature(target).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    decoded = dict(kwargs)
    for name, value in kwargs.items():
        annotation = getattr(parameters.get(name), "annotation", None)
        if not isinstance(value, str):
            continue
        if _annotation_accepts(annotation, torch.dtype):
            decoded[name] = getattr(torch, value)
        elif _annotation_accepts(annotation, torch.device):
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

    That split is also the boundary of what a pre-flight decides. These knobs
    are the declarative half of a run: a count out of range, a reserved policy,
    an unknown key, each of them readable off the recipe text alone. Whether
    the *objects* a recipe names compose with the loop that will drive them —
    a propagator carrying a sampler of its own, a stage whose shape the loop
    cannot chunk into segments, a criterion reading a field no seed carries —
    is settled where the loop is installed and those objects are in hand.
    Deciding it here would mean inferring composition from a class path and a
    kwargs dict, and would move the refusal away from the code that owns the
    contract it enforces. Cheap to read and cheap to fix belongs to the
    pre-flight; compatible-with-this-loop belongs to the loop.
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
                "Propagator steps between teacher labelings, on top of the "
                "segment's own last frame. Larger values trade label density "
                "for generation throughput."
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
                "the reference dataset actually emits its own batches — the "
                "mixture is collated before training moves it — and leaves "
                "them in host memory when the run has no reference dataset. "
                "Set it only to override that, and load the reference dataset "
                "there too."
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

    Notes
    -----
    A clean pass here says the recipe's knobs are self-consistent, not that
    the run will start: the propagator, the scorer, and the seed store it
    names are skipped, and whether they compose with the segment loop is
    :meth:`DistillationStrategy.run`'s call rather than this one's.
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
        Scorer labeling generated frames. Declaring ``label_fields`` on a
        custom one is what makes the fields it writes knowable up front.
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
        Label every this many propagator steps, alongside each segment's last
        frame. Default ``100``.
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
    ``replay_capacity`` is spent by ``replay_eviction="fifo"`` on whole frames
    in arrival order, and a segment contributes one frame per propagated
    trajectory per labeled step. A capacity that is not a multiple of the
    number of trajectories in the seed batch therefore cuts a segment's
    contribution mid-step, leaving the trajectories at the front of the batch
    represented more often than the ones at the back in every mixture drawn
    afterwards. Size it as a multiple of the trajectory count to keep the
    buffer balanced across seeds.

    ``label_frequency`` is the throughput knob: the teacher is the expensive
    model, and a segment that labels every tenth frame costs a tenth of the
    teacher passes while still generating every frame at student speed.
    Frequencies are counted against the propagator's cumulative ``step_count``,
    which chunked runs carry across segments, so the labeling cadence does not
    restart at each segment boundary.

    Each segment additionally labels the frame it ends on, whatever the
    cadence, because that is the most on-policy frame it produced. The cadence
    fires on the step count before it is incremented and the segment's last
    frame is one step later, so the two would otherwise land on adjacent frames
    at every boundary and pay two teacher passes for what is effectively one:
    :class:`~nvalchemi.training.distillation.TeacherLabelHook` passes over a
    cadence dispatch on the step right after a labeled one instead. With
    ``segment_steps`` a multiple of ``label_frequency`` — the default ``100``
    and ``100`` among them — that leaves exactly one label per trajectory per
    segment, on its last frame.

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
    independent need distinct values here rather than a distinct global
    ``torch`` seed, which the sampler's own generator never reads. Distinct is
    not enough on its own, though: because the two are added, consecutive
    values overlap by a shift of one segment — seed ``0``'s second segment
    draws exactly what seed ``1``'s first segment draws — so an ensemble or a
    seed-sensitivity sweep wants values at least as far apart as the number of
    segments a run takes, ``num_steps // steps_per_segment``.

    Any :class:`~nvalchemi.training.distillation.TeacherScorer` may drive
    generation, and a custom one is worth declaring ``label_fields`` on. That
    declaration is what lets
    :class:`~nvalchemi.training.distillation.DistillationStrategy` check the
    generated fields against its ``reference_dataset`` before the first segment
    rather than after it, keeps
    :class:`~nvalchemi.training.distillation.TeacherLabelHook` from re-scoring
    a re-dispatched frame, and promotes a ``teacher_*`` field of the scorer's
    own to a loss target the strategy accepts — generation supplies it, so the
    anchor and any validation data have to carry it as well.

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
        Field(
            description=(
                "Scorer producing the teacher signals for generated frames. A "
                "label_fields declaration on a custom one lets the strategy "
                "check the anchor parity up front and makes a teacher_* field "
                "of its own usable as a loss target."
            )
        ),
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
