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

from typing import Annotated, Any

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook
from nvalchemi.training.distillation.replay import (
    ReplayEviction,
    _batch_allocation,
    _batch_size_remedy,
)
from nvalchemi.training.distillation.scoring import TeacherScorer
from nvalchemi.training.distillation.seeding import (
    InitialStructures,
    _check_structure_fields,
    _propagator_tree,
)

__all__ = ["OnPolicyConfig", "OnPolicySettings"]


class OnPolicySettings(BaseModel):
    """Declarative settings of one on-policy distillation segment loop.

    Every field is a JSON scalar, so the whole set validates without a
    propagator, a teacher, or a store, and a recipe's settings can be refused
    before a teacher is loaded. :class:`OnPolicyConfig` inherits them and adds
    the live objects the loop drives; whether those objects compose with the
    loop is settled there and in the strategy.

    Parameters
    ----------
    replay_ratio : float
        Fraction of every training batch drawn from the replay buffer.
    training_steps_per_segment : int
        Optimizer steps taken per segment, one per training batch.
    batch_size : int, optional
        Samples per training batch, across both mixture sources. Default ``8``.
    generation_steps : int, optional
        Propagator steps generated per segment. Default ``100``.
    label_frequency : int, optional
        Propagator steps between teacher labelings, on top of each segment's
        last frame. Default ``100``.
    replay_capacity : int | None, optional
        Frame capacity of the replay buffer. Default ``None`` (unbounded).
    replay_eviction : {"fifo", "uncertainty"}, optional
        Eviction policy of the replay buffer. Default ``"fifo"``.
    replay_device : str | None, optional
        Device the replay buffer keeps frames on. Default ``None`` (where the
        reference dataset emits its batches; host memory without one).
    seed : int, optional
        Base seed of every segment's mixture sampler. Default ``0``.
    convergence : float | None, optional
        Max-force-norm threshold below which a generated trajectory counts as
        finished, which turns a relaxation run into a trajectory lifecycle.
        Default ``None`` (no trajectory ends, which is what a
        molecular-dynamics run wants).
    weight_sync_frequency : int, optional
        Segments between weight syncs to the propagator. Default ``1``, the
        only accepted value while the propagator shares the student module.

    Raises
    ------
    ValueError
        If a count or the threshold is not positive, if ``replay_ratio`` falls
        outside ``[0, 1]`` or is exactly ``0``, if the ratio and the batch size
        together round a mixture source out of every batch, if
        ``replay_eviction`` is the reserved ``"uncertainty"``, or if
        ``weight_sync_frequency`` is not ``1``.

    Examples
    --------
    >>> from nvalchemi.training.distillation import OnPolicySettings
    >>> settings = OnPolicySettings(replay_ratio=0.25, training_steps_per_segment=32)
    >>> settings.batch_size
    8

    Notes
    -----
    ``label_frequency`` is the throughput setting, counted against the
    propagator's cumulative ``step_count`` so the cadence does not restart at a
    segment boundary; each segment also labels the frame it ends on, and the
    cadence dispatch adjacent to that forced label is passed over, so
    ``generation_steps`` a multiple of ``label_frequency`` labels each
    trajectory once per segment. ``training_steps_per_segment`` is a budget of
    training batches, which is a budget of optimizer steps only while every
    batch takes one. Size ``replay_capacity`` as a multiple of the trajectory
    count, since FIFO eviction otherwise cuts a segment's contribution mid-step
    and over-represents the back of the batch, and space the ``seed`` of
    replicate runs by at least ``num_steps // training_steps_per_segment``,
    since the sampler adds it to the segment index. ``convergence`` is compared
    against the student's forces, the ones the propagator follows, so the
    criterion is the one the relaxation itself converges on. See
    :ref:`training-distillation-api`.
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
    training_steps_per_segment: Annotated[
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
    generation_steps: Annotated[
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
        str | None,
        Field(
            default=None,
            description=(
                "Device the replay buffer holds frames on, named as a string. "
                "Generated frames reach it from a host-memory sink, so None "
                "stages them where the reference dataset actually emits its "
                "own batches — the mixture is collated before training moves "
                "it — and leaves them in host memory when the run has no "
                "reference dataset. Set it only to override that, and load the "
                "reference dataset there too."
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
    convergence: Annotated[
        float | None,
        Field(
            default=None,
            gt=0.0,
            description=(
                "Max-force-norm threshold below which a generated trajectory "
                "counts as finished, which is what turns a relaxation run into "
                "a lifecycle. None manages no lifecycle: nothing graduates and "
                "nothing is backfilled, which is what a molecular-dynamics run "
                "wants."
            ),
        ),
    ] = None
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

    model_config = ConfigDict(extra="forbid")

    @field_validator("replay_device", mode="before")
    @classmethod
    def _name_replay_device(cls, value: Any) -> Any:
        """Accept a torch.device for a setting every reader names as a string."""
        return str(value) if isinstance(value, torch.device) else value

    @model_validator(mode="after")
    def _validate_replay_eviction(self) -> OnPolicySettings:
        """Hold the reserved eviction policy until committee scoring lands."""
        if self.replay_eviction == "uncertainty":
            raise ValueError(
                "replay_eviction='uncertainty' is reserved for committee-based "
                "frame selection and is not implemented yet; use 'fifo'."
            )
        return self

    @model_validator(mode="after")
    def _validate_weight_sync(self) -> OnPolicySettings:
        """Hold the reserved sync setting at 1 until the decoupled paths land."""
        if self.weight_sync_frequency != 1:
            raise ValueError(
                "weight_sync_frequency must be 1: the propagator holds the same "
                "student module the trainer updates, so an eager run is never out "
                f"of sync; got {self.weight_sync_frequency!r}. Larger values are "
                "reserved for the compiled and asynchronous teacher paths."
            )
        return self

    @model_validator(mode="after")
    def _validate_mixture(self) -> OnPolicySettings:
        """Reject a mixture no batch can actually be drawn from."""
        if self.replay_ratio == 0.0:
            raise ValueError(
                "replay_ratio=0 trains on reference data only, which is "
                "offline distillation paying for generation it never uses; "
                "drop on_policy and call run() with a loader over the labeled "
                "dataset instead."
            )
        reference_samples, replay_samples = _batch_allocation(
            self.replay_ratio, self.batch_size
        )
        if self.replay_ratio >= 1.0 or min(reference_samples, replay_samples) > 0:
            return self
        raise ValueError(
            "The mixture is drawn as whole samples of a batch, so replay_ratio "
            "and batch_size only mean something together; got replay_ratio="
            f"{self.replay_ratio!r} with batch_size={self.batch_size!r}, which "
            f"puts {reference_samples} reference and {replay_samples} generated "
            "samples in every batch and leaves one source out of training "
            f"entirely; {_batch_size_remedy(self.replay_ratio)}."
        )


class OnPolicyConfig(OnPolicySettings):
    """One on-policy distillation segment loop, settings and live objects together.

    A *generation* phase runs the student's own propagator for
    ``generation_steps`` steps, labeling frames with the teacher as it goes; a
    *training* phase then takes ``training_steps_per_segment`` optimizer steps
    on batches mixed from the reference dataset and the replay buffer at
    ``replay_ratio``. The propagator holds the module the trainer updates, so
    each segment generates from a fresher policy than the last. The scalar half
    is :class:`OnPolicySettings`, inherited so a recipe stays flat; :attr:`settings`
    is the detached copy a pre-flight or a restart bundle carries.

    The propagator is any :class:`~nvalchemi.dynamics.base.BaseDynamics`, so a
    relaxation optimizer such as :class:`~nvalchemi.dynamics.optimizers.FIRE`
    drives the loop exactly as a thermostat does. Initial structures must carry
    whatever it updates in place through ``__provides_keys__`` — ``velocities``
    for every shipped propagator, plus a ``cell`` for the variable-cell ones;
    the model outputs of ``__needs_keys__`` are primed before the first step —
    and one row is checked here, so a missing field is a construction error
    rather than a failure on the first step.

    What a relaxation propagator adds is a *trajectory lifecycle*: relaxations
    converge, and a converged structure that keeps being propagated fills the
    replay buffer with near-duplicates of a frame it already holds.
    ``convergence`` turns that lifecycle on. Converged structures freeze, are
    stored once as the minimum they reached, and graduate out of the batch at
    the segment boundary, where the initial structures backfill fresh ones for
    as long as the cursor holds rows —
    :attr:`~nvalchemi.training.distillation.InitialStructures.recycle` restarts
    it rather than letting the batch narrow. Generation ends with the last
    trajectory, and the remaining training steps draw on the buffer already
    filled.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator generating on-policy frames, holding the student module.
    teacher_scorer : TeacherScorer
        Scorer labeling generated frames. Declaring ``label_fields`` on a
        custom one makes the fields it writes knowable up front.
    initial_structures : InitialStructures
        Structures the generated trajectories start from, behind the cursor a
        restart resumes. A bare dataset is accepted and wrapped.
    convergence_hook : ConvergenceHook | None, optional
        Live criterion deciding when a generated trajectory is finished, in
        place of the ``convergence`` threshold. Default ``None``.

    Raises
    ------
    ValueError
        If a setting is out of range, if both ``convergence`` and
        ``convergence_hook`` are set, if a hook passed whole cannot manage the
        lifecycle, if ``initial_structures`` recycles without a criterion to
        backfill for, if a criterion is paired with a multi-sub-stage
        :class:`~nvalchemi.dynamics.FusedStage`, or if the initial structures
        lack a field the propagator opens its step with.

    Examples
    --------
    >>> from nvalchemi.training.distillation import (  # doctest: +SKIP
    ...     InProcessTeacherScorer,
    ...     OnPolicyConfig,
    ...     InitialStructures,
    ... )
    >>> config = OnPolicyConfig(  # doctest: +SKIP
    ...     dynamics=NVTLangevin(student, dt=0.5, temperature=300.0),
    ...     teacher_scorer=InProcessTeacherScorer(teacher, ["energy", "forces"]),
    ...     initial_structures=InitialStructures(dataset),
    ...     replay_ratio=0.25,
    ...     training_steps_per_segment=32,
    ...     batch_size=16,
    ...     generation_steps=50,
    ...     label_frequency=10,
    ...     replay_capacity=8192,
    ... )

    The same loop over relaxation paths, graduating each structure as it
    converges below ``0.05`` and backfilling the next structure in its place:

    >>> config = OnPolicyConfig(  # doctest: +SKIP
    ...     dynamics=FIRE(student, dt=0.1),
    ...     teacher_scorer=InProcessTeacherScorer(teacher, ["energy", "forces"]),
    ...     initial_structures=InitialStructures(dataset, recycle=True),
    ...     convergence=0.05,
    ...     replay_ratio=0.25,
    ...     training_steps_per_segment=32,
    ...     batch_size=16,
    ...     generation_steps=50,
    ...     label_frequency=10,
    ... )

    Notes
    -----
    Any :class:`~nvalchemi.training.distillation.TeacherScorer` may drive
    generation. Declaring ``label_fields`` on a custom one lets
    :class:`~nvalchemi.training.distillation.DistillationStrategy` check the
    generated fields against ``reference_dataset`` at construction and keeps
    :class:`~nvalchemi.training.distillation.TeacherLabelHook` from re-scoring
    a re-dispatched frame; a custom ``teacher_*`` field it writes is an
    ordinary loss target the reference dataset and any validation data must carry too.

    ``convergence`` stays the plain number a recipe can hold, and
    :attr:`convergence_criterion` is the live criterion the lifecycle drives:
    :meth:`~nvalchemi.dynamics.base.ConvergenceHook.from_fmax` with the status
    migration a lifecycle needs, ``source_status=0`` to the propagator's own
    ``exit_status``, built once and handed out by identity thereafter, since
    the lifecycle registers and removes that one object. A criterion that has
    to be a live hook goes to ``convergence_hook`` instead, which no recipe
    describes; the two are refused together because they are two spellings of
    one thing. A hook passed whole must migrate status, off the status ``0``
    the run stamps its structures with, on every step — a criterion that only
    reports convergence would freeze and graduate nothing while looking
    configured, and one that skips steps would let both capture routes store
    the frame it graduates late.

    That criterion also becomes the propagator's convergence detector for the
    duration of the loop, replacing one the propagator was built with and
    restored afterwards, and it has to be the only thing migrating status: a
    second migrating :class:`~nvalchemi.dynamics.base.ConvergenceHook` on the
    propagator would graduate structures at its own threshold, so the lifecycle
    refuses to run beside one. Constructing a
    :class:`~nvalchemi.dynamics.FusedStage` puts a migrator on every non-last
    sub-stage, and on the last one whenever it declares a ``convergence_hook``,
    so a fused propagator is accepted only as a single sub-stage without a
    criterion of its own; a multi-sub-stage one is refused at construction,
    where that shape is fixed.
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
                "check its fields against reference_dataset up front and makes a "
                "teacher_* field of its own usable as a loss target."
            )
        ),
    ]
    initial_structures: Annotated[
        InitialStructures,
        Field(
            description=(
                "Structures the generated trajectories are seeded from, behind "
                "the cursor the initial batch, the backfill, and a restart all "
                "share. A bare dataset is wrapped in an unbudgeted source."
            )
        ),
    ]
    convergence_hook: Annotated[
        ConvergenceHook | None,
        Field(
            default=None,
            description=(
                "Live criterion deciding when a generated trajectory is "
                "finished, in place of the convergence threshold. No recipe "
                "describes it, so it is runtime-only."
            ),
        ),
    ] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _convergence_criterion: ConvergenceHook | None = PrivateAttr(default=None)

    @property
    def settings(self) -> OnPolicySettings:
        """Detached copy of the declarative half, for a recipe or a bundle.

        Returns
        -------
        OnPolicySettings
            The scalars this config carries, validated on their own and holding
            no reference back to the live objects beside them.
        """
        return OnPolicySettings.model_validate(
            {name: getattr(self, name) for name in OnPolicySettings.model_fields}
        )

    @property
    def convergence_criterion(self) -> ConvergenceHook | None:
        """Return the criterion the trajectory lifecycle drives, or ``None``.

        A hook passed whole is that criterion; a ``convergence`` threshold
        stands for one built on first read, migrating ``0`` to the propagator's
        ``exit_status``. The same object is returned for the life of the
        config, because the lifecycle registers it on the propagator and
        removes it again by identity.

        Returns
        -------
        ConvergenceHook | None
            The live criterion, or ``None`` for a run managing no lifecycle.
        """
        if self.convergence_hook is not None:
            return self.convergence_hook
        if self.convergence is None:
            return None
        if self._convergence_criterion is None:
            self._convergence_criterion = ConvergenceHook.from_fmax(
                float(self.convergence),
                source_status=0,
                target_status=self.dynamics.exit_status,
            )
        return self._convergence_criterion

    @model_validator(mode="before")
    @classmethod
    def _coerce_initial_structures(cls, data: Any) -> Any:
        """Wrap a bare dataset in an unbudgeted source."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        structures = data.get("initial_structures")
        if structures is not None and not isinstance(structures, InitialStructures):
            if callable(getattr(structures, "load_batches", None)):
                data["initial_structures"] = InitialStructures(structures)
        return data

    @model_validator(mode="after")
    def _validate_convergence(self) -> OnPolicyConfig:
        """Police a criterion passed whole; the threshold needs no checks."""
        if self.convergence_hook is None:
            return self
        if self.convergence is not None:
            raise ValueError(
                "convergence and convergence_hook are two spellings of one "
                "criterion, so exactly one of them names it; got "
                f"convergence={self.convergence!r} beside a "
                f"{type(self.convergence_hook).__name__}. Drop the threshold to "
                "keep the hook, or drop the hook to keep a config a recipe can "
                "describe."
            )
        exit_status = self.dynamics.exit_status
        migrates = (
            self.convergence_hook.source_status is not None
            and self.convergence_hook.target_status is not None
        )
        if not migrates:
            raise ValueError(
                "The convergence hook of a relaxation loop has to migrate "
                "status, because a graph graduates out of the batch on its "
                "status and freezes in the propagator's step on it; got "
                f"source_status={self.convergence_hook.source_status!r} and "
                f"target_status={self.convergence_hook.target_status!r}. Pass "
                "source_status=0 with "
                f"target_status={exit_status!r}, or pass the fmax threshold "
                "itself as convergence and let the shorthand wire them up."
            )
        if self.convergence_hook.target_status < exit_status:
            raise ValueError(
                "Converged graphs must migrate to at least the propagator's "
                "exit status, which is what graduates them out of the active "
                f"batch; got target_status="
                f"{self.convergence_hook.target_status!r} against "
                f"dynamics.exit_status={exit_status!r}."
            )
        if self.convergence_hook.frequency != 1:
            raise ValueError(
                "The convergence hook of a relaxation loop has to run on every "
                "step, because a structure is captured at the step it converges "
                "and has to be frozen and left out of the path capture on that "
                f"same step; got frequency={self.convergence_hook.frequency!r}, "
                "which would store it by both routes and keep propagating it "
                "until the next firing. Pass frequency=1, or pass the fmax "
                "threshold itself as convergence and let the shorthand wire it "
                "up."
            )
        return self

    @model_validator(mode="after")
    def _validate_lifecycle_shape(self) -> OnPolicyConfig:
        """Reject a lifecycle the structures or the propagator's shape cannot carry."""
        managed = self.convergence is not None or self.convergence_hook is not None
        if self.initial_structures.recycle and not managed:
            raise ValueError(
                "InitialStructures.recycle restarts a backfill that has reached "
                "the end of the rows, and only a run managing a trajectory "
                "lifecycle ever backfills; got it set with convergence=None. "
                "Pass a convergence criterion, or drop the flag."
            )
        if not managed:
            return self
        fused = [
            len(node.sub_stages)
            for node in _propagator_tree(self.dynamics)
            if len(getattr(node, "sub_stages", ())) > 1
        ]
        if fused:
            raise ValueError(
                "The relaxation lifecycle owns graduation for this run, so the "
                "propagator must carry no other status-migrating "
                "ConvergenceHook, and a FusedStage builds one for every "
                "non-last sub-stage as it is constructed; got a stage of "
                f"{fused[0]!r} sub-stages under a convergence criterion. "
                "Generate from a single sub-stage, or drop convergence and let "
                "the propagator manage its own lifecycle."
            )
        return self

    @model_validator(mode="after")
    def _validate_structure_fields(self) -> OnPolicyConfig:
        """Check one row against what the propagator updates in place from its first step."""
        _check_structure_fields(self.initial_structures.probe(), self.dynamics)
        return self
