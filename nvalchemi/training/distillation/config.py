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

import warnings
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
    SeedSource,
    _check_seed_fields,
    _propagator_tree,
)

__all__ = ["OnPolicyConfig", "OnPolicyKnobs"]


class OnPolicyKnobs(BaseModel):
    """Declarative knobs of one on-policy distillation segment loop.

    Every field here is a JSON scalar, so the whole set validates without a
    propagator, a teacher, or a store: a count out of range, a reserved policy,
    a ratio that rounds a mixture source out of every batch, each of them
    readable off a recipe's text alone and refusable before a teacher is loaded
    onto a device. :class:`OnPolicyConfig` inherits them and adds the live
    objects the loop drives, so a knob is validated identically whether it was
    checked standalone by a pre-flight or composed into the config a run holds.

    That split is also the boundary of what a pre-flight decides. Whether the
    *objects* a recipe names compose with the loop that will drive them — a
    propagator carrying a sampler of its own, a stage whose shape the loop
    cannot chunk into segments, a criterion reading a field no seed carries —
    is settled where the loop is installed and those objects are in hand.
    Cheap to read and cheap to fix belongs here; compatible-with-this-loop
    belongs to :class:`OnPolicyConfig` and to the strategy.

    Parameters
    ----------
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
    replay_device : str | None, optional
        Device the replay buffer keeps frames on, as a string a
        :class:`torch.device` is accepted for. Default ``None`` (wherever the
        reference dataset emits its own batches, and host memory without one).
    seed : int, optional
        Base seed of every segment's mixture sampler. Default ``0``.
    convergence : float | None, optional
        ``fmax`` threshold below which a generated trajectory is finished.
        Default ``None``, which manages no trajectory lifecycle.
    weight_sync_frequency : int, optional
        Segments between pushing student weights to the propagator. Default
        ``1``, currently the only accepted value.

    Raises
    ------
    ValueError
        If a count is not positive, if ``replay_ratio`` falls outside
        ``[0, 1]`` or is exactly ``0``, if the ratio and the batch size
        together round a mixture source out of every batch, if
        ``replay_eviction`` is the reserved ``"uncertainty"``, or if
        ``weight_sync_frequency`` is not ``1``.

    Examples
    --------
    >>> from nvalchemi.training.distillation import OnPolicyKnobs
    >>> knobs = OnPolicyKnobs(replay_ratio=0.25, steps_per_segment=32)
    >>> knobs.batch_size
    8

    Notes
    -----
    ``replay_capacity`` is spent by ``replay_eviction="fifo"`` on whole frames
    in arrival order, and a segment contributes one frame per propagated
    trajectory per labeled step. A capacity that is not a multiple of the
    number of trajectories in the seed batch therefore cuts a segment's
    contribution mid-step. Eviction keeps the newest frames, and a step is
    written in trajectory order, so it is the *back* of the seed batch that
    survives a partial step and ends up represented more often than the front
    in every mixture drawn afterwards. Size it as a multiple of the trajectory
    count to keep the buffer balanced across seeds.

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

    ``weight_sync_frequency`` is reserved and must be ``1`` for now. Eager runs
    need no sync at all — the propagator and the trainer share one module
    object, so an optimizer step is visible to the next generated frame
    immediately — and the knob only becomes meaningful once the propagator
    holds a compiled or remote copy of the student.
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
        """Accept a torch.device for a knob every reader names as a string."""
        return str(value) if isinstance(value, torch.device) else value

    @model_validator(mode="after")
    def _validate_replay_eviction(self) -> OnPolicyKnobs:
        """Hold the reserved eviction policy until committee scoring lands."""
        if self.replay_eviction == "uncertainty":
            raise ValueError(
                "replay_eviction='uncertainty' is reserved for committee-based "
                "frame selection and is not implemented yet; use 'fifo'."
            )
        return self

    @model_validator(mode="after")
    def _validate_weight_sync(self) -> OnPolicyKnobs:
        """Hold the reserved sync knob at 1 until the decoupled paths land."""
        if self.weight_sync_frequency != 1:
            raise ValueError(
                "weight_sync_frequency must be 1: the propagator holds the same "
                "student module the trainer updates, so an eager run is never out "
                f"of sync; got {self.weight_sync_frequency!r}. Larger values are "
                "reserved for the compiled and asynchronous teacher paths."
            )
        return self

    @model_validator(mode="after")
    def _validate_mixture(self) -> OnPolicyKnobs:
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


class OnPolicyConfig(OnPolicyKnobs):
    """One on-policy distillation segment loop, knobs and live objects together.

    On-policy distillation alternates two phases. A *generation* phase runs the
    student's own propagator for ``segment_steps`` steps from the seeded state,
    labeling frames with the teacher as it goes; a *training* phase then takes
    ``steps_per_segment`` optimizer steps on batches mixed from the reference
    dataset and the replay buffer at ``replay_ratio``. The student the
    propagator holds is the module the trainer updates, so each segment
    generates from a fresher policy than the last.

    The scalar half is :class:`OnPolicyKnobs`, inherited rather than nested so
    that every knob keeps its own name here and a recipe stays flat; read
    :attr:`knobs` for the detached copy a pre-flight or a restart bundle
    carries. What this class adds is the four live objects the loop drives, and
    the checks that need them: whether the seed structures carry what the
    propagator reads, and whether a convergence criterion can manage the
    lifecycle it is being asked to.

    The propagator is deliberately typed as
    :class:`~nvalchemi.dynamics.base.BaseDynamics` and named ``dynamics``, not
    ``integrator``: a relaxation optimizer such as
    :class:`~nvalchemi.dynamics.optimizers.FIRE` drives the loop exactly as a
    thermostat does, and nothing downstream of this config reads a velocity or
    a temperature. Seed structures must carry the batch fields the chosen
    propagator reads before its first force evaluation: the fields its
    ``__needs_keys__`` model outputs are written back into — ``forces`` for
    every shipped integrator and optimizer, plus ``stress`` for the
    variable-cell ones (:class:`~nvalchemi.dynamics.integrators.NPT`,
    :class:`~nvalchemi.dynamics.integrators.NPH`,
    :class:`~nvalchemi.dynamics.optimizers.FIREVariableCell`) — and the state it
    updates in place, which is ``velocities`` and ``atomic_masses`` for the
    integrators and the optimizers alike and ``cell`` on top of those for the
    variable-cell ones. One seed row is loaded here to check that, so a missing
    field is a construction error rather than ``'Batch' object has no attribute
    'forces'`` on the propagator's first step.

    What a relaxation propagator adds is a *trajectory lifecycle*: relaxations
    converge, and a converged structure that keeps being propagated fills the
    replay buffer with near-duplicates of a frame the buffer already holds.
    ``convergence`` turns that lifecycle on. Converged structures freeze, are
    stored once as the minimum they reached, and graduate out of the batch at
    the segment boundary, where the seed source backfills fresh ones in their
    place for as long as its cursor still holds rows —
    :attr:`~nvalchemi.training.distillation.SeedSource.recycle` restarts that
    cursor rather than letting the batch narrow. Generation ends with the last
    trajectory, and the remaining training steps draw on the buffer already
    filled.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator generating on-policy frames, holding the student module.
    teacher_scorer : TeacherScorer
        Scorer labeling generated frames. Declaring ``label_fields`` on a
        custom one is what makes the fields it writes knowable up front.
    seeds : SeedSource
        Structures the generated trajectories start from, behind the cursor a
        backfill and a restart share. A bare dataset is accepted and wrapped.
    convergence_hook : ConvergenceHook | None, optional
        Criterion deciding when a generated trajectory is finished, passed
        whole instead of as the ``convergence`` threshold. Default ``None``.
        It has to migrate status, off the status the seeds enter on, and run on
        every step. Live objects are not describable in a recipe, so a run that
        wants to stay serializable passes ``convergence`` instead.

    Raises
    ------
    ValueError
        If a knob is out of range, if both ``convergence`` and
        ``convergence_hook`` are set, if a hook passed whole cannot manage the
        lifecycle, if ``seeds`` recycles without a criterion to backfill for,
        if a criterion is paired with a multi-sub-stage
        :class:`~nvalchemi.dynamics.FusedStage`, or if the seed structures lack
        a field the propagator opens its step with.

    Examples
    --------
    >>> from nvalchemi.training.distillation import (  # doctest: +SKIP
    ...     InProcessTeacherScorer,
    ...     OnPolicyConfig,
    ...     SeedSource,
    ... )
    >>> config = OnPolicyConfig(  # doctest: +SKIP
    ...     dynamics=NVTLangevin(student, dt=0.5, temperature=300.0),
    ...     teacher_scorer=InProcessTeacherScorer(teacher, ["energy", "forces"]),
    ...     seeds=SeedSource(seed_dataset),
    ...     replay_ratio=0.25,
    ...     steps_per_segment=32,
    ...     batch_size=16,
    ...     segment_steps=50,
    ...     label_frequency=10,
    ...     replay_capacity=8192,
    ... )

    The same loop over relaxation paths, graduating each structure as it
    converges below ``0.05`` and backfilling the next seed in its place:

    >>> config = OnPolicyConfig(  # doctest: +SKIP
    ...     dynamics=FIRE(student, dt=0.1),
    ...     teacher_scorer=InProcessTeacherScorer(teacher, ["energy", "forces"]),
    ...     seeds=SeedSource(seed_dataset, recycle=True),
    ...     convergence=0.05,
    ...     replay_ratio=0.25,
    ...     steps_per_segment=32,
    ...     batch_size=16,
    ...     segment_steps=50,
    ...     label_frequency=10,
    ... )

    Notes
    -----
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

    ``convergence`` stays the plain number a recipe can hold, and
    :attr:`convergence_criterion` is the live criterion the lifecycle drives:
    the threshold becomes
    :meth:`~nvalchemi.dynamics.base.ConvergenceHook.from_fmax` with the status
    migration a lifecycle needs — ``source_status=0`` to the propagator's own
    ``exit_status`` — built once and handed out by identity thereafter, since
    the lifecycle registers and removes that one object. A criterion that has
    to be a live hook goes to ``convergence_hook`` instead, which is bound to
    one propagator and describable in no recipe, and the two are refused
    together because they are two spellings of one thing.

    A hook passed whole must already carry the migration, because a criterion
    that only reports convergence would freeze nothing and graduate nothing
    while looking configured, and it must migrate off the status the seeds
    enter on, which the run stamps itself. It must also run on every step: a
    structure is captured on the step it converges, and it has to be frozen and
    left out of that step's path capture for the two capture routes to
    partition a segment's frames. The threshold is compared against the
    student's forces, which are the forces the propagator is following, so the
    criterion is exactly the one the relaxation itself converges on.

    That criterion also becomes the propagator's convergence detector for the
    duration of the loop, so a ``convergence_hook`` the propagator was built
    with is replaced on the way in and restored on the way out — a run
    configured with both relaxes to the threshold named here, not to the
    propagator's own. It has to be the *only* thing migrating status, though: a
    second migrating :class:`~nvalchemi.dynamics.base.ConvergenceHook` already
    on the propagator would graduate structures at its own threshold, and the
    lifecycle refuses to run alongside one — including one the caller never
    registered. Constructing a :class:`~nvalchemi.dynamics.FusedStage` puts a
    migrator on every non-last sub-stage, and on the last one whenever it
    declares a ``convergence_hook``, so a fused propagator is accepted here
    only in the one shape that carries none: a single sub-stage with no
    criterion of its own. A multi-sub-stage one is refused at construction,
    because its sub-stages migrate status themselves and would step the batch
    through the codes the configured criterion is trying to graduate off, and
    that shape is fixed the moment the stage is built. The lifecycle assumes a
    single-status propagator either way, migrating ``0`` to ``exit_status`` in
    one hop.

    Distribution-matching and path objectives are defined on equilibrium
    ensembles, and a relaxation path is not one: those objectives will be
    rejected when paired with a relaxation propagator once they land. Energy,
    force, and per-atom energy matching are pointwise and distill a relaxation
    path exactly as they distill a trajectory.

    The pre-``SeedSource`` spellings — ``seed_dataset``, ``sampler``,
    ``recycle_seeds``, and a hook-valued ``convergence`` — are still accepted
    and mapped onto the new shape with a :class:`DeprecationWarning`, so a
    caller migrating from them keeps running while the call sites move.
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
    seeds: Annotated[
        SeedSource,
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
    def knobs(self) -> OnPolicyKnobs:
        """Detached copy of the declarative half, for a recipe or a bundle.

        Returns
        -------
        OnPolicyKnobs
            The scalars this config carries, validated on their own and holding
            no reference back to the live objects beside them.
        """
        return OnPolicyKnobs.model_validate(
            {name: getattr(self, name) for name in OnPolicyKnobs.model_fields}
        )

    @property
    def convergence_criterion(self) -> ConvergenceHook | None:
        """Return the criterion the trajectory lifecycle drives, or ``None``.

        A hook passed whole is that criterion; a ``convergence`` threshold
        stands for one built on first read, migrating ``0`` to the
        propagator's ``exit_status``. Either way the same object is returned
        for the life of the config, because the lifecycle registers it on the
        propagator and removes it again by identity.

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
    def _coerce_seeds(cls, data: Any) -> Any:
        """Wrap a bare dataset, and map the pre-SeedSource spellings onto seeds."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("seed_dataset") is not None and data.get("sampler") is not None:
            raise ValueError(
                "Exactly one of seed_dataset or sampler must be set: a sampler "
                "builds the initial batch from its own dataset under its own "
                "size budget, so a seed_dataset alongside it would never be "
                "read. Got ['seed_dataset', 'sampler']."
            )
        legacy_dataset = data.pop("seed_dataset", None)
        legacy_sampler = data.pop("sampler", None)
        if legacy_dataset is not None:
            warnings.warn(
                "OnPolicyConfig.seed_dataset is now OnPolicyConfig.seeds, a "
                "SeedSource holding the cursor a backfill and a restart share; "
                "wrapping the dataset in an unbudgeted source. Pass "
                "seeds=SeedSource(dataset) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["seeds"] = SeedSource(legacy_dataset)
        elif legacy_sampler is not None:
            data["seeds"] = SeedSource.from_sampler(legacy_sampler)
        seeds = data.get("seeds")
        if seeds is not None and not isinstance(seeds, SeedSource):
            if callable(getattr(seeds, "load_batches", None)):
                data["seeds"] = SeedSource(seeds)
        if "recycle_seeds" in data:
            recycle = data.pop("recycle_seeds")
            warnings.warn(
                "OnPolicyConfig.recycle_seeds is now SeedSource.recycle, which "
                "is where the cursor that wraps lives; setting it on the "
                "source. Pass seeds=SeedSource(dataset, recycle=True) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if isinstance(data.get("seeds"), SeedSource):
                data["seeds"].recycle = bool(recycle)
        if isinstance(data.get("convergence"), ConvergenceHook):
            warnings.warn(
                "OnPolicyConfig.convergence is the fmax threshold now, and a "
                "criterion passed whole belongs to convergence_hook, which no "
                "recipe describes; moving it there. Pass "
                "convergence_hook=hook instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["convergence_hook"] = data.pop("convergence")
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
        """Reject a lifecycle the seeds or the propagator's shape cannot carry."""
        managed = self.convergence is not None or self.convergence_hook is not None
        if self.seeds.recycle and not managed:
            raise ValueError(
                "SeedSource.recycle restarts a backfill that has reached the "
                "end of the seed rows, and only a run managing a trajectory "
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
    def _validate_seed_fields(self) -> OnPolicyConfig:
        """Check one seed row against what the propagator reads before its first step."""
        _check_seed_fields(self.seeds.probe(), self.dynamics)
        return self
