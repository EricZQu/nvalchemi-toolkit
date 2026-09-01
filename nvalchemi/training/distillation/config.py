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

from typing import Annotated

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.training.distillation.replay import ReplayEviction
from nvalchemi.training.distillation.scoring import TeacherScorer

__all__ = ["OnPolicyConfig"]


class OnPolicyConfig(BaseModel):
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
    a temperature. Seed structures must still carry the batch fields the chosen
    propagator reads before its first force evaluation: the fields its
    ``__needs_keys__`` model outputs are written back into — ``forces``, and
    ``stress`` as well for a variable-cell propagator — and the state it updates
    in place, which is ``velocities`` and ``atomic_masses`` for the integrators
    and the optimizers alike and ``cell`` on top of those for the variable-cell
    ones.

    What a relaxation propagator adds is a *trajectory lifecycle*: relaxations
    converge, and a converged structure that keeps being propagated fills the
    replay buffer with near-duplicates of a frame the buffer already holds.
    ``convergence`` turns that lifecycle on. Converged structures freeze, are
    stored once as the minimum they reached, and graduate out of the batch at
    the segment boundary. Whether anything takes their slot depends on the seed
    source: a ``sampler`` backfills from its own dataset, while a
    ``seed_dataset`` is propagated whole, so the batch simply narrows unless
    ``recycle_seeds`` restarts the dataset from the beginning. Generation ends
    with the last trajectory, and the remaining training steps draw on the
    buffer already filled.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator generating on-policy frames, holding the student module.
    teacher_scorer : TeacherScorer
        Scorer labeling generated frames.
    seed_dataset : BatchDatasetProtocol | None, optional
        Structures the generated trajectories start from, propagated as one
        batch that consumes every one of them. Default ``None``, which requires
        a ``sampler`` instead.
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
    convergence : ConvergenceHook | float | None, optional
        Convergence criterion driving the trajectory lifecycle of a relaxation
        run. A float is the ``fmax`` shorthand. Default ``None`` (no lifecycle
        management).
    recycle_seeds : bool, optional
        Whether a backfill restarts at the beginning of ``seed_dataset``, which
        the initial batch consumed whole. Default ``False``, which lets the
        batch narrow as its trajectories converge.
    weight_sync_frequency : int, optional
        Segments between pushing student weights to the propagator. Default
        ``1``, currently the only accepted value.

    Raises
    ------
    ValueError
        If a count is not positive, if ``replay_ratio`` falls outside
        ``[0, 1]``, if neither or both of ``seed_dataset`` and ``sampler`` are
        given, if ``replay_eviction`` is the reserved ``"uncertainty"``, if
        ``convergence`` is a hook that migrates no status, migrates below the
        propagator's ``exit_status``, or runs on anything but every step, or if
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

    The same loop over relaxation paths, graduating each structure as it
    converges below ``0.05`` and backfilling the next seed in its place:

    >>> config = OnPolicyConfig(  # doctest: +SKIP
    ...     dynamics=FIRE(student, dt=0.1),
    ...     teacher_scorer=InProcessTeacherScorer(teacher, ["energy", "forces"]),
    ...     seed_dataset=seed_dataset,
    ...     convergence=0.05,
    ...     recycle_seeds=True,
    ...     replay_ratio=0.25,
    ...     steps_per_segment=32,
    ...     batch_size=16,
    ...     segment_steps=50,
    ...     label_frequency=10,
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

    ``convergence`` is resolved once, here: a float becomes
    :meth:`~nvalchemi.dynamics.base.ConvergenceHook.from_fmax` with the status
    migration a lifecycle needs — ``source_status=0`` to the propagator's own
    ``exit_status`` — and a hook passed whole must already carry it, because a
    criterion that only reports convergence would freeze nothing and graduate
    nothing while looking configured. It must also run on every step: a
    structure is captured on the step it converges, and it has to be frozen and
    left out of that step's path capture for the two capture routes to
    partition a segment's frames. The threshold is compared against the
    student's forces, which are the forces the propagator is following, so the
    criterion is exactly the one the relaxation itself converges on.

    The resolved criterion also becomes the propagator's convergence detector
    for the duration of the loop, so a ``convergence_hook`` the propagator was
    built with is replaced on the way in and restored on the way out — a run
    configured with both relaxes to the threshold named here, not to the
    propagator's own. And because the criterion migrates ``0`` to
    ``exit_status`` in one hop, the lifecycle assumes a single-status
    propagator: on a :class:`~nvalchemi.dynamics.FusedStage`, whose
    ``exit_status`` is one past the last sub-stage code, a structure that meets
    the criterion while still in the first sub-stage graduates out of the batch
    without ever reaching the others.

    Distribution-matching and path objectives are defined on equilibrium
    ensembles, and a relaxation path is not one: those objectives will be
    rejected when paired with a relaxation propagator once they land. Energy,
    force, and per-atom energy matching are pointwise and distill a relaxation
    path exactly as they distill a trajectory.

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
                "propagated as one batch that consumes every one of them, so a "
                "lifecycle run backfills only under recycle_seeds or from a "
                "sampler. Mutually exclusive with sampler."
            ),
        ),
    ] = None
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
    sampler: Annotated[
        SizeAwareSampler | None,
        Field(
            default=None,
            description=(
                "Size-aware sampler bin-packing the initial batch under its own "
                "size budget, in place of seed_dataset. Without a convergence "
                "criterion it seeds the run and nothing more; with one it also "
                "serves the backfill, under its own budget rather than the "
                "seeded batch's."
            ),
        ),
    ] = None
    convergence: Annotated[
        ConvergenceHook | float | None,
        Field(
            default=None,
            description=(
                "Criterion deciding when a generated trajectory is finished. A "
                "float is the fmax shorthand for a max-force-norm hook; a hook "
                "passed whole has to migrate status and run on every step, and "
                "stands in for the propagator's own criterion until the run "
                "ends. It migrates to exit_status in one hop, so a fused stage "
                "graduates past its remaining sub-stages. None manages no "
                "lifecycle: nothing graduates and nothing is backfilled, which "
                "is what a molecular-dynamics run wants."
            ),
        ),
    ] = None
    recycle_seeds: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Whether a backfill restarts at the beginning of the seed "
                "dataset, which the initial batch consumed whole. False lets "
                "the batch narrow as its trajectories converge, then trains "
                "the remaining steps on the buffer already filled."
            ),
        ),
    ] = False
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

    @model_validator(mode="after")
    def _validate_recycle_seeds(self) -> OnPolicyConfig:
        """Reject seed recycling on a run that never backfills from the dataset."""
        if not self.recycle_seeds:
            return self
        if self.convergence is None:
            raise ValueError(
                "recycle_seeds restarts a backfill that has reached the end of "
                "seed_dataset, and only a run managing a trajectory lifecycle "
                "ever backfills; got it set with convergence=None. Pass a "
                "convergence criterion, or drop the flag."
            )
        if self.sampler is not None:
            raise ValueError(
                "recycle_seeds restarts seed_dataset, while a sampler serves "
                "the backfill from its own dataset and consumes each structure "
                "once; got both. Drop the flag, or seed from a seed_dataset."
            )
        return self

    @model_validator(mode="after")
    def _resolve_convergence(self) -> OnPolicyConfig:
        """Turn the fmax shorthand into a hook, and police the one passed whole."""
        if self.convergence is None:
            return self
        exit_status = self.dynamics.exit_status
        if not isinstance(self.convergence, ConvergenceHook):
            self.convergence = ConvergenceHook.from_fmax(
                float(self.convergence), source_status=0, target_status=exit_status
            )
            return self
        migrates = (
            self.convergence.source_status is not None
            and self.convergence.target_status is not None
        )
        if not migrates:
            raise ValueError(
                "The convergence hook of a relaxation loop has to migrate "
                "status, because a graph graduates out of the batch on its "
                "status and freezes in the propagator's step on it; got "
                f"source_status={self.convergence.source_status!r} and "
                f"target_status={self.convergence.target_status!r}. Pass "
                "source_status=0 with "
                f"target_status={exit_status!r}, or pass the fmax threshold "
                "itself and let the shorthand wire them up."
            )
        if self.convergence.target_status < exit_status:
            raise ValueError(
                "Converged graphs must migrate to at least the propagator's "
                "exit status, which is what graduates them out of the active "
                f"batch; got target_status={self.convergence.target_status!r} "
                f"against dynamics.exit_status={exit_status!r}."
            )
        if self.convergence.frequency != 1:
            raise ValueError(
                "The convergence hook of a relaxation loop has to run on every "
                "step, because a structure is captured at the step it converges "
                "and has to be frozen and left out of the path capture on that "
                f"same step; got frequency={self.convergence.frequency!r}, "
                "which would store it by both routes and keep propagating it "
                "until the next firing. Pass frequency=1, or pass the fmax "
                "threshold itself and let the shorthand wire it up."
            )
        return self

    @model_validator(mode="after")
    def _validate_replay_eviction(self) -> OnPolicyConfig:
        """Hold the reserved eviction policy until committee scoring lands."""
        if self.replay_eviction == "uncertainty":
            raise ValueError(
                "replay_eviction='uncertainty' is reserved for committee-based "
                "frame selection and is not implemented yet; use 'fifo'."
            )
        return self

    @model_validator(mode="after")
    def _validate_weight_sync(self) -> OnPolicyConfig:
        """Hold the reserved sync knob at 1 until the decoupled paths land."""
        if self.weight_sync_frequency != 1:
            raise ValueError(
                "weight_sync_frequency must be 1: the propagator holds the same "
                "student module the trainer updates, so an eager run is never out "
                f"of sync; got {self.weight_sync_frequency!r}. Larger values are "
                "reserved for the compiled and asynchronous teacher paths."
            )
        return self
