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
from nvalchemi.dynamics.base import BaseDynamics
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
        Frame capacity of the replay buffer. Default ``None`` (unbounded); see
        the Notes for what an ensemble objective needs here.
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
