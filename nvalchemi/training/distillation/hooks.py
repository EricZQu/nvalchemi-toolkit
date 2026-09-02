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
"""Dynamics hook that labels on-policy frames as a propagator produces them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.training.distillation._labels import (
    _attach_teacher_labels,
    _prune_empty_edges,
    _reject_foreign_fields,
)
from nvalchemi.training.distillation.scoring import _NEIGHBOR_KEYS, scorer_fields

if TYPE_CHECKING:
    from enum import Enum

    from nvalchemi.data import Batch
    from nvalchemi.data.level_storage import BaseLevelStorage
    from nvalchemi.dynamics.sinks import DataSink
    from nvalchemi.hooks._context import DynamicsContext
    from nvalchemi.training.distillation.scoring import TeacherScorer

__all__ = ["TeacherLabelHook"]

_PREDICTION_KEYS = frozenset(BaseDynamics._OUTPUT_KEY_TO_BATCH_ATTR.values())
"""Batch fields a propagator overwrites with the propagated model's predictions."""


def _run_local_keys() -> frozenset[str]:
    """Return the fields of a live frame that mean nothing outside its run.

    Read at call time rather than at import time, because
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.register_bookkeeping_key` grows
    the bookkeeping registry as stages are built — a fused stage registers one
    step counter per sub-stage.
    """
    return _NEIGHBOR_KEYS | _PREDICTION_KEYS | frozenset(BaseDynamics._bookkeeping_keys)


class TeacherLabelHook:
    """Label the live propagator frame with teacher signals, inline.

    This is the inline half of on-policy labeling: the hook fires at
    :attr:`~nvalchemi.dynamics.base.DynamicsStage.AFTER_STEP`, once the
    propagator has fully resolved the step, and attaches every signal its
    scorer produces to the very :class:`~nvalchemi.data.Batch` being
    propagated. Each signal lands at the level it declares, through
    :meth:`~nvalchemi.data.Batch.add_key`: a per-atom field written as a plain
    attribute would be routed to the system group and left out of the level
    tracking that the Zarr writer and the loss target lookup both read.

    The propagator is left alone. Teacher signals populate ``teacher_*`` fields
    only, so the ``energy`` and ``forces`` the student wrote during
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` — the values driving
    the next step — are never overwritten, and a scorer that declares, or
    returns, a field outside that namespace is rejected — at construction and
    at labeling respectively — rather than allowed to clobber propagator
    state. Nothing here assumes molecular dynamics: a relaxation optimizer
    such as :class:`~nvalchemi.dynamics.optimizers.FIRE` is labeled the same
    way, at the same stage.

    A ``sink`` additionally mirrors every labeled frame into a
    :class:`~nvalchemi.dynamics.sinks.DataSink`, so a segment's trajectory can
    be drained into a replay buffer or a store after the run. What reaches the
    sink is a training sample rather than a propagator state: a copy stripped
    of the ephemeral neighbor tensors — whose neighbor dimension changes
    between adaptive rebuilds and would break concatenation — of the dynamics
    bookkeeping fields (``status``, ``system_id``, the per-status step
    counters) that are meaningless outside the run that wrote them, and of the
    ``energy``, ``forces``, and ``stress`` that
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` overwrote with the
    propagated model's own predictions. Keeping those last three would store
    the student's self-labels under the names a reference target uses, which a
    mixed training batch cannot tell apart from ground truth. The live batch
    keeps all of it.

    Parameters
    ----------
    teacher_scorer : TeacherScorer
        Scorer producing the teacher signals for each labeled frame. A scorer
        publishing ``label_fields`` makes the idempotency check below exact
        from the first dispatch.
    sink : DataSink | None, optional
        Sink each labeled frame is copied into. Default ``None`` (label the
        live batch and write nothing).
    frequency : int, optional
        Label every ``frequency`` steps; the dynamics registry does the
        gating. Default ``1`` (every step).

    Attributes
    ----------
    teacher_scorer : TeacherScorer
        The scorer signals are requested from.
    sink : DataSink | None
        The sink labeled frames are copied into, if any.
    frequency : int
        Labeling frequency in steps.
    stage : DynamicsStage
        Fixed to ``AFTER_STEP``.

    Raises
    ------
    ValueError
        If the scorer declares, or returns, a field outside the ``teacher_*``
        namespace.

    See Also
    --------
    nvalchemi.dynamics.hooks.SnapshotHook : Capture frames without labeling them.
    nvalchemi.training.distillation.label_dataset : Label a dataset offline.

    Examples
    --------
    >>> from nvalchemi.dynamics.sinks import HostMemory
    >>> from nvalchemi.training.distillation import (
    ...     InProcessTeacherScorer,
    ...     TeacherLabelHook,
    ... )
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", "forces"])  # doctest: +SKIP
    >>> sink = HostMemory(capacity=10_000)  # doctest: +SKIP
    >>> hook = TeacherLabelHook(scorer, sink=sink, frequency=10)  # doctest: +SKIP
    >>> dynamics.register_hook(hook)  # doctest: +SKIP

    Notes
    -----
    Do not confuse this hook with the labeling seam inside
    :class:`~nvalchemi.training.distillation.DistillationStrategy`, which is a
    private ``TrainingStage.BEFORE_FORWARD`` hook: that one labels a batch the
    *trainer* is about to run a forward pass on, this one labels a frame the
    *propagator* just produced. They are different stage enums on different
    engines and both can be active in one on-policy run.

    Labeling is idempotent per step: a frame already carrying every field the
    scorer writes, at the step it was labeled on, is passed over, so
    dispatching the hook twice on one state — re-entering a chunked
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.run`, or force-labeling a
    segment's last frame — never pays for a second teacher pass and never
    duplicates it into the sink. The step count is what makes the check safe on
    a live batch, whose ``teacher_*`` fields stay attached while the positions
    underneath them move. Which fields those are comes from
    :func:`~nvalchemi.training.distillation.scorer_fields` — a ``label_fields``
    declaration, or the built-in signals behind a scorer's signal names — and
    from the first pass's own labels for a scorer publishing neither, so a
    scorer with a signal name of its own is re-scored exactly once, on the
    dispatch that reveals what it writes, and skipped on every re-dispatch
    after that. The sink write is gated on the step count alone, so no frame is
    stored twice whatever the scorer declares.

    The teacher runs with autocast disabled whatever precision context the
    propagator establishes, so a frame labeled inside a mixed-precision
    generation phase carries exactly the labels a full-precision one would, and
    matches what :func:`~nvalchemi.training.distillation.label_dataset` would
    have written offline.

    ``requires_grad`` hygiene is the scorer's contract, not this hook's:
    :meth:`~nvalchemi.training.distillation.TeacherScorer.label` snapshots the
    flags of ``positions`` and the teacher's autograd inputs and restores them
    before returning, which leaves the batch exactly as
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` left it — flags
    cleared — so the next step's in-place updates stay legal.
    """

    def __init__(
        self,
        teacher_scorer: TeacherScorer,
        sink: DataSink | None = None,
        frequency: int = 1,
    ) -> None:
        """Resolve the fields the scorer populates, when they can be known."""
        self.teacher_scorer = teacher_scorer
        self.sink = sink
        self.frequency = frequency
        self.stage = DynamicsStage.AFTER_STEP
        self._teacher_fields: tuple[str, ...] | None = scorer_fields(teacher_scorer)
        if self._teacher_fields is not None:
            _reject_foreign_fields(self._teacher_fields, "A scorer's label_fields")
        self._labeled_step: int | None = None

    @torch.compiler.disable
    def _label_frame(self, batch: Batch, step_count: int) -> None:
        """Label *batch* unless it was already labeled at *step_count*."""
        stored = step_count == self._labeled_step
        if (
            stored
            and self._teacher_fields is not None
            and all(field in batch for field in self._teacher_fields)
        ):
            return
        with torch.autocast(device_type=batch.device.type, enabled=False):
            labels = self.teacher_scorer.label(batch)
        _reject_foreign_fields(labels, "Teacher labels")
        _attach_teacher_labels(batch, labels)
        if self._teacher_fields is None:
            self._teacher_fields = tuple(sorted(labels))
        self._labeled_step = step_count
        if self.sink is not None and not stored:
            self.sink.write(self._captured_frame(batch))

    def _captured_frame(self, batch: Batch) -> Batch:
        """Return a labeled copy of *batch* holding nothing run-local.

        The dropped fields leave the live batch only for the duration of the
        copy and are put back before returning, so the next step still finds
        the neighbor tensors and the predictions it reuses. Cloning first and
        deleting afterwards would be simpler, but it would allocate a full
        copy of the neighbor list — usually the largest tensor in a frame — on
        the propagation device just to throw it away. A whole-frame
        :meth:`~nvalchemi.data.Batch.clone` is still the operation wanted here
        rather than the partial :meth:`~nvalchemi.data.Batch.index_select` that
        :class:`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook` uses to pick
        converged graphs out of a frame. An edge group emptied by dropping the
        neighbor list is removed as well, so a store does not record edges that
        no array backs.
        """
        dropped = _run_local_keys()
        detached: list[tuple[BaseLevelStorage, str, torch.Tensor]] = []
        try:
            for group in batch._storage.groups.values():
                for key in [name for name in group.keys() if name in dropped]:
                    detached.append((group, key, group[key]))
                    del group[key]
            frame = batch.clone()
        finally:
            for group, key, tensor in detached:
                group[key] = tensor
        if frame.keys is not None:
            for names in frame.keys.values():
                names -= dropped
        _prune_empty_edges(frame)
        return frame

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:  # noqa: ARG002
        """Label the frame the propagator has just resolved."""
        self._label_frame(ctx.batch, ctx.step_count)
