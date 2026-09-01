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
"""Restart state of an interrupted on-policy segment loop.

A strategy checkpoint carries model weights, optimizer state, and counters, and
none of those describe where the propagator had got to. This module supplies
the missing half: the live trajectory batch, the propagator's cumulative step
count, and the frames already in the replay buffer, packed as the flat tensor
bundle a checkpoint's hook-state file can hold and unpacked again on the way
back in.

The channel is :class:`~nvalchemi.hooks.CheckpointableHook`, which the
checkpoint layer already snapshots to CPU, writes with the rest of a
checkpoint, and matches back onto a rebuilt strategy by hook class. No
checkpoint-format change is needed, and a run that never generates simply
contributes an empty state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data.batch import Batch
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distillation.scoring import _NEIGHBOR_KEYS

if TYPE_CHECKING:
    from nvalchemi.hooks._context import TrainContext

_LEVEL_BY_GROUP = {"atoms": "atom", "edges": "edge", "system": "system"}
"""Batch storage group name mapped to the level a rebuild classifies fields by."""

_FLAT_SEPARATOR = ":"
"""Separator joining a field's level and name into one flat state-dict key."""


def _batch_state(batch: Batch, *, drop: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Return *batch* as a flat ``{level:field -> tensor}`` bundle.

    Parameters
    ----------
    batch : Batch
        Batch to pack. It is read, never modified.
    drop : frozenset[str], optional
        Field names to leave out. Default an empty set.

    Returns
    -------
    dict[str, Any]
        Segment lengths plus one CPU tensor per stored field, keyed by level
        and name. Suitable for :func:`torch.save` under ``weights_only``.
    """
    state: dict[str, Any] = {
        "num_nodes_list": torch.tensor(batch.num_nodes_list, dtype=torch.long),
        "num_edges_list": torch.tensor(batch.num_edges_list, dtype=torch.long),
    }
    for group_name, group in batch._storage.groups.items():
        level = _LEVEL_BY_GROUP[group_name]
        for key, tensor in group.items():
            if key in drop:
                continue
            state[f"{level}{_FLAT_SEPARATOR}{key}"] = tensor.detach().cpu()
    return state


def _batch_from_state(state: dict[str, Any]) -> Batch:
    """Rebuild the batch :func:`_batch_state` packed.

    Fields are split back into per-graph slices and handed to
    :meth:`~nvalchemi.data.Batch.from_raw_dicts` with the level of every one of
    them named explicitly, so a field the default key sets do not know — a
    ``teacher_*`` label, a dynamics counter — lands at the level it was stored
    at rather than at the raw-dict fallback.

    Parameters
    ----------
    state : dict[str, Any]
        Bundle produced by :func:`_batch_state`.

    Returns
    -------
    Batch
        Batch on the host, equal to the packed one field for field.
    """
    node_counts = [int(count) for count in state["num_nodes_list"]]
    edge_counts = [int(count) for count in state["num_edges_list"]]
    samples: list[dict[str, torch.Tensor]] = [{} for _ in node_counts]
    field_levels: dict[str, str] = {}
    for flat_key, tensor in state.items():
        level, separator, key = flat_key.partition(_FLAT_SEPARATOR)
        if not separator:
            continue
        field_levels[key] = level
        if level == "system":
            for index, sample in enumerate(samples):
                sample[key] = tensor[index : index + 1]
            continue
        counts = node_counts if level == "atom" else edge_counts
        for sample, chunk in zip(samples, torch.split(tensor, counts), strict=True):
            sample[key] = chunk
    return Batch.from_raw_dicts(samples, field_levels=field_levels)


class _OnPolicyRestartHook:
    """Carry the segment loop's propagator and replay state through a checkpoint.

    The hook owns no state of its own: it reads the live trajectory batch, the
    propagator's step count, and the replay buffer off the strategy it is bound
    to when a checkpoint is written, and holds the restored bundle until the
    segment loop consumes it on the way back in. A strategy checkpointed
    outside a run, or before its first segment, contributes an empty bundle and
    restarts by seeding afresh.
    """

    frequency = 1
    stage = TrainingStage.SETUP

    def __init__(self) -> None:
        """Start unbound, with nothing restored."""
        self._strategy: Any = None
        self._restored: dict[str, Any] | None = None

    def prepare_strategy(self, strategy: Any) -> None:
        """Bind the strategy whose on-policy state this hook checkpoints."""
        self._strategy = strategy

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Bind the running strategy, for a run that reached setup by another path."""
        self._strategy = ctx.workflow

    def take(self) -> dict[str, Any] | None:
        """Return the restored bundle once, clearing it.

        Returns
        -------
        dict[str, Any] | None
            Bundle written by :meth:`load_state_dict`, or ``None`` when the run
            starts fresh. Cleared by the call, so a strategy that runs twice
            resumes once and then seeds normally.
        """
        restored, self._restored = self._restored, None
        return restored

    def state_dict(self) -> dict[str, Any]:
        """Return the live trajectory, propagator counter, and replay frames.

        The trajectory's neighbor tensors are left out: they are ephemeral,
        rebuilt from the positions that are stored, and an edge index carried
        across a rebuild would be offset twice.
        """
        strategy = self._strategy
        state = None if strategy is None else strategy._on_policy_state
        if state is None:
            return {}
        buffer = strategy.replay_buffer
        bundle: dict[str, Any] = {
            "dynamics_step_count": torch.tensor(
                strategy.on_policy.dynamics.step_count, dtype=torch.long
            ),
            "md_state": _batch_state(state, drop=_NEIGHBOR_KEYS),
        }
        if buffer is not None and len(buffer) > 0:
            bundle["replay_frames"] = _batch_state(buffer.dataset.in_memory_batch)
        return bundle

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Hold a checkpoint's on-policy bundle for the next :meth:`take`."""
        self._restored = dict(state) or None
