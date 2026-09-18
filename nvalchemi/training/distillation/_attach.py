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
"""Attach teacher labels to a batch as ordinary fields at their signal levels.

Shared by the offline path in
:mod:`nvalchemi.training.distillation.labeling`, which attaches labels before
persisting a chunk, and the online path in
:mod:`nvalchemi.training.distillation.strategy`, which attaches them to a live
training batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from nvalchemi.data.level_storage import UniformLevelStorage

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.training.distillation.scoring import SignalLevel, TeacherLabels


def _ensure_system_group(batch: Batch) -> None:
    """Give *batch* an empty, sized system group when it has none.

    A batch of bare positions and atomic numbers carries no system group, and
    :meth:`~nvalchemi.data.Batch.add_key` cannot create one for a built-in level.
    """
    if "system" in batch._storage.groups:
        return
    batch._storage.groups["system"] = UniformLevelStorage(
        data=TensorDict({}, batch_size=[batch.num_graphs], device=batch.device),
        device=batch.device,
        attr_map=batch._storage.attr_map,
        validate=False,
    )


def _split_per_graph(
    batch: Batch, values: torch.Tensor, level: SignalLevel
) -> list[torch.Tensor]:
    """Split a concatenated teacher tensor into one entry per graph."""
    if level == "node":
        return list(torch.split(values, batch.num_nodes_list, dim=0))
    return [values[index : index + 1] for index in range(batch.num_graphs)]


def _attach_teacher_labels(batch: Batch, labels: TeacherLabels) -> None:
    """Attach every teacher label to *batch* at the level its signal declares.

    Existing fields of the same name are overwritten, so re-labeling a batch is
    idempotent.

    Parameters
    ----------
    batch : Batch
        Batch to attach the labels to; mutated in place.
    labels : TeacherLabels
        Mapping from batch field name to ``(tensor, level)`` as returned by
        :meth:`~nvalchemi.training.distillation.TeacherScorer.label`.
    """
    for field, (values, level) in labels.items():
        if level == "system":
            _ensure_system_group(batch)
        batch.add_key(
            field,
            _split_per_graph(batch, values, level),
            level=level,
            overwrite=True,
        )
