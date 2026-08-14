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
"""Offline labeling of a dataset with teacher signals."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
)
from nvalchemi.data.level_storage import UniformLevelStorage

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.data.datapipes.backends.zarr import StoreLike
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
    from nvalchemi.training.distillation.scoring import TeacherScorer

__all__ = ["label_dataset"]


_DENSE_NEIGHBOR_KEYS = frozenset(
    {"neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"}
)
"""Node-level neighbor tensors whose neighbor dimension varies between chunks."""


@dataclasses.dataclass(frozen=True)
class _StoreState:
    """Sample counts and field names of an existing labeled store."""

    active: int
    total: int
    fields: frozenset[str]


def _existing_store_state(store: StoreLike) -> _StoreState | None:
    """Return the state of *store*, or ``None`` when it cannot be read."""
    try:
        reader = AtomicDataZarrReader(store)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    try:
        return _StoreState(
            active=len(reader),
            total=int(reader._samples_mask.numel()),
            fields=frozenset(reader.field_levels),
        )
    finally:
        reader.close()


def _ensure_system_group(batch: Batch) -> None:
    """Give *batch* a system group so system-level fields can be attached.

    A batch built from samples that carry no system-level field at all — bare
    positions and atomic numbers, say — has no system group, and
    :meth:`~nvalchemi.data.Batch.add_key` cannot create one.  The group is
    materialized empty but sized, the form
    :class:`~nvalchemi.data.level_storage.BaseLevelStorage` accepts for exactly
    this case.
    """
    if "system" in batch._storage.groups:
        return
    batch._storage.groups["system"] = UniformLevelStorage(
        data=TensorDict({}, batch_size=[batch.num_graphs], device=batch.device),
        device=batch.device,
        attr_map=batch._storage.attr_map,
        validate=False,
    )


def _persisted_fields(batch: Batch) -> frozenset[str]:
    """Return the field names a writer would persist for *batch*."""
    tracked: set[str] = set()
    for names in (batch.keys or {}).values():
        tracked |= names
    return frozenset(name for name in tracked if name in batch)


def _check_store_schema(stored: frozenset[str], outgoing: frozenset[str]) -> None:
    """Raise when a resumed chunk would write a different field set than the store.

    ``AtomicDataZarrWriter.append`` extends only the arrays a store already
    holds and silently ignores everything else, so a mismatched resume would
    leave arrays at different lengths rather than fail.
    """
    if stored == outgoing:
        return
    raise ValueError(
        "Resumed labeling must write the same fields the store already holds; got "
        f"extra {sorted(outgoing - stored)!r} and missing {sorted(stored - outgoing)!r}."
    )


def _split_per_graph(
    batch: Batch, values: torch.Tensor, level: str
) -> list[torch.Tensor]:
    """Split a concatenated teacher tensor into one entry per graph."""
    if level == "node":
        return list(torch.split(values, batch.num_nodes_list, dim=0))
    return [values[index : index + 1] for index in range(batch.num_graphs)]


def _strip_unstorable(batch: Batch, keep: frozenset[str]) -> None:
    """Drop fields that must not reach the store, keeping *keep* intact.

    Removes the dense neighbor tensors, whose neighbor dimension changes
    between rebuilds and so cannot append into a fixed-width store array, plus
    anything else that appeared on *batch* during labeling.  An edge group left
    with no fields is dropped as well, so the store's edge pointers do not
    record edges that no array backs.
    """
    for key in _DENSE_NEIGHBOR_KEYS | (_persisted_fields(batch) - keep):
        if key in batch:
            del batch[key]
    edges = batch._storage.groups.get("edges")
    if edges is not None and next(edges.keys(), None) is None:
        batch._storage.groups.pop("edges")


def label_dataset(
    dataset: BatchDatasetProtocol,
    scorer: TeacherScorer,
    store: StoreLike,
    *,
    batch_size: int = 32,
    device: torch.device | str | None = None,
    resume: bool = True,
) -> int:
    """Label *dataset* with teacher signals and persist the result to *store*.

    Walks *dataset* in contiguous chunks of *batch_size* samples, scores each
    chunk with *scorer*, attaches every returned signal to the chunk as a
    batch field, and writes the augmented chunk to a Zarr store.  The store
    ends up holding the original fields plus the teacher fields, so the
    labeled dataset is read back through the ordinary
    :class:`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrReader` /
    :class:`~nvalchemi.data.datapipes.dataset.Dataset` path.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Source dataset.  Only ``__len__`` and ``load_batches`` are used, so
        both :class:`~nvalchemi.data.datapipes.dataset.Dataset` and
        :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`
        qualify.
    scorer : TeacherScorer
        Scorer producing the teacher signals for each chunk.
    store : StoreLike
        Destination Zarr store: a path, a zarr store instance, or a dict.
    batch_size : int, optional
        Number of samples scored per forward pass.  Default ``32``.
    device : torch.device | str | None, optional
        Device to move each chunk to before scoring.  Default ``None``
        (score on whatever device the dataset emits).
    resume : bool, optional
        If ``True`` (default), an existing store is treated as a partial run:
        the first ``len(store)`` samples are skipped and labeling continues
        from there.  If ``False``, an existing store is an error.

    Returns
    -------
    int
        Number of samples labeled by this call.  ``0`` when a resumed store
        already covers the whole dataset.

    Raises
    ------
    ValueError
        If *batch_size* is not positive, *store* exists but cannot be read as
        an ALCHEMI Zarr store, *resume* is ``False`` and *store* exists,
        *store* holds soft-deleted samples, or a resumed run would write a
        different field set than the store holds.

    Examples
    --------
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", "forces"])  # doctest: +SKIP
    >>> label_dataset(dataset, scorer, "labeled.zarr", batch_size=64)  # doctest: +SKIP
    1024

    Notes
    -----
    - The first write defines the store schema: every field present on the
      first chunk — teacher fields included — becomes a store array, and later
      chunks only extend arrays that already exist.  All chunks therefore
      carry an identical key set, and a resumed run whose field set differs
      from the store's is rejected instead of silently misaligning arrays.
    - Resuming counts on stored sample *i* being dataset sample *i*, which
      soft-deleted samples break, so a store with deletions is rejected rather
      than continued from the wrong offset.
    - Every source field is carried over, including edge-level ones such as
      ``neighbor_list``, with one exception: the dense neighbor tensors
      (``neighbor_matrix``, ``num_neighbors``, ``neighbor_matrix_shifts``) are
      dropped, because their neighbor dimension is rebuilt per chunk and cannot
      append into a fixed-width store array.  Rebuild them from the stored
      positions when reading.
    - This store is the consumption path for training on teacher labels: point
      a reader at it and the teacher fields arrive alongside the reference
      labels, at the levels recorded here.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}.")

    state = _existing_store_state(store)
    if state is None and isinstance(store, (str, Path)) and Path(store).exists():
        raise ValueError(
            "Store path exists but is not a readable ALCHEMI Zarr store; got "
            f"{store!s}."
        )
    if state is not None and not resume:
        raise ValueError(
            f"Store already exists with {state.active!r} samples and resume is False; "
            "pass resume=True to continue labeling or write to a fresh store."
        )
    if state is not None and state.active != state.total:
        raise ValueError(
            f"Store holds {state.total - state.active!r} soft-deleted samples, so a "
            "resumed run cannot line up with the dataset; defragment the store or "
            "label into a fresh one."
        )

    total = len(dataset)
    start = state.active if state is not None else 0
    stored_fields = state.fields if state is not None else None
    if start >= total:
        return 0

    writer = AtomicDataZarrWriter(store)
    labeled = 0
    for begin in range(start, total, batch_size):
        indices = list(range(begin, min(begin + batch_size, total)))
        batch = dataset.load_batches([indices])[0]
        if device is not None:
            batch = batch.to(device)
        loaded_fields = _persisted_fields(batch)
        labels = scorer.label(batch)
        for field, (values, level) in labels.items():
            if level == "system":
                _ensure_system_group(batch)
            batch.add_key(
                field,
                _split_per_graph(batch, values, level),
                level=level,
                overwrite=True,
            )
        _strip_unstorable(batch, loaded_fields | frozenset(labels))
        if stored_fields is not None:
            _check_store_schema(stored_fields, _persisted_fields(batch))
            stored_fields = None
        if state is None and begin == start:
            writer.write(batch)
        else:
            writer.append(batch)
        labeled += batch.num_graphs
    return labeled
