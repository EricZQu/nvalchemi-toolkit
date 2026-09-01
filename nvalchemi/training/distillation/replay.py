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
"""Replay buffer of generated frames and the reference/replay mixing loader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import torch

from nvalchemi.data.datapipes.dataloader import DataLoader
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.data.datapipes.multidataset import MultiDataset
from nvalchemi.data.datapipes.samplers import MultiDatasetBatchSampler

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol

__all__ = ["ReplayBuffer", "ReplayEviction", "build_mixed_loader"]

ReplayEviction: TypeAlias = Literal["fifo", "uncertainty"]
"""Policy choosing which frames leave a replay buffer that is over capacity."""

_GROUP_LEVELS = {"atoms": "node", "edges": "edge", "system": "system"}
"""Batch level each storage group holds, used to report a schema mismatch."""


def _frame_schema(frames: Batch) -> frozenset[str]:
    """Return the ``level.field`` names :meth:`Batch.append` intersects over."""
    return frozenset(
        f"{_GROUP_LEVELS.get(name, name)}.{key}"
        for name, group in frames._storage.groups.items()
        for key in group.keys()
    )


class ReplayBuffer:
    """Hold generated frames for replay, behind one frozen key schema.

    The buffer is an :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`
    grown one segment at a time, which makes it a plain
    :class:`~nvalchemi.data.datapipes.dataset.BatchDatasetProtocol` source that
    a :class:`~nvalchemi.data.datapipes.dataloader.DataLoader` or a
    :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` consumes like
    any other dataset. It starts empty and materializes on the first
    :meth:`extend`.

    The schema check is the point of the class rather than a safety net.
    :meth:`~nvalchemi.data.Batch.append` keeps only the keys both sides hold,
    so a single unlabeled frame appended to a labeled buffer would silently
    strip ``teacher_*`` from *every* frame already stored and leave the loss
    with a missing target several segments later. The first :meth:`extend`
    freezes the incoming schema — levels included, because ``append`` merges
    group by group — and every later one must match it exactly.

    Over capacity, ``eviction="fifo"`` drops the oldest frames by rebuilding
    the resident batch from the kept indices.

    Parameters
    ----------
    capacity : int | None, optional
        Maximum number of frames kept. Default ``None`` (unbounded).
    eviction : {"fifo", "uncertainty"}, optional
        Policy deciding which frames leave a full buffer. Default ``"fifo"``.
        ``"uncertainty"`` is reserved for uncertainty-steered sampling and is
        not implemented yet.
    device : torch.device | str | None, optional
        Device the buffer keeps frames on, and emits them from. Default
        ``None`` (keep frames wherever they arrive, typically the propagator's
        device).

    Raises
    ------
    ValueError
        If *capacity* is not positive.
    NotImplementedError
        If ``eviction="uncertainty"`` is selected.

    Examples
    --------
    >>> from nvalchemi.training.distillation import ReplayBuffer
    >>> buffer = ReplayBuffer(capacity=4096)
    >>> buffer.extend(labeled_frames)  # doctest: +SKIP
    >>> len(buffer)  # doctest: +SKIP
    128

    Notes
    -----
    Frames are owned, not aliased: the batch that seeds the buffer is copied,
    and later ones are concatenated into fresh tensors, so a propagator may
    keep integrating the batch it handed over.
    """

    def __init__(
        self,
        *,
        capacity: int | None = None,
        eviction: ReplayEviction = "fifo",
        device: torch.device | str | None = None,
    ) -> None:
        """Validate the capacity and eviction policy of an empty buffer."""
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be positive or None; got {capacity!r}.")
        if eviction == "uncertainty":
            raise NotImplementedError(
                "Uncertainty-steered eviction is reserved for committee-based "
                f"frame selection and is not implemented yet; got {eviction!r}, "
                "use 'fifo'."
            )
        self.capacity = capacity
        self.eviction = eviction
        self.device = device
        self._dataset: InMemoryDataset | None = None
        self._schema: frozenset[str] = frozenset()

    def __len__(self) -> int:
        """Return the number of frames currently held."""
        return 0 if self._dataset is None else len(self._dataset)

    @property
    def dataset(self) -> InMemoryDataset:
        """Dataset view of the stored frames, for a loader to draw from."""
        if self._dataset is None:
            raise RuntimeError(
                "ReplayBuffer holds no frames yet; call extend() before reading "
                "its dataset."
            )
        return self._dataset

    @property
    def schema(self) -> frozenset[str]:
        """Frozen ``level.field`` schema every frame must match, empty until filled."""
        return self._schema

    def extend(self, frames: Batch) -> None:
        """Add *frames* to the buffer and evict down to capacity.

        Parameters
        ----------
        frames : Batch
            Frames to store, one graph each. The first call freezes the
            buffer's key schema; later calls must match it.

        Raises
        ------
        ValueError
            If the key schema of *frames* differs from the buffer's.
        """
        if frames.num_graphs == 0:
            return
        if self.device is not None:
            frames = frames.to(self.device)
        incoming = _frame_schema(frames)
        if self._dataset is None:
            self._schema = incoming
            self._dataset = InMemoryDataset(
                in_memory_batch=frames.clone(), device=self.device
            )
        else:
            self._check_schema(incoming)
            self._dataset.in_memory_batch.append(frames)
        self._evict()

    def _check_schema(self, incoming: frozenset[str]) -> None:
        """Reject frames whose keys or levels differ from the frozen schema."""
        if incoming == self._schema:
            return
        raise ValueError(
            "Replay frames must carry the buffer's key schema, because appending "
            "keeps only the keys both sides hold; got extra "
            f"{sorted(incoming - self._schema)!r} and missing "
            f"{sorted(self._schema - incoming)!r}."
        )

    def _evict(self) -> None:
        """Drop the oldest frames until the buffer fits its capacity."""
        if self._dataset is None or self.capacity is None:
            return
        resident = self._dataset.in_memory_batch
        if resident.num_graphs <= self.capacity:
            return
        kept = torch.arange(
            resident.num_graphs - self.capacity,
            resident.num_graphs,
            device=resident.device,
        )
        self._dataset.in_memory_batch = resident.index_select(kept)


def build_mixed_loader(
    reference_dataset: BatchDatasetProtocol | None,
    replay_buffer: ReplayBuffer,
    *,
    replay_ratio: float,
    batch_size: int,
    num_batches: int | None = None,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build a loader drawing a fixed reference/replay mixture in every batch.

    The two sources are composed into a
    :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` and drawn by a
    :class:`~nvalchemi.data.datapipes.samplers.MultiDatasetBatchSampler` with
    ``samples_per_dataset=(1 - replay_ratio, replay_ratio)``. Float rates give
    *exact* per-batch composition rather than an average, so with
    ``replay_ratio=0.25`` and ``batch_size=8`` every optimizer step sees six
    reference samples and two replay samples.

    **Rebuild this loader after every segment.** The batch sampler reads the
    child dataset lengths once, in its constructor, and a buffer that has grown
    since is invisible to it: the loader keeps drawing from the prefix the
    sampler was built against and the newest frames — the on-policy ones — are
    never sampled.

    Parameters
    ----------
    reference_dataset : BatchDatasetProtocol | None
        Anchor dataset, typically a teacher-labeled store. ``None`` means the
        run trains on generated data only and requires ``replay_ratio=1.0``.
    replay_buffer : ReplayBuffer
        Buffer of generated frames. An empty buffer falls back to a
        reference-only loader, which is the shape of a run whose first segment
        has not been stored yet.
    replay_ratio : float
        Fraction of every batch drawn from *replay_buffer*, in ``[0, 1]``.
    batch_size : int
        Samples per batch across both sources.
    num_batches : int | None, optional
        Batches per epoch. Default ``None`` (the sampler's own
        ``"dataset_size"`` policy); pass the number of optimizer steps a
        segment runs to size the epoch to the segment.
    shuffle : bool, optional
        Randomize sample order within each child and within each batch.
        Default ``True``.
    generator : torch.Generator | None, optional
        Generator for reproducible mixing. Default ``None``. Used by the mixed
        sampler; the single-source fallbacks draw from the global RNG.

    Returns
    -------
    DataLoader
        Loader yielding :class:`~nvalchemi.data.Batch` objects of the requested
        composition.

    Raises
    ------
    ValueError
        If *replay_ratio* is outside ``[0, 1]``, if both sources are empty, if
        *reference_dataset* is ``None`` while ``replay_ratio < 1``, or if the
        two sources expose different field names.

    Examples
    --------
    >>> from nvalchemi.training.distillation import build_mixed_loader
    >>> loader = build_mixed_loader(  # doctest: +SKIP
    ...     reference_dataset,
    ...     buffer,
    ...     replay_ratio=0.25,
    ...     batch_size=8,
    ...     num_batches=64,
    ... )

    Notes
    -----
    ``MultiDataset`` requires both sources to expose identical field names, so
    mixing a labeled buffer with an unlabeled reference dataset fails here
    rather than silently dropping the teacher fields during collation. Label
    the reference dataset with
    :func:`~nvalchemi.training.distillation.label_dataset` first.

    The sampler draws with replacement, so a replay buffer smaller than its
    per-batch allocation oversamples rather than failing.
    """
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError(f"replay_ratio must lie in [0, 1]; got {replay_ratio!r}.")

    if len(replay_buffer) == 0:
        if reference_dataset is None:
            raise ValueError(
                "build_mixed_loader needs something to draw from; got an empty "
                "replay buffer and reference_dataset=None."
            )
        return DataLoader(reference_dataset, batch_size=batch_size, shuffle=shuffle)

    if reference_dataset is None:
        if replay_ratio != 1.0:
            raise ValueError(
                "A replay_ratio below 1 mixes in reference data, so a reference "
                "dataset is required; got reference_dataset=None and "
                f"replay_ratio={replay_ratio!r}."
            )
        return DataLoader(replay_buffer.dataset, batch_size=batch_size, shuffle=shuffle)

    mixed = MultiDataset(reference_dataset, replay_buffer.dataset)
    return DataLoader(
        mixed,
        batch_sampler=MultiDatasetBatchSampler(
            mixed,
            batch_size=batch_size,
            samples_per_dataset=(1.0 - replay_ratio, replay_ratio),
            num_batches=num_batches,
            shuffle=shuffle,
            generator=generator,
        ),
    )
