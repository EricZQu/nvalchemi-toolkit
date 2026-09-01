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

from collections.abc import Iterable
from math import ceil
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

_TEACHER_FIELD_PREFIX = "teacher_"
"""Prefix of every batch field a teacher signal populates."""


def _frame_schema(frames: Batch) -> frozenset[str]:
    """Return the ``level.field`` names :meth:`Batch.append` intersects over."""
    return frozenset(
        f"{_GROUP_LEVELS.get(name, name)}.{key}"
        for name, group in frames._storage.groups.items()
        for key in group.keys()
    )


def _teacher_fields(names: Iterable[str]) -> frozenset[str]:
    """Return the ``teacher_*`` entries of *names*, without any level prefix."""
    fields = (name.rpartition(".")[2] for name in names)
    return frozenset(
        field for field in fields if field.startswith(_TEACHER_FIELD_PREFIX)
    )


def _emitted_device(dataset: BatchDatasetProtocol) -> torch.device | None:
    """Return the device *dataset* emits batches on, or ``None`` if it declares none."""
    target = getattr(dataset, "target_device", None)
    if target is not None:
        return torch.device(target)
    resident = getattr(dataset, "in_memory_batch", None)
    return None if resident is None else resident.device


def _check_mixture_sources(
    reference_dataset: BatchDatasetProtocol, replay_buffer: ReplayBuffer
) -> None:
    """Reject two sources that cannot be collated into one training batch.

    Raises
    ------
    ValueError
        If the sources carry different ``teacher_*`` fields, or emit their
        batches on different devices.
    """
    reference_fields = _teacher_fields(reference_dataset.field_names)
    replay_fields = _teacher_fields(replay_buffer.schema)
    if reference_fields != replay_fields:
        raise ValueError(
            "Both mixture sources must carry the same teacher fields, because "
            "collation keeps only the fields both hold; got reference "
            f"{sorted(reference_fields)!r} and replay {sorted(replay_fields)!r}. "
            "Label the reference dataset with label_dataset, requesting the "
            "signals the propagator's scorer produces."
        )
    reference_device = _emitted_device(reference_dataset)
    replay_device = _emitted_device(replay_buffer.dataset)
    if (
        reference_device is not None
        and replay_device is not None
        and reference_device.type != replay_device.type
    ):
        raise ValueError(
            "Both mixture sources must emit batches on one device, because "
            "collation concatenates their tensors; got reference on "
            f"{reference_device!s} and replay on {replay_device!s}. Pass "
            "ReplayBuffer(device=...) — OnPolicyConfig.replay_device from a "
            "segment loop — to stage generated frames where the reference "
            "dataset lives."
        )


def _batch_allocation(replay_ratio: float, batch_size: int) -> tuple[int, int]:
    """Return the ``(reference, replay)`` sample counts of one mixed batch."""
    replay = int(replay_ratio * batch_size + 0.5)
    return batch_size - replay, replay


def _minimum_batch_size(replay_ratio: float) -> int:
    """Return the smallest batch size giving both mixture sources a sample."""
    return ceil(0.5 / min(replay_ratio, 1.0 - replay_ratio))


def _single_source_loader(
    dataset: BatchDatasetProtocol,
    *,
    batch_size: int,
    num_batches: int | None,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader:
    """Return a loader over one source, sized to *num_batches* when given."""
    if num_batches is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    single = MultiDataset(dataset)
    return DataLoader(
        single,
        batch_sampler=MultiDatasetBatchSampler(
            single,
            batch_size=batch_size,
            samples_per_dataset=(batch_size,),
            num_batches=num_batches,
            shuffle=shuffle,
            generator=generator,
        ),
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

    A stored frame is a *training sample*, not a propagator state: it carries
    the structure the student generated — positions, cell, atomic numbers,
    velocities — and the ``teacher_*`` labels, and nothing that describes the
    run that produced it.
    :class:`~nvalchemi.training.distillation.TeacherLabelHook` is what enforces
    that contract on the way in, dropping the ephemeral neighbor tensors, the
    dynamics bookkeeping fields, and the ``energy``, ``forces``, and ``stress``
    the propagator overwrote with the student's own predictions. That last one
    is what keeps a replay frame from carrying a self-label under the name a
    reference target uses: on-policy losses read ``teacher_*``, and a mixed
    batch holds no reference-labeled ``energy`` or ``forces`` at all.

    Over capacity, ``eviction="fifo"`` drops the oldest frames by rebuilding
    the resident batch from the kept indices.

    Parameters
    ----------
    capacity : int | None, optional
        Maximum number of frames kept. Default ``None`` (unbounded), which
        grows for the whole run — bound it on long runs, or on any run whose
        frames stay on the propagator's device.
    eviction : {"fifo", "uncertainty"}, optional
        Policy deciding which frames leave a full buffer. Default ``"fifo"``.
        ``"uncertainty"`` is reserved for uncertainty-steered sampling and is
        not implemented yet.
    device : torch.device | str | None, optional
        Device the buffer keeps frames on, and emits them from. Default
        ``None`` (keep frames wherever they arrive, typically the propagator's
        device). ``OnPolicyConfig.replay_device`` is the knob that reaches this
        one from a segment loop; ``"cpu"`` stages generated frames off the
        accelerator.

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
    the ratio resolved to whole samples of *batch_size*. The composition is
    therefore *exact* per batch rather than an average — with
    ``replay_ratio=0.25`` and ``batch_size=8`` every optimizer step sees six
    reference samples and two replay samples — and the achievable granularity
    is ``1 / batch_size``. A ratio strictly between 0 and 1 that rounds either
    source down to no samples at all is rejected rather than silently trained
    as a single-source run.

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
        Batches per epoch, honored on every path. Default ``None`` (the
        sampler's own ``"dataset_size"`` policy, and one pass over a lone
        source); pass the number of optimizer steps a segment runs to size the
        epoch to the segment.
    shuffle : bool, optional
        Randomize sample order within each child and within each batch.
        Default ``True``.
    generator : torch.Generator | None, optional
        Generator for reproducible mixing. Default ``None``. Used wherever a
        batch sampler draws, which is every path except an unsized
        single-source fallback; that one draws from the global RNG.

    Returns
    -------
    DataLoader
        Loader yielding :class:`~nvalchemi.data.Batch` objects of the requested
        composition.

    Raises
    ------
    ValueError
        If *replay_ratio* is outside ``[0, 1]``, if both sources are empty, if
        *reference_dataset* is ``None`` while ``replay_ratio < 1``, if the two
        sources carry different teacher fields or emit on different devices,
        or if the ratio allocates no samples at all to one of them.

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
    Collation keeps only the fields both sources hold, so the contract the two
    have to meet is the ``teacher_*`` namespace: it is compared here and a
    difference in either direction is rejected rather than left to strip the
    teacher fields out of every mixed batch. Full field-name equality is
    deliberately *not* required, because it is unreachable — a Zarr-backed
    :class:`~nvalchemi.data.datapipes.dataset.Dataset` reports the arrays it
    stores while the buffer's
    :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`
    reports the whole canonical key set. Label the reference dataset with
    :func:`~nvalchemi.training.distillation.label_dataset` first, requesting
    the same signals the propagator's scorer produces.

    Replay frames carry no student-written ``energy`` or ``forces`` (the
    labeling hook strips them), so a mixed batch carries neither, whatever the
    reference dataset stores under those names. On-policy losses read
    ``teacher_*``.

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
        return _single_source_loader(
            reference_dataset,
            batch_size=batch_size,
            num_batches=num_batches,
            shuffle=shuffle,
            generator=generator,
        )

    if reference_dataset is None:
        if replay_ratio != 1.0:
            raise ValueError(
                "A replay_ratio below 1 mixes in reference data, so a reference "
                "dataset is required; got reference_dataset=None and "
                f"replay_ratio={replay_ratio!r}."
            )
        return _single_source_loader(
            replay_buffer.dataset,
            batch_size=batch_size,
            num_batches=num_batches,
            shuffle=shuffle,
            generator=generator,
        )

    _check_mixture_sources(reference_dataset, replay_buffer)
    reference_samples, replay_samples = _batch_allocation(replay_ratio, batch_size)
    if 0.0 < replay_ratio < 1.0 and min(reference_samples, replay_samples) == 0:
        raise ValueError(
            f"replay_ratio={replay_ratio!r} allocates {reference_samples} "
            f"reference and {replay_samples} replay samples of "
            f"batch_size={batch_size!r}, so one source never reaches an "
            "optimizer step; raise batch_size to at least "
            f"{_minimum_batch_size(replay_ratio)} or move replay_ratio toward "
            "0.5."
        )
    mixed = MultiDataset(reference_dataset, replay_buffer.dataset, output_strict=False)
    return DataLoader(
        mixed,
        batch_sampler=MultiDatasetBatchSampler(
            mixed,
            batch_size=batch_size,
            samples_per_dataset=(reference_samples, replay_samples),
            num_batches=num_batches,
            shuffle=shuffle,
            generator=generator,
        ),
    )
