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

_SCHEMA_REMEDY = (
    "Label the reference dataset with label_dataset, requesting the signals the "
    "propagator's scorer produces, and store it in the shape a replay frame "
    "has: the structure, whatever propagator state travels with it, and the "
    "teacher_* labels, with none of the energy, forces, or stress the labeling "
    "hook strips."
)
"""Remedy naming the replay-frame contract both mixture sources have to meet."""


def _frame_schema(frames: Batch) -> frozenset[str]:
    """Return the ``level.field`` names :meth:`Batch.append` intersects over."""
    return frozenset(
        f"{_GROUP_LEVELS.get(name, name)}.{key}"
        for name, group in frames._storage.groups.items()
        for key in group.keys()
    )


def _frame_dtypes(frames: Batch) -> dict[str, torch.dtype]:
    """Return the dtype every ``level.field`` of *frames* is stored at."""
    return {
        f"{_GROUP_LEVELS.get(name, name)}.{key}": group[key].dtype
        for name, group in frames._storage.groups.items()
        for key in group.keys()
    }


def _schema_levels(schema: Iterable[str]) -> frozenset[str]:
    """Return the batch levels *schema* holds at least one field at."""
    return frozenset(name.partition(".")[0] for name in schema)


def _emitted_device(
    dataset: BatchDatasetProtocol, probe: Batch | None = None
) -> torch.device:
    """Return the concrete device *dataset* emits its batches on.

    A declaration settles it where one exists — a ``target_device`` or the
    device of a resident ``in_memory_batch`` — and a batch is drawn otherwise:
    a :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` declares no
    device, and a store opened without one declares an index-less ``cuda``
    naming whichever device is current, so both are measured instead.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Dataset to resolve the emission device of.
    probe : Batch | None, optional
        A batch already drawn from *dataset*. Default ``None`` (draw one when
        needed).

    Returns
    -------
    torch.device
        Device batches are emitted on.
    """
    target = getattr(dataset, "target_device", None)
    resident = getattr(dataset, "in_memory_batch", None)
    declared = (
        torch.device(target)
        if target is not None
        else None
        if resident is None
        else resident.device
    )
    if declared is not None and not (
        declared.type == "cuda" and declared.index is None
    ):
        return declared
    if probe is None:
        probe = dataset.load_batches([[0]])[0]
    return probe.device


def _same_device(left: torch.device | None, right: torch.device | None) -> bool:
    """Return whether two emitted devices collate without a cross-device copy.

    An index-less device is compared by type alone; two indexed devices have to
    name the same one. ``None`` on either side is no constraint.
    """
    if left is None or right is None:
        return True
    if left.type != right.type:
        return False
    return left.index is None or right.index is None or left.index == right.index


def _check_mixture_sources(
    reference_dataset: BatchDatasetProtocol, replay_buffer: ReplayBuffer
) -> None:
    """Reject two sources that cannot be collated into one training batch.

    The reference schema is read from a one-sample probe rather than from
    ``field_names``, which a Zarr-backed dataset and an in-memory one never
    report alike. Fields are compared by dtype as well as by name, because
    collation casts the second part of a mixed batch to the first's dtype and
    which source leads a chunk is not fixed.

    Raises
    ------
    ValueError
        If one source holds a batch level the other lacks, if they carry
        different fields, if they carry a field at different dtypes, or if they
        emit their batches on different devices.
    """
    probe = reference_dataset.load_batches([[0]])[0]
    reference_schema = _frame_schema(probe)
    replay_schema = replay_buffer.schema
    reference_levels = _schema_levels(reference_schema)
    replay_levels = _schema_levels(replay_schema)
    if reference_levels != replay_levels:
        raise ValueError(
            "Both mixture sources must hold the same batch levels, because "
            "collation zero-fills a level only one of them carries instead of "
            f"dropping it; got {sorted(reference_levels)!r} on the reference "
            f"dataset and {sorted(replay_levels)!r} on the replay buffer, "
            f"differing in {sorted(reference_levels ^ replay_levels)!r}. "
            f"{_SCHEMA_REMEDY}"
        )
    if reference_schema != replay_schema:
        raise ValueError(
            "Both mixture sources must carry the same fields, because collation "
            "keeps only the fields both hold and drops the rest out of every "
            f"mixed batch; got {sorted(reference_schema - replay_schema)!r} on "
            "the reference dataset alone and "
            f"{sorted(replay_schema - reference_schema)!r} on the replay buffer "
            f"alone. {_SCHEMA_REMEDY}"
        )
    reference_dtypes = _frame_dtypes(probe)
    replay_dtypes = _frame_dtypes(replay_buffer.dataset.in_memory_batch)
    mismatched = sorted(
        name
        for name in reference_dtypes
        if reference_dtypes[name] != replay_dtypes[name]
    )
    if mismatched:
        detail = "; ".join(
            f"{name!r} at {reference_dtypes[name]!s} on the reference dataset "
            f"and {replay_dtypes[name]!s} on the replay buffer"
            for name in mismatched
        )
        raise ValueError(
            "Both mixture sources must carry each field at one dtype, because "
            "collation casts the second part of a mixed batch to the dtype of "
            "the first and the two sources take turns leading a chunk; got "
            f"{detail}. Label the reference dataset with the dtype the "
            "on-policy scorer uses — the student's parameter dtype — or cast "
            "it in a batch transform."
        )
    reference_device = _emitted_device(reference_dataset, probe)
    replay_device = _emitted_device(replay_buffer.dataset)
    if not _same_device(reference_device, replay_device):
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
    """Return the smallest batch size giving both mixture sources a sample.

    The ratio algebra alone is not enough: :func:`_batch_allocation` rounds a
    half sample up into the replay share, so a size where the reference share
    lands exactly on that boundary still starves it. The count is therefore
    walked up until the allocator itself agrees, which takes one step at most.
    """
    size = ceil(0.5 / min(replay_ratio, 1.0 - replay_ratio))
    while min(_batch_allocation(replay_ratio, size)) == 0:
        size += 1
    return size


def _batch_size_remedy(replay_ratio: float) -> str:
    """Return the remedy clause naming a batch size the allocator does accept."""
    remedy = f"raise batch_size to at least {_minimum_batch_size(replay_ratio)}"
    if replay_ratio == 0.5:
        return remedy
    return f"{remedy}, or move replay_ratio toward 0.5"


def _single_source_loader(
    dataset: BatchDatasetProtocol,
    *,
    batch_size: int,
    num_batches: int | None,
    shuffle: bool,
    generator: torch.Generator | None,
    seed: int,
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
            seed=seed,
        ),
    )


class ReplayBuffer:
    """Hold generated frames for replay, behind one frozen key schema.

    An :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`
    grown one segment at a time, so a loader or a
    :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` consumes it
    like any dataset. The first :meth:`extend` freezes the incoming schema,
    levels included, and every later one must match it exactly:
    :meth:`~nvalchemi.data.Batch.append` keeps only the keys both sides hold,
    so one unlabeled frame would otherwise strip ``teacher_*`` from every frame
    already stored. A stored frame is a training sample rather than a
    propagator state — the structure and its ``teacher_*`` labels, none of the
    predictions the propagator wrote — which is the shape
    :class:`~nvalchemi.training.distillation.TeacherLabelHook` delivers and
    :func:`build_mixed_loader` holds the reference dataset to. Over capacity,
    ``eviction="fifo"`` drops the oldest frames.

    Parameters
    ----------
    capacity : int | None, optional
        Maximum number of frames kept. Default ``None`` (unbounded); bound it
        on long runs.
    eviction : {"fifo", "uncertainty"}, optional
        Policy deciding which frames leave a full buffer. Default ``"fifo"``;
        ``"uncertainty"`` is reserved and not implemented yet.
    device : torch.device | str | None, optional
        Device the buffer keeps frames on and emits them from. Default
        ``None`` (wherever they arrive). A segment loop resolves
        ``OnPolicyConfig.replay_device`` into this.

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
    Frames are owned, not aliased: the batch that seeds the buffer is copied
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
    seed: int = 0,
) -> DataLoader:
    """Build a loader drawing a fixed reference/replay mixture in every batch.

    The two sources are composed into a
    :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` and drawn by a
    :class:`~nvalchemi.data.datapipes.samplers.MultiDatasetBatchSampler` with
    the ratio resolved to whole samples of *batch_size*, so the composition is
    exact per batch — ``replay_ratio=0.25`` and ``batch_size=8`` is six
    reference and two replay samples every step — at a granularity of
    ``1 / batch_size``. Rebuild the loader after every segment: the sampler
    reads the child dataset lengths once, at construction, so frames added
    since are never sampled.

    Parameters
    ----------
    reference_dataset : BatchDatasetProtocol | None
        Anchor dataset, typically a teacher-labeled store. ``None`` trains on
        generated data only and requires ``replay_ratio=1.0``.
    replay_buffer : ReplayBuffer
        Buffer of generated frames. An empty buffer falls back to a
        reference-only loader.
    replay_ratio : float
        Fraction of every batch drawn from *replay_buffer*, in ``[0, 1]``.
    batch_size : int
        Samples per batch across both sources.
    num_batches : int | None, optional
        Batches per epoch, honored on every path. Default ``None`` (the
        sampler's ``"dataset_size"`` policy, and one pass over a lone source).
    shuffle : bool, optional
        Randomize sample order within each child and each batch. Default
        ``True``.
    generator : torch.Generator | None, optional
        Generator for reproducible mixing. Default ``None``. An unsized
        single-source fallback draws from the global RNG instead.
    seed : int, optional
        Base seed the batch sampler draws from when it owns its generator,
        combined with the epoch set on it. Default ``0``.

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
        sources differ in batch levels, fields, a field's dtype, or emission
        device, or if the ratio allocates no samples to one of them.

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
    Collation is not a merge: :meth:`~nvalchemi.data.Batch.append` drops a
    field only one side holds and zero-fills a whole level only one side
    holds, so both sources have to carry one schema, compared on a probe batch
    from each. That schema is the replay-frame contract — the structure, the
    propagator state travelling with it, and the ``teacher_*`` labels, with
    none of the ``energy``, ``forces``, or ``stress`` the labeling hook strips
    — so an anchor carrying plain reference labels is rejected; label it with
    :func:`~nvalchemi.training.distillation.label_dataset` requesting the
    signals the propagator's scorer produces. The sampler draws with
    replacement, so a buffer smaller than its allocation oversamples.
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
            seed=seed,
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
            seed=seed,
        )

    _check_mixture_sources(reference_dataset, replay_buffer)
    reference_samples, replay_samples = _batch_allocation(replay_ratio, batch_size)
    if 0.0 < replay_ratio < 1.0 and min(reference_samples, replay_samples) == 0:
        raise ValueError(
            f"replay_ratio={replay_ratio!r} allocates {reference_samples} "
            f"reference and {replay_samples} replay samples of "
            f"batch_size={batch_size!r}, so one source never reaches an "
            f"optimizer step; {_batch_size_remedy(replay_ratio)}."
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
            seed=seed,
        ),
    )
