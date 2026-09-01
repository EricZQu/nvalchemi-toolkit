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
"""Tests for :mod:`nvalchemi.training.distillation.replay`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.training.distillation import (
    InProcessTeacherScorer,
    ReplayBuffer,
    build_mixed_loader,
    label_dataset,
)
from test.training.distillation.conftest import (
    _build_direct_force_teacher,
    _build_small_dataset,
)

_ATOMS_PER_FRAME = 3
"""Atoms in every synthetic frame, so tagged frames stay directly comparable."""


def _make_frames(
    tags: list[float], *, labeled: bool = True, device: str = "cpu"
) -> Batch:
    """Return one frame per entry of *tags*, tagged by its system energy."""
    frames = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.full((_ATOMS_PER_FRAME, 3), tag),
                atomic_numbers=torch.ones(_ATOMS_PER_FRAME, dtype=torch.long),
                energy=torch.full((1, 1), tag),
            )
            for tag in tags
        ]
    ).to(device)
    if labeled:
        frames.add_key(
            "teacher_forces",
            [torch.full((_ATOMS_PER_FRAME, 3), tag) for tag in tags],
            level="node",
        )
        frames.add_key(
            "teacher_energy",
            [torch.full((1, 1), tag) for tag in tags],
            level="system",
        )
    return frames


def _make_buffer(tags: list[float], **kwargs: object) -> ReplayBuffer:
    """Return a buffer already holding one labeled frame per tag."""
    buffer = ReplayBuffer(**kwargs)
    buffer.extend(_make_frames(tags))
    return buffer


def _tags(batch: Batch) -> list[float]:
    """Return the per-graph tag of every frame in *batch*."""
    return batch.energy.view(-1).tolist()


def _make_labeled_store(store: Path) -> Dataset:
    """Return a teacher-labeled Zarr dataset, the shape ``label_dataset`` writes."""
    teacher = _build_direct_force_teacher(seed=2)
    label_dataset(
        _build_small_dataset(),
        InProcessTeacherScorer(teacher, ("energy", "forces")),
        store,
        batch_size=2,
    )
    return Dataset(reader=AtomicDataZarrReader(store), device="cpu")


class TestReplayBufferSchema:
    def test_first_extend_freezes_the_schema(self) -> None:
        """The schema records the level each stored field lives at."""
        buffer = _make_buffer([0.0, 1.0])

        assert "node.teacher_forces" in buffer.schema
        assert "system.teacher_energy" in buffer.schema
        assert len(buffer) == 2

    def test_unlabeled_frames_are_rejected(self) -> None:
        """An unlabeled frame would strip ``teacher_*`` from the whole buffer."""
        buffer = _make_buffer([0.0])

        with pytest.raises(ValueError, match="missing \\['node.teacher_forces'"):
            buffer.extend(_make_frames([1.0], labeled=False))

        assert len(buffer) == 1

    def test_extra_key_is_rejected(self) -> None:
        """A frame carrying more than the schema is rejected just as loudly."""
        buffer = _make_buffer([0.0])
        wider = _make_frames([1.0])
        wider.add_key("teacher_stress", [torch.zeros(1, 3, 3)], level="system")

        with pytest.raises(ValueError, match="extra \\['system.teacher_stress'\\]"):
            buffer.extend(wider)

    def test_matching_frames_are_appended(self) -> None:
        """Frames sharing the schema concatenate in arrival order."""
        buffer = _make_buffer([0.0, 1.0])

        buffer.extend(_make_frames([2.0]))

        assert len(buffer) == 3
        assert _tags(buffer.dataset.in_memory_batch) == [0.0, 1.0, 2.0]

    def test_buffer_does_not_alias_the_batch_it_was_seeded_with(self) -> None:
        """A propagator may keep integrating the batch it handed over."""
        frames = _make_frames([0.0])
        buffer = ReplayBuffer()
        buffer.extend(frames)

        frames.positions.add_(5.0)

        assert torch.allclose(
            buffer.dataset.in_memory_batch.positions,
            torch.zeros_like(buffer.dataset.in_memory_batch.positions),
        )


class TestReplayBufferCapacity:
    def test_fifo_eviction_keeps_the_newest_frames(self) -> None:
        """Overflowing a capacity-3 buffer drops the oldest frames first."""
        buffer = _make_buffer([0.0, 1.0, 2.0], capacity=3)

        buffer.extend(_make_frames([3.0, 4.0]))

        assert len(buffer) == 3
        assert _tags(buffer.dataset.in_memory_batch) == [2.0, 3.0, 4.0]

    def test_unbounded_buffer_keeps_every_frame(self) -> None:
        """Without a capacity nothing is ever evicted."""
        buffer = _make_buffer([0.0, 1.0])

        buffer.extend(_make_frames([2.0, 3.0]))

        assert len(buffer) == 4

    def test_seeding_over_capacity_evicts_immediately(self) -> None:
        """Capacity is enforced on the first extend as well as later ones."""
        buffer = _make_buffer([0.0, 1.0, 2.0], capacity=2)

        assert _tags(buffer.dataset.in_memory_batch) == [1.0, 2.0]

    def test_non_positive_capacity_raises(self) -> None:
        """A capacity of zero could never hold a frame."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=0)

    def test_uncertainty_eviction_raises(self) -> None:
        """The uncertainty policy is reserved, not implemented."""
        with pytest.raises(NotImplementedError, match="Uncertainty-steered eviction"):
            ReplayBuffer(capacity=8, eviction="uncertainty")

    def test_empty_buffer_has_no_dataset(self) -> None:
        """Reading the dataset before the first extend is an error, not None."""
        with pytest.raises(RuntimeError, match="holds no frames yet"):
            _ = ReplayBuffer().dataset

        assert len(ReplayBuffer()) == 0


class TestReplayBufferDevice:
    def test_frames_are_moved_to_the_buffer_device(self, device: str) -> None:
        """A buffer with a device of its own holds frames there, not where they arrive."""
        buffer = ReplayBuffer(device="cpu")

        buffer.extend(_make_frames([0.0], device=device))

        assert buffer.dataset.in_memory_batch.device.type == "cpu"


class TestBuildMixedLoader:
    def test_every_batch_holds_the_requested_mixture(self) -> None:
        """A quarter replay ratio puts exactly one replay frame in a batch of four."""
        reference = InMemoryDataset(in_memory_batch=_make_frames([0.0] * 8))
        buffer = _make_buffer([1.0] * 4)

        loader = build_mixed_loader(
            reference, buffer, replay_ratio=0.25, batch_size=4, num_batches=3
        )
        batches = list(loader)

        assert len(batches) == 3
        for batch in batches:
            tags = _tags(batch)
            assert len(tags) == 4
            assert tags.count(1.0) == 1
            assert tags.count(0.0) == 3

    def test_replay_only_loader_when_there_is_no_reference_dataset(self) -> None:
        """A full replay ratio draws straight from the buffer's dataset."""
        buffer = _make_buffer([1.0] * 4)

        loader = build_mixed_loader(None, buffer, replay_ratio=1.0, batch_size=2)

        assert loader.dataset is buffer.dataset
        assert _tags(next(iter(loader))) == [1.0, 1.0]

    def test_empty_buffer_falls_back_to_the_reference_dataset(self) -> None:
        """A run whose first segment is not stored yet trains on reference data."""
        reference = InMemoryDataset(in_memory_batch=_make_frames([0.0] * 4))

        loader = build_mixed_loader(
            reference, ReplayBuffer(), replay_ratio=0.5, batch_size=2
        )

        assert loader.dataset is reference
        assert _tags(next(iter(loader))) == [0.0, 0.0]

    def test_partial_ratio_without_a_reference_dataset_raises(self) -> None:
        """Mixing in reference data requires a reference dataset."""
        buffer = _make_buffer([1.0] * 2)

        with pytest.raises(ValueError, match="reference dataset is required"):
            build_mixed_loader(None, buffer, replay_ratio=0.5, batch_size=2)

    def test_two_empty_sources_raise(self) -> None:
        """Nothing to draw from is a configuration error, not an empty loader."""
        with pytest.raises(ValueError, match="needs something to draw from"):
            build_mixed_loader(None, ReplayBuffer(), replay_ratio=1.0, batch_size=2)

    def test_ratio_outside_the_unit_interval_raises(self) -> None:
        """A replay ratio is a fraction of the batch."""
        buffer = _make_buffer([1.0])

        with pytest.raises(ValueError, match="replay_ratio must lie in"):
            build_mixed_loader(None, buffer, replay_ratio=1.5, batch_size=2)

    def test_unlabeled_reference_dataset_is_rejected(self) -> None:
        """Mixing a labeled buffer with an unlabeled store fails at composition."""
        reference = InMemoryDataset(
            in_memory_batch=_make_frames([0.0] * 4, labeled=False)
        )
        buffer = _make_buffer([1.0] * 4)

        with pytest.raises(ValueError, match="same teacher fields"):
            build_mixed_loader(reference, buffer, replay_ratio=0.5, batch_size=2)

    def test_a_reference_dataset_carrying_extra_teacher_fields_is_rejected(
        self,
    ) -> None:
        """The teacher namespaces must match in both directions, not just one."""
        frames = _make_frames([0.0] * 4)
        frames.add_key(
            "teacher_stress", [torch.zeros(1, 3, 3) for _ in range(4)], level="system"
        )
        reference = InMemoryDataset(in_memory_batch=frames)
        buffer = _make_buffer([1.0] * 4)

        with pytest.raises(ValueError, match="teacher_stress"):
            build_mixed_loader(reference, buffer, replay_ratio=0.5, batch_size=2)

    def test_a_teacher_labeled_store_mixes_with_the_buffer(
        self, tmp_path: Path
    ) -> None:
        """The documented anchor — a labeled Zarr store — composes with the buffer."""
        reference = _make_labeled_store(tmp_path / "labeled.zarr")
        buffer = _make_buffer([1.0] * 4)

        loader = build_mixed_loader(
            reference, buffer, replay_ratio=0.5, batch_size=4, num_batches=2
        )
        batch = next(iter(loader))

        assert batch.num_graphs == 4
        assert batch.teacher_energy.shape == (4, 1)
        assert batch.teacher_forces.shape == (batch.num_nodes, 3)

    def test_a_ratio_that_rounds_a_source_away_is_rejected(self) -> None:
        """A ratio below half a sample of the batch would train on one source."""
        reference = InMemoryDataset(in_memory_batch=_make_frames([0.0] * 8))
        buffer = _make_buffer([1.0] * 4)

        with pytest.raises(ValueError, match="never reaches an optimizer step"):
            build_mixed_loader(reference, buffer, replay_ratio=0.05, batch_size=8)

        with pytest.raises(ValueError, match="never reaches an optimizer step"):
            build_mixed_loader(reference, buffer, replay_ratio=0.95, batch_size=8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sources_on_different_devices_are_rejected(self) -> None:
        """Collation would concatenate across devices, so it fails by name here."""
        reference = InMemoryDataset(in_memory_batch=_make_frames([0.0] * 4))
        buffer = ReplayBuffer()
        buffer.extend(_make_frames([1.0] * 4, device="cuda"))

        with pytest.raises(ValueError, match="ReplayBuffer\\(device=...\\)"):
            build_mixed_loader(reference, buffer, replay_ratio=0.5, batch_size=4)

    def test_num_batches_sizes_a_replay_only_loader(self) -> None:
        """A lone source is oversampled to the requested epoch, not cut short by it."""
        buffer = _make_buffer([1.0] * 4)

        loader = build_mixed_loader(
            None, buffer, replay_ratio=1.0, batch_size=4, num_batches=7
        )
        batches = list(loader)

        assert len(batches) == 7
        assert all(batch.num_graphs == 4 for batch in batches)

    def test_num_batches_sizes_a_reference_only_loader(self) -> None:
        """The empty-buffer fallback is sized to the segment the same way."""
        reference = InMemoryDataset(in_memory_batch=_make_frames([0.0] * 4))

        loader = build_mixed_loader(
            reference, ReplayBuffer(), replay_ratio=0.5, batch_size=4, num_batches=5
        )

        assert len(list(loader)) == 5

    def test_advancing_the_sampler_epoch_redraws_the_reference_share(self) -> None:
        """A rebuilt sampler restarts its seeded stream unless the epoch moves on."""
        reference = InMemoryDataset(
            in_memory_batch=_make_frames([float(index) for index in range(32)])
        )
        buffer = _make_buffer([100.0] * 8)
        kwargs = {"replay_ratio": 0.5, "batch_size": 8, "num_batches": 4}

        first = build_mixed_loader(reference, buffer, **kwargs)
        second = build_mixed_loader(reference, buffer, **kwargs)
        second.batch_sampler.set_epoch(1)
        drawn = [
            [tag for batch in loader for tag in _tags(batch) if tag < 100.0]
            for loader in (first, second)
        ]

        assert drawn[0] != drawn[1]

    def test_loader_must_be_rebuilt_after_the_buffer_grows(self) -> None:
        """The batch sampler caches child lengths, so a grown buffer is invisible."""
        reference = InMemoryDataset(in_memory_batch=_make_frames([0.0] * 8))
        buffer = _make_buffer([1.0] * 4)
        stale = build_mixed_loader(
            reference, buffer, replay_ratio=0.5, batch_size=4, num_batches=2
        )

        buffer.extend(_make_frames([2.0] * 4))
        rebuilt = build_mixed_loader(
            reference, buffer, replay_ratio=0.5, batch_size=4, num_batches=2
        )

        assert stale.batch_sampler.lengths == [8, 4]
        assert rebuilt.batch_sampler.lengths == [8, 8]
        assert 2.0 in {tag for batch in rebuilt for tag in _tags(batch)}
