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
"""Tests for :mod:`nvalchemi.training.distillation.labeling`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
)
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training.distillation import InProcessTeacherScorer, label_dataset
from test.training.distillation.conftest import (
    _build_small_dataset,
    _DirectForceTeacher,
)

_SIGNALS = ["energy", "forces", "node_energies", "embeddings"]
"""Signal set exercised by the labeling tests."""

_TEACHER_FIELDS = (
    "teacher_energy",
    "teacher_forces",
    "teacher_node_energies",
    "teacher_node_embeddings",
)
"""Batch fields the ``_SIGNALS`` scorer writes into the store."""


def _make_scorer(teacher: _DirectForceTeacher) -> InProcessTeacherScorer:
    """Return a scorer over every signal the direct-force demo teacher supports."""
    return InProcessTeacherScorer(teacher, _SIGNALS)


def _read_all(store: Path) -> Batch:
    """Return every stored sample as a single CPU batch, with reader field levels.

    ``Dataset(device=None)`` auto-selects CUDA when available, so the device is
    pinned to keep comparisons against CPU scorer outputs hardware-independent.
    """
    reader = AtomicDataZarrReader(store)
    dataset = Dataset(reader=reader, device="cpu")
    return dataset.load_batches([list(range(len(dataset)))])[0]


def _make_neighbor_dataset() -> InMemoryDataset:
    """Return a dataset whose samples carry a sparse neighbor list of varying size."""
    batch = _build_small_dataset().in_memory_batch
    compute_neighbors(batch, cutoff=6.0, format=NeighborListFormat.COO)
    batch.keys["edge"].add("neighbor_list")
    return InMemoryDataset(in_memory_batch=batch)


class _EmptyDataset:
    """Zero-length stand-in for a :class:`BatchDatasetProtocol` dataset."""

    def __len__(self) -> int:
        """Return zero samples."""
        return 0

    def load_batches(
        self,
        batch_index_lists: Sequence[Sequence[int]],  # noqa: ARG002
        stream: torch.cuda.Stream | None = None,  # noqa: ARG002
    ) -> list[Batch]:
        """Fail loudly, since a zero-length dataset must never be read."""
        raise AssertionError("load_batches must not be called for an empty dataset")


class TestLabelDataset:
    """Offline labeling of a dataset into a Zarr store."""

    def test_every_sample_is_labeled_and_stored(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Labeling returns the sample count and the store holds them all."""
        store = tmp_path / "labeled.zarr"
        labeled = label_dataset(
            small_dataset, _make_scorer(direct_force_teacher), store, batch_size=2
        )
        assert labeled == len(small_dataset)
        assert len(AtomicDataZarrReader(store)) == len(small_dataset)

    def test_teacher_fields_are_stored_at_the_expected_levels(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Node signals land at atom level and energy at system level."""
        store = tmp_path / "labeled.zarr"
        label_dataset(
            small_dataset, _make_scorer(direct_force_teacher), store, batch_size=2
        )
        levels = AtomicDataZarrReader(store).field_levels
        assert levels["teacher_energy"] == "system"
        assert levels["teacher_forces"] == "atom"
        assert levels["teacher_node_energies"] == "atom"
        assert levels["teacher_node_embeddings"] == "atom"

    def test_stored_values_match_a_direct_scorer_call(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Round-tripped teacher fields equal the scorer's own output."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        label_dataset(small_dataset, scorer, store, batch_size=2)
        expected = scorer.label(
            small_dataset.load_batches([list(range(len(small_dataset)))])[0]
        )
        stored = _read_all(store)
        for field, (values, _) in expected.items():
            torch.testing.assert_close(stored[field], values)

    def test_stress_round_trips_as_a_system_level_matrix(
        self,
        periodic_dataset: InMemoryDataset,
        lj_teacher: LennardJonesModelWrapper,
        tmp_path: Path,
    ) -> None:
        """A ``(B, 3, 3)`` signal keeps its shape, level, and values through the store."""
        store = tmp_path / "stress.zarr"
        scorer = InProcessTeacherScorer(lj_teacher, ["energy", "stress"])
        label_dataset(periodic_dataset, scorer, store, batch_size=2)
        assert AtomicDataZarrReader(store).field_levels["teacher_stress"] == "system"
        expected = scorer.label(
            periodic_dataset.load_batches([list(range(len(periodic_dataset)))])[0]
        )["teacher_stress"][0]
        stored = _read_all(store)
        assert stored.teacher_stress.shape == (len(periodic_dataset), 3, 3)
        torch.testing.assert_close(stored.teacher_stress, expected)

    def test_dense_neighbor_fields_are_not_persisted(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Dense neighbor tensors are dropped even when the source carries them."""
        store = tmp_path / "labeled.zarr"
        batch = small_dataset.in_memory_batch
        batch._atoms_group["num_neighbors"] = torch.zeros(
            batch.num_nodes, dtype=torch.int32
        )
        batch.keys["node"].add("num_neighbors")
        label_dataset(
            small_dataset, _make_scorer(direct_force_teacher), store, batch_size=2
        )
        assert "num_neighbors" not in AtomicDataZarrReader(store).field_levels

    def test_source_edge_fields_are_carried_over(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A sparse source neighbor list survives labeling with consistent pointers."""
        dataset = _make_neighbor_dataset()
        expected_edges = list(dataset.in_memory_batch.num_edges_list)
        store = tmp_path / "edges.zarr"
        label_dataset(dataset, _make_scorer(direct_force_teacher), store, batch_size=2)
        reader = AtomicDataZarrReader(store)
        assert set(reader.field_levels) == {
            "atom_categories",
            "atomic_masses",
            "atomic_numbers",
            "energy",
            "forces",
            "neighbor_list",
            "positions",
            "velocities",
            *_TEACHER_FIELDS,
        }
        stored = _read_all(store)
        assert stored.num_edges_list == expected_edges

    def test_labeling_moves_each_chunk_to_the_requested_device(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Every loaded chunk is moved to the requested device before scoring."""
        store = tmp_path / "labeled.zarr"
        with patch.object(Batch, "to", autospec=True, side_effect=Batch.to) as spy:
            label_dataset(
                small_dataset,
                _make_scorer(direct_force_teacher),
                store,
                batch_size=2,
                device="cpu",
            )
        assert spy.call_count == 3
        assert all(call.args[1] == "cpu" for call in spy.call_args_list)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_labeling_on_cuda_matches_the_cpu_store(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Labeling on CUDA stores the same values as labeling on CPU."""
        cpu_store = tmp_path / "cpu.zarr"
        label_dataset(
            small_dataset, _make_scorer(direct_force_teacher), cpu_store, batch_size=2
        )
        cuda_store = tmp_path / "cuda.zarr"
        label_dataset(
            small_dataset,
            _make_scorer(direct_force_teacher.to("cuda")),
            cuda_store,
            batch_size=2,
            device="cuda",
        )
        stored_cpu = _read_all(cpu_store)
        stored_cuda = _read_all(cuda_store)
        for field in _TEACHER_FIELDS:
            torch.testing.assert_close(stored_cuda[field], stored_cpu[field])

    def test_resume_on_a_complete_store_labels_nothing(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A second pass over a fully labeled store is a no-op."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        label_dataset(small_dataset, scorer, store, batch_size=2)
        assert label_dataset(small_dataset, scorer, store, batch_size=2) == 0

    def test_resume_continues_a_partial_store(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Resuming labels only the samples the store does not already hold."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        prefix = InMemoryDataset(
            in_memory_batch=small_dataset.in_memory_batch.index_select([0, 1])
        )
        label_dataset(prefix, scorer, store, batch_size=2)
        assert label_dataset(small_dataset, scorer, store, batch_size=2) == 3
        assert len(AtomicDataZarrReader(store)) == len(small_dataset)

    def test_resumed_store_matches_a_single_pass_store(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A resumed run reproduces the store a single uninterrupted run writes."""
        scorer = _make_scorer(direct_force_teacher)
        single = tmp_path / "single.zarr"
        label_dataset(small_dataset, scorer, single, batch_size=2)
        resumed = tmp_path / "resumed.zarr"
        prefix = InMemoryDataset(
            in_memory_batch=small_dataset.in_memory_batch.index_select([0, 1])
        )
        label_dataset(prefix, scorer, resumed, batch_size=2)
        label_dataset(small_dataset, scorer, resumed, batch_size=2)
        expected = _read_all(single)
        actual = _read_all(resumed)
        assert actual.num_nodes_list == expected.num_nodes_list
        for field, values in expected:
            torch.testing.assert_close(actual[field], values)

    def test_resume_with_a_different_signal_set_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A resumed run that would write a different field set is refused."""
        store = tmp_path / "labeled.zarr"
        prefix = InMemoryDataset(
            in_memory_batch=small_dataset.in_memory_batch.index_select([0, 1])
        )
        label_dataset(prefix, _make_scorer(direct_force_teacher), store, batch_size=2)
        narrowed = InProcessTeacherScorer(direct_force_teacher, ["energy"])
        with pytest.raises(ValueError, match="teacher_forces"):
            label_dataset(small_dataset, narrowed, store, batch_size=2)

    def test_resume_false_on_an_existing_store_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """Overwriting an existing store is refused rather than silently appended."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        label_dataset(small_dataset, scorer, store, batch_size=2)
        with pytest.raises(ValueError, match="resume"):
            label_dataset(small_dataset, scorer, store, resume=False)

    def test_resume_false_on_an_emptied_store_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A store whose samples were all deleted still counts as existing."""
        store = tmp_path / "labeled.zarr"
        scorer = _make_scorer(direct_force_teacher)
        label_dataset(small_dataset, scorer, store, batch_size=2)
        AtomicDataZarrWriter(store).delete(list(range(len(small_dataset))))
        assert len(AtomicDataZarrReader(store)) == 0
        with pytest.raises(ValueError, match="resume"):
            label_dataset(small_dataset, scorer, store, resume=False)

    def test_unreadable_store_path_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A path that exists but is not a Zarr store is reported clearly."""
        store = tmp_path / "not-a-store.zarr"
        store.mkdir()
        with pytest.raises(ValueError, match="not a readable"):
            label_dataset(small_dataset, _make_scorer(direct_force_teacher), store)

    def test_empty_dataset_labels_nothing_and_writes_no_store(
        self,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A zero-length dataset is a no-op that leaves no store behind."""
        store = tmp_path / "labeled.zarr"
        labeled = label_dataset(
            _EmptyDataset(), _make_scorer(direct_force_teacher), store
        )
        assert labeled == 0
        assert not store.exists()

    def test_non_positive_batch_size_raises(
        self,
        small_dataset: InMemoryDataset,
        direct_force_teacher: _DirectForceTeacher,
        tmp_path: Path,
    ) -> None:
        """A batch size of zero is rejected before any store is created."""
        with pytest.raises(ValueError, match="batch_size"):
            label_dataset(
                small_dataset,
                _make_scorer(direct_force_teacher),
                tmp_path / "labeled.zarr",
                batch_size=0,
            )
