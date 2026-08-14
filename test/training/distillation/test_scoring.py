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
"""Tests for :mod:`nvalchemi.training.distillation.scoring`."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training.distillation import scoring
from nvalchemi.training.distillation.scoring import (
    InProcessTeacherScorer,
    TeacherScorer,
)
from test.training.distillation.conftest import (
    _LJ_CUTOFF,
    _build_lj_teacher,
    _build_periodic_batch,
    _DirectForceTeacher,
)

_COO_CUTOFF = 4.0
"""Cutoff of the COO stub teacher used in the neighbor-isolation tests."""


def _make_spread_batch(n_systems: int = 2, n_atoms: int = 6) -> Batch:
    """Return a non-periodic batch whose atoms are spread over a few angstroms."""
    data_list = []
    for index in range(n_systems):
        generator = torch.Generator().manual_seed(index)
        data_list.append(
            AtomicData(
                positions=torch.rand(n_atoms, 3, generator=generator) * 6.0,
                atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
                atomic_masses=torch.ones(n_atoms),
                energy=torch.zeros(1, 1),
            )
        )
    return Batch.from_data_list(data_list)


def _build_declared_neighbors(
    batch: Batch,
    cutoff: float,
    neighbor_format: NeighborListFormat,
    half_list: bool = False,
) -> None:
    """Build a neighbor list on *batch* and declare its half-list provenance."""
    compute_neighbors(batch, cutoff=cutoff, format=neighbor_format, half_list=half_list)
    batch._neighbor_list_half = half_list


def _storage_snapshot(batch: Batch) -> dict[str, torch.Tensor]:
    """Return a value copy of every tensor currently stored on *batch*."""
    return {key: value.clone() for key, value in batch}


def _tracked_snapshot(batch: Batch) -> dict[str, set[str]]:
    """Return a copy of the batch's per-level tracked key sets."""
    return {level: set(names) for level, names in batch.keys.items()}


class _GradModeRecorder:
    """Delegating callable that records the ambient grad mode at each call."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.grad_enabled: list[bool] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record whether grad is enabled, then delegate to the wrapped callable."""
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.inner(*args, **kwargs)


class _CooNeighborTeacher(torch.nn.Module, BaseModelMixin):
    """Teacher requiring a sparse (COO) neighbor list, returning an edge-count energy."""

    def __init__(self, cutoff: float = _COO_CUTOFF) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=NeighborConfig(
                cutoff=cutoff, format=NeighborListFormat.COO
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes."""
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Raise, since this teacher produces no embeddings."""
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs: Any) -> OrderedDict:  # noqa: ARG002
        """Return a per-graph energy proportional to the edge count."""
        edges = data.neighbor_list
        energy = torch.full(
            (data.num_graphs, 1), float(edges.shape[0]), dtype=data.positions.dtype
        )
        return OrderedDict([("energy", energy)])


class _AddKeyEmbeddingTeacher(BaseModelMixin):
    """Embedding-only teacher attaching both embedding levels via ``Batch.add_key``.

    Mirrors the in-repo wrappers that publish embeddings through ``add_key``,
    which registers the key in ``batch.keys`` and rejects an existing key.  It
    is deliberately not an ``nn.Module``, so it also covers a teacher that has
    no ``eval()``.
    """

    hidden_dim = 4

    def __init__(self) -> None:
        self.model_config = ModelConfig(outputs=frozenset({"energy"}))

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the per-node embedding shape published by this teacher."""
        return {"node_embeddings": (self.hidden_dim,)}

    def compute_embeddings(self, data: Batch, **kwargs: Any) -> Batch:  # noqa: ARG002
        """Attach node and graph embeddings derived from the atomic numbers."""
        node = (
            data.atomic_numbers.reshape(-1, 1)
            .to(torch.float32)
            .repeat(1, self.hidden_dim)
        )
        data.add_key(
            "node_embeddings",
            list(torch.split(node, data.num_nodes_list)),
            level="node",
        )
        graph = torch.zeros(data.num_graphs, self.hidden_dim)
        graph.scatter_add_(0, data.batch_idx.unsqueeze(-1).expand_as(node), node)
        data.add_key(
            "graph_embeddings",
            [graph[index : index + 1] for index in range(data.num_graphs)],
            level="system",
        )
        return data


class _NonMutatingEmbeddingTeacher(BaseModelMixin):
    """Teacher whose ``compute_embeddings`` returns a copy instead of mutating input."""

    def __init__(self) -> None:
        self.model_config = ModelConfig(outputs=frozenset({"energy"}))

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the per-node embedding shape published by this teacher."""
        return {"node_embeddings": (2,)}

    def compute_embeddings(self, data: Batch, **kwargs: Any) -> Batch:  # noqa: ARG002
        """Return a clone carrying embeddings, leaving *data* untouched."""
        clone = data.clone()
        clone._atoms_group["node_embeddings"] = torch.zeros(clone.num_nodes, 2)
        return clone


class _EmptyOutputTeacher(torch.nn.Module, BaseModelMixin):
    """Teacher that declares an energy output but returns ``None`` for it."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}), autograd_inputs=frozenset()
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes."""
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Raise, since this teacher produces no embeddings."""
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs: Any) -> OrderedDict:  # noqa: ARG002
        """Return an output dict whose only entry is missing."""
        return OrderedDict([("energy", None)])


class _RaisingTeacher(torch.nn.Module, BaseModelMixin):
    """Neighbor-consuming teacher whose forward always raises."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=NeighborConfig(
                cutoff=_LJ_CUTOFF, format=NeighborListFormat.MATRIX
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes."""
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Raise, since this teacher produces no embeddings."""
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs: Any) -> OrderedDict:  # noqa: ARG002
        """Raise to exercise the scorer's rollback paths."""
        raise RuntimeError("teacher forward failed")


class TestInProcessTeacherScorerValidation:
    """Construction-time validation of the requested signal set."""

    def test_unknown_signal_name_raises(self, demo_teacher: Any) -> None:
        """An unrecognized signal name is rejected and named in the message."""
        with pytest.raises(ValueError, match="bogus"):
            InProcessTeacherScorer(demo_teacher, ["energy", "bogus"])

    def test_signal_the_teacher_cannot_produce_raises(self, demo_teacher: Any) -> None:
        """Requesting stress from an energy/forces teacher names the missing output."""
        with pytest.raises(ValueError, match="stress"):
            InProcessTeacherScorer(demo_teacher, ["energy", "stress"])

    def test_empty_signal_selection_raises(self, demo_teacher: Any) -> None:
        """A scorer with no signals is rejected."""
        with pytest.raises(ValueError, match="At least one teacher signal"):
            InProcessTeacherScorer(demo_teacher, [])

    def test_embeddings_from_teacher_without_embeddings_raises(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A teacher publishing no node-embedding shape cannot serve embeddings."""
        with pytest.raises(ValueError, match="node_embeddings"):
            InProcessTeacherScorer(lj_teacher, ["embeddings"])

    def test_scorer_satisfies_the_teacher_scorer_protocol(
        self, demo_teacher: Any
    ) -> None:
        """:class:`InProcessTeacherScorer` is a structural :class:`TeacherScorer`."""
        scorer = InProcessTeacherScorer(demo_teacher, ["energy"])
        assert isinstance(scorer, TeacherScorer)


class TestInProcessTeacherScorerLabeling:
    """Signal shapes, levels, and detachment produced by ``label()``."""

    def test_energy_and_forces_have_canonical_shapes_and_levels(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """Energy is ``(B, 1)`` system-level and forces are ``(V, 3)`` node-level."""
        labels = InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(
            small_batch
        )
        assert labels["teacher_energy"][0].shape == (small_batch.num_graphs, 1)
        assert labels["teacher_energy"][1] == "system"
        assert labels["teacher_forces"][0].shape == (small_batch.num_nodes, 3)
        assert labels["teacher_forces"][1] == "node"

    def test_returned_tensors_are_detached(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """No returned tensor carries an autograd graph."""
        labels = InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(
            small_batch
        )
        assert all(value.requires_grad is False for value, _ in labels.values())

    def test_label_leaves_the_batch_unmodified(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """Scoring changes no stored tensor, key set, or ``requires_grad`` flag."""
        before = _storage_snapshot(small_batch)
        flags = {key: value.requires_grad for key, value in small_batch}
        tracked = _tracked_snapshot(small_batch)
        InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(small_batch)
        after = {key: value for key, value in small_batch}
        assert set(after) == set(before)
        assert all(torch.equal(after[key], before[key]) for key in before)
        assert {key: value.requires_grad for key, value in after.items()} == flags
        assert _tracked_snapshot(small_batch) == tracked

    def test_labels_match_a_direct_teacher_forward(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """Scored values equal the teacher's own forward outputs."""
        labels = InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(
            small_batch
        )
        expected = demo_teacher(small_batch)
        torch.testing.assert_close(labels["teacher_energy"][0], expected["energy"])
        torch.testing.assert_close(labels["teacher_forces"][0], expected["forces"])

    def test_direct_force_teacher_labels_energy_and_forces(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """A teacher with no autograd outputs still yields detached forces."""
        labels = InProcessTeacherScorer(
            direct_force_teacher, ["energy", "forces"]
        ).label(small_batch)
        forces, level = labels["teacher_forces"]
        assert forces.shape == (small_batch.num_nodes, 3)
        assert level == "node"
        assert forces.requires_grad is False

    def test_direct_force_teacher_leaves_positions_grad_free(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """A non-autograd teacher never enables gradients on positions."""
        InProcessTeacherScorer(direct_force_teacher, ["energy", "forces"]).label(
            small_batch
        )
        assert small_batch.positions.requires_grad is False

    def test_autograd_teacher_restores_cleared_positions_grad_flag(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """A flag the teacher enabled for autograd forces is cleared again."""
        InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(small_batch)
        assert small_batch.positions.requires_grad is False

    def test_incoming_positions_grad_flag_survives_a_non_autograd_teacher(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """A live student graph on the batch is intact after labeling."""
        small_batch.positions.requires_grad_(True)
        student_energy = (small_batch.positions**2).sum()
        InProcessTeacherScorer(direct_force_teacher, ["energy", "forces"]).label(
            small_batch
        )
        assert small_batch.positions.requires_grad is True
        gradient = torch.autograd.grad(student_energy, small_batch.positions)[0]
        torch.testing.assert_close(gradient, 2.0 * small_batch.positions.detach())

    def test_incoming_positions_grad_flag_survives_an_autograd_teacher(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """An autograd-force teacher also hands back the caller's grad flag."""
        small_batch.positions.requires_grad_(True)
        InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(small_batch)
        assert small_batch.positions.requires_grad is True

    def test_active_outputs_are_restored_after_label(
        self, lj_teacher: LennardJonesModelWrapper, periodic_batch: Batch
    ) -> None:
        """A teacher whose active set is narrower than its outputs keeps that set."""
        before = set(lj_teacher.model_config.active_outputs)
        assert before != set(lj_teacher.model_config.outputs)
        InProcessTeacherScorer(lj_teacher, ["energy"]).label(periodic_batch)
        assert lj_teacher.model_config.active_outputs == before

    def test_active_outputs_are_restored_when_the_teacher_raises(self) -> None:
        """A failing forward still restores the config and rolls back neighbors."""
        teacher = _RaisingTeacher()
        batch = _make_spread_batch()
        before = set(teacher.model_config.active_outputs)
        with pytest.raises(RuntimeError, match="teacher forward failed"):
            InProcessTeacherScorer(teacher, ["energy"]).label(batch)
        assert teacher.model_config.active_outputs == before
        assert "neighbor_matrix" not in batch
        assert not hasattr(batch, "_neighbor_list_cutoff")

    def test_energy_only_scorer_does_not_request_forces(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """An energy-only scorer narrows the teacher so no force pass runs."""
        scorer = InProcessTeacherScorer(demo_teacher, ["energy"])
        with patch.object(
            demo_teacher.model, "forward", wraps=demo_teacher.model.forward
        ) as spy:
            labels = scorer.label(small_batch)
        assert spy.call_args.kwargs["compute_forces"] is False
        assert set(labels) == {"teacher_energy"}

    def test_node_energies_are_flattened_to_one_dimension(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """Per-atom energies are normalized to ``(V,)``."""
        labels = InProcessTeacherScorer(direct_force_teacher, ["node_energies"]).label(
            small_batch
        )
        values, level = labels["teacher_node_energies"]
        assert values.shape == (small_batch.num_nodes,)
        assert level == "node"

    def test_stress_is_labeled_as_a_system_level_matrix(
        self, lj_teacher: LennardJonesModelWrapper, periodic_batch: Batch
    ) -> None:
        """Stress is normalized to ``(B, 3, 3)`` and matches a direct forward."""
        reference = _build_periodic_batch()
        compute_neighbors(reference, config=lj_teacher.model_config.neighbor_config)
        lj_teacher.set_config("active_outputs", {"stress"})
        expected = lj_teacher(reference)["stress"]
        lj_teacher.set_config("active_outputs", {"energy", "forces"})
        labels = InProcessTeacherScorer(lj_teacher, ["stress"]).label(periodic_batch)
        values, level = labels["teacher_stress"]
        assert values.shape == (periodic_batch.num_graphs, 3, 3)
        assert level == "system"
        torch.testing.assert_close(values, expected)

    def test_cast_to_casts_floating_point_outputs(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """``cast_to`` changes the dtype of every floating-point signal."""
        labels = InProcessTeacherScorer(
            demo_teacher, ["energy", "forces"], cast_to=torch.float64
        ).label(small_batch)
        assert all(value.dtype is torch.float64 for value, _ in labels.values())

    def test_missing_teacher_output_raises(self, small_batch: Batch) -> None:
        """A declared output the teacher does not return is reported by name."""
        scorer = InProcessTeacherScorer(_EmptyOutputTeacher(), ["energy"])
        with pytest.raises(RuntimeError, match="'energy'"):
            scorer.label(small_batch)


class TestInProcessTeacherScorerGradMode:
    """Grad mode selected for the teacher forward pass."""

    def test_autograd_teacher_runs_with_grad_enabled(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """An autograd-force teacher gets a grad-enabled forward even under no_grad."""
        recorder = _GradModeRecorder(demo_teacher.model.forward)
        scorer = InProcessTeacherScorer(demo_teacher, ["energy", "forces"])
        with patch.object(demo_teacher.model, "forward", recorder), torch.no_grad():
            scorer.label(small_batch)
        assert recorder.grad_enabled == [True]

    def test_non_autograd_scorer_runs_under_no_grad(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """An energy-only scorer forwards with grad disabled."""
        recorder = _GradModeRecorder(demo_teacher.model.forward)
        scorer = InProcessTeacherScorer(demo_teacher, ["energy"])
        with patch.object(demo_teacher.model, "forward", recorder):
            scorer.label(small_batch)
        assert recorder.grad_enabled == [False]

    def test_labeling_inside_no_grad_still_yields_autograd_forces(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """Forces scored inside ``torch.no_grad()`` equal an ordinary forward."""
        expected = demo_teacher(small_batch.clone())["forces"].detach()
        with torch.no_grad():
            labels = InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(
                small_batch
            )
        torch.testing.assert_close(labels["teacher_forces"][0], expected)


class TestInProcessTeacherScorerEmbeddings:
    """Embedding signal extraction and batch restoration."""

    def test_embeddings_signal_returns_node_embeddings(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """The embeddings signal is a node-level ``(V, D)`` tensor."""
        labels = InProcessTeacherScorer(direct_force_teacher, ["embeddings"]).label(
            small_batch
        )
        values, level = labels["teacher_node_embeddings"]
        hidden_dim = direct_force_teacher.embedding_shapes["node_embeddings"][0]
        assert values.shape == (small_batch.num_nodes, hidden_dim)
        assert level == "node"
        assert values.requires_grad is False

    def test_batch_keeps_no_embedding_keys_afterwards(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """``compute_embeddings`` output is removed from the batch again."""
        InProcessTeacherScorer(direct_force_teacher, ["embeddings"]).label(small_batch)
        assert "node_embeddings" not in small_batch
        assert "graph_embeddings" not in small_batch

    def test_pre_existing_node_embeddings_are_restored(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """Embeddings already on the batch survive scoring unchanged."""
        existing = torch.arange(small_batch.num_nodes * 4, dtype=torch.float32).reshape(
            small_batch.num_nodes, 4
        )
        small_batch._atoms_group["node_embeddings"] = existing
        InProcessTeacherScorer(direct_force_teacher, ["embeddings"]).label(small_batch)
        torch.testing.assert_close(small_batch.node_embeddings, existing)

    def test_add_key_teacher_leaves_both_embedding_levels_clean(
        self, small_batch: Batch
    ) -> None:
        """A teacher writing both levels via ``add_key`` leaves neither behind."""
        before = _storage_snapshot(small_batch)
        tracked = _tracked_snapshot(small_batch)
        labels = InProcessTeacherScorer(
            _AddKeyEmbeddingTeacher(), ["embeddings"]
        ).label(small_batch)
        assert labels["teacher_node_embeddings"][0].shape == (small_batch.num_nodes, 4)
        assert "node_embeddings" not in small_batch
        assert "graph_embeddings" not in small_batch
        assert {key for key, _ in small_batch} == set(before)
        assert _tracked_snapshot(small_batch) == tracked

    def test_registered_embeddings_stay_registered_and_unchanged(
        self, small_batch: Batch
    ) -> None:
        """Embeddings registered in ``batch.keys`` come back with the same values."""
        existing = torch.full((small_batch.num_nodes, 4), 7.0)
        small_batch.add_key(
            "node_embeddings",
            list(torch.split(existing, small_batch.num_nodes_list)),
            level="node",
        )
        InProcessTeacherScorer(_AddKeyEmbeddingTeacher(), ["embeddings"]).label(
            small_batch
        )
        assert "node_embeddings" in small_batch.keys["node"]
        torch.testing.assert_close(small_batch.node_embeddings, existing)

    def test_teacher_that_does_not_mutate_the_batch_raises(
        self, small_batch: Batch
    ) -> None:
        """Embeddings returned on a copy instead of written in place are an error."""
        scorer = InProcessTeacherScorer(_NonMutatingEmbeddingTeacher(), ["embeddings"])
        with pytest.raises(RuntimeError, match="node_embeddings"):
            scorer.label(small_batch)
        assert "node_embeddings" not in small_batch


class TestInProcessTeacherScorerNeighborIsolation:
    """Neighbor-list construction and rollback around the teacher forward."""

    def test_prebuilt_matrix_neighbors_are_restored_exactly(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A list built at another cutoff is rebuilt for the teacher, then restored."""
        batch = _make_spread_batch()
        compute_neighbors(batch, cutoff=8.0, format=NeighborListFormat.MATRIX)
        expected_matrix = batch.neighbor_matrix.clone()
        expected_counts = batch.num_neighbors.clone()
        InProcessTeacherScorer(lj_teacher, ["energy"]).label(batch)
        assert batch._neighbor_list_cutoff == 8.0
        assert torch.equal(batch.neighbor_matrix, expected_matrix)
        assert torch.equal(batch.num_neighbors, expected_counts)

    def test_batch_without_neighbors_has_none_afterwards(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """Neighbor tensors built for the teacher do not leak onto a bare batch."""
        batch = _make_spread_batch()
        InProcessTeacherScorer(lj_teacher, ["energy"]).label(batch)
        assert "neighbor_matrix" not in batch
        assert "num_neighbors" not in batch
        assert not hasattr(batch, "_neighbor_list_cutoff")
        assert not hasattr(batch, "_neighbor_list_half")

    def test_matching_neighbor_list_is_reused(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A declared full list at the teacher's cutoff and format is not rebuilt."""
        batch = _make_spread_batch()
        _build_declared_neighbors(batch, _LJ_CUTOFF, NeighborListFormat.MATRIX)
        scorer = InProcessTeacherScorer(lj_teacher, ["energy"])
        with patch.object(
            scoring, "compute_neighbors", wraps=scoring.compute_neighbors
        ) as spy:
            scorer.label(batch)
        assert spy.call_count == 0

    def test_list_of_unknown_provenance_is_rebuilt(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A list matching on cutoff and format alone is not trusted for reuse."""
        batch = _make_spread_batch()
        compute_neighbors(batch, config=lj_teacher.model_config.neighbor_config)
        scorer = InProcessTeacherScorer(lj_teacher, ["energy"])
        with patch.object(
            scoring, "compute_neighbors", wraps=scoring.compute_neighbors
        ) as spy:
            scorer.label(batch)
        assert spy.call_count == 1

    def test_mismatched_neighbor_list_is_rebuilt(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A list at the wrong cutoff triggers exactly one rebuild."""
        batch = _make_spread_batch()
        compute_neighbors(batch, cutoff=8.0, format=NeighborListFormat.MATRIX)
        scorer = InProcessTeacherScorer(lj_teacher, ["energy"])
        with patch.object(
            scoring, "compute_neighbors", wraps=scoring.compute_neighbors
        ) as spy:
            scorer.label(batch)
        assert spy.call_count == 1

    def test_matrix_list_is_rebuilt_for_a_coo_teacher(self) -> None:
        """A dense list at the right cutoff is still rebuilt for a sparse teacher."""
        teacher = _CooNeighborTeacher()
        batch = _make_spread_batch()
        compute_neighbors(batch, cutoff=_COO_CUTOFF, format=NeighborListFormat.MATRIX)
        scorer = InProcessTeacherScorer(teacher, ["energy"])
        with patch.object(
            scoring, "compute_neighbors", wraps=scoring.compute_neighbors
        ) as spy:
            labels = scorer.label(batch)
        assert spy.call_count == 1
        assert labels["teacher_energy"][0].shape == (batch.num_graphs, 1)
        assert "neighbor_list" not in batch

    def test_coo_list_is_rebuilt_for_a_matrix_teacher(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A sparse list at the right cutoff is still rebuilt for a dense teacher."""
        batch = _make_spread_batch()
        compute_neighbors(batch, cutoff=_LJ_CUTOFF, format=NeighborListFormat.COO)
        scorer = InProcessTeacherScorer(lj_teacher, ["energy"])
        with patch.object(
            scoring, "compute_neighbors", wraps=scoring.compute_neighbors
        ) as spy:
            scorer.label(batch)
        assert spy.call_count == 1
        assert "neighbor_matrix" not in batch
        assert "neighbor_list" in batch

    def test_prebuilt_coo_neighbors_are_restored_exactly(self) -> None:
        """A pre-built edge group survives a COO teacher's rebuild untouched."""
        teacher = _CooNeighborTeacher()
        batch = _make_spread_batch()
        compute_neighbors(batch, cutoff=8.0, format=NeighborListFormat.COO)
        expected_edges = batch.neighbor_list.clone()
        expected_counts = list(batch.num_edges_list)
        scorer = InProcessTeacherScorer(teacher, ["energy"])
        with patch.object(
            scoring, "compute_neighbors", wraps=scoring.compute_neighbors
        ) as spy:
            scorer.label(batch)
        assert spy.call_count == 1
        assert batch._neighbor_list_cutoff == 8.0
        assert torch.equal(batch.neighbor_list, expected_edges)
        assert batch.num_edges_list == expected_counts

    def test_matching_coo_list_is_reused(self) -> None:
        """A declared full sparse list at the teacher's cutoff is not rebuilt."""
        teacher = _CooNeighborTeacher()
        batch = _make_spread_batch()
        _build_declared_neighbors(batch, _COO_CUTOFF, NeighborListFormat.COO)
        scorer = InProcessTeacherScorer(teacher, ["energy"])
        with patch.object(
            scoring, "compute_neighbors", wraps=scoring.compute_neighbors
        ) as spy:
            scorer.label(batch)
        assert spy.call_count == 0

    def test_half_list_teacher_never_reuses_a_full_list(self) -> None:
        """A half-list teacher rebuilds, so a full list cannot double-count pairs."""
        teacher = _build_lj_teacher(half_list=True)
        scorer = InProcessTeacherScorer(teacher, ["energy"])
        prebuilt = _make_spread_batch()
        _build_declared_neighbors(prebuilt, _LJ_CUTOFF, NeighborListFormat.MATRIX)
        with patch.object(
            scoring, "compute_neighbors", wraps=scoring.compute_neighbors
        ) as spy:
            reused = scorer.label(prebuilt)["teacher_energy"][0]
        fresh = scorer.label(_make_spread_batch())["teacher_energy"][0]
        assert spy.call_count == 1
        torch.testing.assert_close(reused, fresh)

    def test_full_list_teacher_never_reuses_a_half_list(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A full-list teacher rebuilds, so a half list cannot halve the energy."""
        scorer = InProcessTeacherScorer(lj_teacher, ["energy"])
        prebuilt = _make_spread_batch()
        _build_declared_neighbors(
            prebuilt, _LJ_CUTOFF, NeighborListFormat.MATRIX, half_list=True
        )
        with patch.object(
            scoring, "compute_neighbors", wraps=scoring.compute_neighbors
        ) as spy:
            reused = scorer.label(prebuilt)["teacher_energy"][0]
        fresh = scorer.label(_make_spread_batch())["teacher_energy"][0]
        assert spy.call_count == 1
        torch.testing.assert_close(reused, fresh)

    def test_periodic_neighbor_shifts_are_rolled_back(
        self, lj_teacher: LennardJonesModelWrapper, periodic_batch: Batch
    ) -> None:
        """Shift tensors built for a periodic teacher are removed again."""
        InProcessTeacherScorer(lj_teacher, ["energy"]).label(periodic_batch)
        assert "neighbor_matrix_shifts" not in periodic_batch
        assert "neighbor_matrix" not in periodic_batch
