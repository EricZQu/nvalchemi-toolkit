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
"""Tests for :mod:`nvalchemi.training.distillation.evaluation.accuracy`."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training.distillation.evaluation import (
    evaluate_accuracy,
    nonconservative_residual,
)
from nvalchemi.training.losses.terms import EnergyMSELoss
from nvalchemi.training.strategy import default_training_fn
from test.training.conftest import _build_batch, _build_demo_model
from test.training.distillation.conftest import (
    _build_direct_force_teacher,
    _build_lattice_batch,
    _build_lj_teacher,
)


def _make_holdout(sizes: tuple[int, ...] = (2, 5)) -> list[Batch]:
    """Return one batch per entry of *sizes*, so per-batch means differ."""
    return [
        _build_batch(n_systems=1, n_atoms_each=size, seed=40 + index)
        for index, size in enumerate(sizes)
    ]


def _make_lj_batch(n_atoms: int = 6, cell_length: float = 20.0) -> Batch:
    """Return one periodic argon batch with a Lennard-Jones neighbor list built."""
    generator = torch.Generator().manual_seed(11)
    data = AtomicData(
        positions=torch.rand(n_atoms, 3, generator=generator) * 8.0,
        atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),
        atomic_masses=torch.full((n_atoms,), 39.948),
        cell=torch.eye(3).unsqueeze(0) * cell_length,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    batch = Batch.from_data_list([data])
    compute_neighbors(batch, cutoff=5.0)
    return batch


def _reference_force_error(model: Any, batches: list[Batch]) -> tuple[float, float]:
    """Return the hand-computed global force MAE and RMSE over *batches*."""
    absolute = 0.0
    squared = 0.0
    count = 0
    for batch in batches:
        predicted = default_training_fn(model, batch)["predicted_forces"]
        residual = (predicted - batch.forces).detach()
        absolute += float(residual.abs().sum())
        squared += float(residual.pow(2).sum())
        count += residual.numel()
    return absolute / count, (squared / count) ** 0.5


class _SignallessScorer:
    """Scorer that declares a signal set with no forces in it."""

    signals = frozenset({"energy"})

    def label(self, batch: Batch) -> dict[str, Any]:  # noqa: ARG002
        """Return no labels; the evaluation never gets this far."""
        return {}


class TestEvaluateAccuracy:
    """Accuracy metrics of a student against reference and teacher targets."""

    def test_student_scored_against_itself_reports_zero_error(self) -> None:
        """A teacher that is the student gives zero error and perfect alignment."""
        student = _build_demo_model()
        metrics = evaluate_accuracy(
            student, _make_holdout(), targets="teacher", scorer=student
        )
        assert metrics.energy_mae == 0.0
        assert metrics.forces_rmse == 0.0
        assert metrics.force_cosine_mean == pytest.approx(1.0)
        assert metrics.force_cosine_aggregate == pytest.approx(1.0)

    def test_force_errors_match_hand_computed_global_sums(self) -> None:
        """Force MAE and RMSE equal the residual sums divided by component count."""
        student = _build_demo_model()
        holdout = _make_holdout()
        expected_mae, expected_rmse = _reference_force_error(student, holdout)
        metrics = evaluate_accuracy(student, holdout)
        assert metrics.forces_mae == pytest.approx(expected_mae, rel=1e-6)
        assert metrics.forces_rmse == pytest.approx(expected_rmse, rel=1e-6)

    def test_metrics_do_not_depend_on_how_the_holdout_is_batched(self) -> None:
        """The same systems split into one or two batches give the same metrics."""
        student = _build_demo_model()
        split = _make_holdout()
        joined = [
            Batch.from_data_list([*split[0].to_data_list(), *split[1].to_data_list()])
        ]
        assert evaluate_accuracy(student, split).to_dict() == pytest.approx(
            evaluate_accuracy(student, joined).to_dict(), rel=1e-6
        )

    def test_counts_cover_every_graph_and_atom(self) -> None:
        """Reported graph and atom counts are the totals over the whole holdout."""
        metrics = evaluate_accuracy(_build_demo_model(), _make_holdout(sizes=(2, 5)))
        assert metrics.num_graphs == 2
        assert metrics.num_atoms == 7

    def test_scaled_teacher_is_perfectly_aligned_but_wrong_in_magnitude(self) -> None:
        """Doubling the teacher's epsilon leaves cosine at one and MAE at |F|."""
        student = _build_lj_teacher()
        student.set_config("active_outputs", {"energy", "forces", "stress"})
        teacher = LennardJonesModelWrapper(epsilon=0.02, sigma=3.4, cutoff=5.0)
        batch = _make_lj_batch()
        predictions = default_training_fn(student, batch)
        metrics = evaluate_accuracy(
            student,
            [batch],
            targets="teacher",
            scorer=teacher,
            quantities=("energy", "forces", "stress"),
        )
        assert metrics.force_cosine_mean == pytest.approx(1.0)
        assert metrics.forces_mae == pytest.approx(
            float(predictions["predicted_forces"].abs().mean()), rel=1e-5
        )
        assert metrics.stress_mae == pytest.approx(
            float(predictions["predicted_stress"].abs().mean()), rel=1e-5
        )

    def test_atomic_energy_residuals_fill_in_when_both_sides_publish_them(self) -> None:
        """A student and teacher with energy heads report per-atom energy error."""
        student = _build_direct_force_teacher(seed=1)
        teacher = _build_direct_force_teacher(seed=2)
        metrics = evaluate_accuracy(
            student,
            _make_holdout(),
            targets="teacher",
            scorer=teacher,
            quantities=("energy", "forces", "atomic_energies"),
        )
        assert metrics.atomic_energy_mae is not None
        assert metrics.atomic_energy_rmse >= metrics.atomic_energy_mae

    def test_unknown_quantity_is_rejected(self) -> None:
        """A misspelled quantity raises before any forward pass."""
        with pytest.raises(ValueError, match="unsupported"):
            evaluate_accuracy(
                _build_demo_model(), _make_holdout(), quantities=("dipole",)
            )

    def test_evaluation_without_a_supervised_quantity_is_rejected(self) -> None:
        """Diagnostic-only quantities leave the pass with no loss to run."""
        with pytest.raises(ValueError, match="must be evaluated"):
            evaluate_accuracy(
                _build_demo_model(), _make_holdout(), quantities=("atomic_energies",)
            )

    def test_measuring_nothing_at_all_raises(self) -> None:
        """A quantity neither side carries leaves no metric to report."""
        with pytest.raises(ValueError, match="No accuracy metric"):
            evaluate_accuracy(
                _build_demo_model(),
                _make_holdout(),
                quantities=("atomic_energies",),
                loss_fn=EnergyMSELoss(),
                grad_mode="enabled",
            )

    def test_a_target_of_the_wrong_shape_is_rejected(self) -> None:
        """A mismatched target would broadcast into meaningless residuals."""
        with pytest.raises(ValueError, match="same shape"):
            evaluate_accuracy(
                _build_direct_force_teacher(),
                _make_holdout(),
                quantities=("atomic_energies",),
                target_keys={"atomic_energies": "energy"},
                loss_fn=EnergyMSELoss(),
                grad_mode="enabled",
            )

    def test_scorer_missing_a_requested_signal_is_rejected(self) -> None:
        """A scorer that cannot produce forces cannot back a force evaluation."""
        with pytest.raises(ValueError, match="missing"):
            evaluate_accuracy(
                _build_demo_model(),
                _make_holdout(),
                targets="teacher",
                scorer=_SignallessScorer(),
            )

    def test_explicit_target_keys_override_the_selected_family(self) -> None:
        """A target-key override points one quantity at any batch field."""
        student = _build_demo_model()
        holdout = _make_holdout()
        overridden = evaluate_accuracy(
            student,
            holdout,
            targets="teacher",
            scorer=student,
            target_keys={"energy": "energy"},
        )
        assert overridden.energy_mae > 0.0
        assert overridden.forces_mae == 0.0


class TestNonConservativeResidual:
    """The residual floor separating conservative from direct-force teachers."""

    def test_conservative_teacher_reports_a_numerical_noise_floor(self) -> None:
        """An autograd-force teacher's loops close to within float32 noise."""
        residual = nonconservative_residual(
            _build_demo_model(), _build_batch(n_atoms_each=4, seed=3), num_loops=2
        )
        assert residual.relative_floor < 1e-6

    def test_direct_force_teacher_reports_a_nonzero_floor(self) -> None:
        """A force head that is not an energy gradient leaves work in every loop."""
        batch = _build_batch(n_atoms_each=4, seed=3)
        conservative = nonconservative_residual(_build_demo_model(), batch, num_loops=2)
        direct = nonconservative_residual(
            _build_direct_force_teacher(), batch, num_loops=2
        )
        assert direct.relative_floor > 1e-5
        assert direct.force_floor > 100.0 * conservative.force_floor

    def test_floor_scales_linearly_with_the_probe_amplitude(self) -> None:
        """Doubling the loop side doubles the reported force floor."""
        teacher = _build_direct_force_teacher()
        batch = _build_batch(n_atoms_each=4, seed=3)
        small = nonconservative_residual(
            teacher,
            batch,
            num_loops=3,
            amplitude=0.02,
            generator=torch.Generator().manual_seed(0),
        )
        large = nonconservative_residual(
            teacher,
            batch,
            num_loops=3,
            amplitude=0.04,
            generator=torch.Generator().manual_seed(0),
        )
        assert large.force_floor / small.force_floor == pytest.approx(2.0, rel=0.05)

    def test_probing_restores_the_positions_it_displaced(self) -> None:
        """The probed batch is left exactly as it arrived."""
        batch = _build_batch(n_atoms_each=4, seed=3)
        original = batch.positions.clone()
        nonconservative_residual(_build_direct_force_teacher(), batch, num_loops=2)
        torch.testing.assert_close(batch.positions, original)

    def test_a_periodic_teacher_needing_neighbors_is_probed_too(self) -> None:
        """A neighbor-list teacher is probed through its scorer's own rebuilds."""
        batch = _build_lattice_batch(jitter=0.2)
        residual = nonconservative_residual(
            _build_lj_teacher(), batch, num_loops=2, amplitude=0.02
        )
        assert residual.relative_floor < 1e-4
        assert "neighbor_matrix" not in batch

    @pytest.mark.parametrize(
        ("amplitude", "num_loops", "segments"),
        [(0.0, 2, 4), (0.05, 0, 4), (0.05, 2, 0)],
        ids=["zero-amplitude", "no-loops", "no-segments"],
    )
    def test_degenerate_probe_geometry_is_rejected(
        self, amplitude: float, num_loops: int, segments: int
    ) -> None:
        """A probe with no extent, no loops, or no samples raises."""
        with pytest.raises(ValueError, match="must all be positive"):
            nonconservative_residual(
                _build_demo_model(),
                _build_batch(),
                amplitude=amplitude,
                num_loops=num_loops,
                segments=segments,
            )

    def test_scorer_without_forces_is_rejected(self) -> None:
        """The probe integrates forces, so a scorer without them cannot serve it."""
        with pytest.raises(ValueError, match="missing"):
            nonconservative_residual(_SignallessScorer(), _build_batch())
