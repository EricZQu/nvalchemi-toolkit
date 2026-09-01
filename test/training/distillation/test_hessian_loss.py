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
"""Tests for :mod:`nvalchemi.training.distillation.losses.hessian`.

Also covers the teacher side of the objective — the ``hessian`` signal and
:meth:`~nvalchemi.training.distillation.InProcessTeacherScorer.label_hvp` — whose
products this loss consumes.
"""

from __future__ import annotations

import json

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training.distillation import (
    HessianMatchingLoss,
    InProcessTeacherScorer,
    hessian_vector_product,
)
from nvalchemi.training.distillation._labels import _attach_teacher_labels
from nvalchemi.training.losses.composition import (
    ComposedLossFunction,
    compute_supervised_loss,
    loss_component_to_spec,
)
from nvalchemi.training.losses.terms import EnergyMSELoss
from test.training.conftest import _build_batch
from test.training.distillation.conftest import (
    _DirectForceTeacher,
    _PairPotentialTeacher,
)

_UNEVEN_BATCH_IDX = torch.tensor([0, 0, 0, 1])
"""Graph assignment of four atoms split three-to-one across two graphs."""

_FD_STEP = 1e-4
"""Displacement of the central finite difference the teacher product is checked against."""


def _energy_gradient(
    teacher: _DirectForceTeacher, positions: torch.Tensor
) -> torch.Tensor:
    """Return the gradient of the teacher's energy at *positions*."""
    batch = _build_batch()
    batch.positions = positions.clone().requires_grad_(True)
    energy = teacher(batch)["energy"]
    return torch.autograd.grad(energy.sum(), batch.positions)[0]


def _finite_difference_hvp(
    teacher: _DirectForceTeacher, positions: torch.Tensor, probe: torch.Tensor
) -> torch.Tensor:
    """Return a central-difference estimate of the teacher's product with *probe*."""
    forward = _energy_gradient(teacher, positions + _FD_STEP * probe)
    backward = _energy_gradient(teacher, positions - _FD_STEP * probe)
    return (forward - backward) / (2.0 * _FD_STEP)


class TestHessianVectorProduct:
    """The double-backward estimator both sides of the objective go through."""

    def test_product_matches_a_finite_difference_reference(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """Autograd curvature reproduces a central difference of the energy gradient."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["hessian"])
        probe = torch.randn_like(small_batch.positions)
        product = scorer.label_hvp(small_batch, probe)
        direct_force_teacher.set_config("active_outputs", {"energy"})
        reference = _finite_difference_hvp(
            direct_force_teacher, small_batch.positions.detach(), probe
        )
        torch.testing.assert_close(product, reference, atol=1e-3, rtol=1e-2)

    def test_product_of_a_quadratic_energy_is_the_constant_hessian(self) -> None:
        """For ``E = c |r|^2 / 2`` the product is ``c v``, exactly."""
        positions = torch.randn(4, 3, requires_grad=True)
        probe = torch.randn(4, 3)
        energy = (3.0 * positions.pow(2).sum()).reshape(1, 1) / 2.0
        product = hessian_vector_product(energy, positions, probe)
        torch.testing.assert_close(product, 3.0 * probe)

    def test_product_is_block_diagonal_over_graphs(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """A probe on one graph's atoms leaves the other graph's product zero."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["hessian"])
        probe = torch.ones_like(small_batch.positions)
        probe[small_batch.batch_idx == 1] = 0.0
        product = scorer.label_hvp(small_batch, probe)
        assert bool((product[small_batch.batch_idx == 1] == 0.0).all())

    def test_detached_energy_is_reported_as_a_missing_graph(self) -> None:
        """Differentiating an energy with no graph names what the estimator needs."""
        positions = torch.randn(4, 3, requires_grad=True)
        with pytest.raises(RuntimeError, match="twice differentiable"):
            hessian_vector_product(torch.zeros(1, 1), positions, torch.randn(4, 3))

    def test_created_graph_keeps_the_product_differentiable(self) -> None:
        """``create_graph=True`` is what lets a loss backpropagate through it."""
        positions = torch.randn(4, 3, requires_grad=True)
        weight = torch.tensor(2.0, requires_grad=True)
        energy = (weight * positions.pow(2).sum()).reshape(1, 1)
        product = hessian_vector_product(
            energy, positions, torch.ones(4, 3), create_graph=True
        )
        product.sum().backward()
        assert weight.grad is not None


class TestHessianSignalLabeling:
    """The ``hessian`` teacher signal and the two fields it materializes."""

    def test_signal_writes_the_product_and_the_probe(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """Both fields arrive at node level, shaped like positions."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["hessian"])
        labels = scorer.label(small_batch)
        assert set(labels) == {"teacher_hvp", "teacher_hvp_probe"}
        for values, level in labels.values():
            assert level == "node"
            assert values.shape == small_batch.positions.shape

    def test_stored_probe_reproduces_the_stored_product(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """Relabeling along the stored probe returns the stored product."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["hessian"])
        labels = scorer.label(small_batch)
        probe = labels["teacher_hvp_probe"][0]
        torch.testing.assert_close(
            scorer.label_hvp(small_batch, probe), labels["teacher_hvp"][0]
        )

    def test_neighbor_list_teacher_is_scored_and_rolled_back(
        self, periodic_batch: Batch, pair_potential_teacher: _PairPotentialTeacher
    ) -> None:
        """A teacher needing its own neighbor list gets one for the second derivative.

        The product is taken inside the same isolation the forward-pass signals
        use, so the list built for it is rolled back with theirs.
        """
        scorer = InProcessTeacherScorer(
            pair_potential_teacher, ["energy", "forces", "hessian"]
        )
        labels = scorer.label(periodic_batch)
        assert labels["teacher_hvp"][0].shape == periodic_batch.positions.shape
        assert "neighbor_matrix" not in periodic_batch

    def test_labeled_batch_is_left_as_it_was_found(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """Scoring restores the grad flags it enabled to differentiate twice."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["energy", "hessian"])
        scorer.label(small_batch)
        assert small_batch.positions.requires_grad is False
        assert "teacher_hvp" not in small_batch

    def test_labels_are_detached(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """A stored product holds no autograd graph back to the teacher."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["hessian"])
        labels = scorer.label(small_batch)
        assert labels["teacher_hvp"][0].requires_grad is False

    def test_labels_are_cast_to_the_requested_dtype(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """``cast_to`` reaches the product and the probe alike."""
        scorer = InProcessTeacherScorer(
            direct_force_teacher, ["hessian"], cast_to=torch.float64
        )
        labels = scorer.label(small_batch)
        assert labels["teacher_hvp"][0].dtype == torch.float64
        assert labels["teacher_hvp_probe"][0].dtype == torch.float64

    def test_teacher_without_energy_is_rejected(
        self, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """A teacher declaring no energy cannot serve the signal at all."""
        direct_force_teacher.model_config.outputs = frozenset({"forces"})
        with pytest.raises(ValueError, match="must declare an ``energy`` output"):
            InProcessTeacherScorer(direct_force_teacher, ["hessian"])


class TestHessianMatchingLossValues:
    """Scalar values the Hessian matching loss produces."""

    def test_graph_balanced_value_matches_a_hand_computation(self) -> None:
        """Each graph's mean squared component error is averaged over graphs."""
        loss_fn = HessianMatchingLoss()
        pred = torch.ones(4, 3)
        pred[3] = 2.0
        loss = loss_fn(
            pred, torch.zeros(4, 3), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2
        )
        assert loss.item() == pytest.approx((1.0 + 4.0) / 2)

    def test_global_mean_weights_every_component_equally(self) -> None:
        """Without graph balancing the loss is one mean over all components."""
        loss_fn = HessianMatchingLoss(normalize_by_atom_count=False)
        pred = torch.ones(4, 3)
        pred[3] = 2.0
        loss = loss_fn(
            pred, torch.zeros(4, 3), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2
        )
        assert loss.item() == pytest.approx((9 * 1.0 + 3 * 4.0) / 12)

    def test_zero_residual_gives_zero_loss(self) -> None:
        """A student reproducing the teacher's curvature scores zero."""
        loss_fn = HessianMatchingLoss()
        target = torch.randn(4, 3)
        loss = loss_fn(
            target.clone(), target, batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2
        )
        assert loss.item() == pytest.approx(0.0)

    def test_nonfinite_targets_are_excluded_from_the_loss(self) -> None:
        """A product that overflowed on one component drops only that component."""
        loss_fn = HessianMatchingLoss()
        target = torch.zeros(2, 3)
        target[0, 0] = float("inf")
        loss = loss_fn(
            torch.ones(2, 3), target, batch_idx=torch.tensor([0, 1]), num_graphs=2
        )
        assert loss.item() == pytest.approx(1.0)

    def test_per_sample_loss_holds_one_value_per_graph(self) -> None:
        """The graph-balanced reduction publishes a detached ``(B,)`` diagnostic."""
        loss_fn = HessianMatchingLoss()
        pred = torch.ones(4, 3)
        pred[3] = 2.0
        loss_fn(pred, torch.zeros(4, 3), batch_idx=_UNEVEN_BATCH_IDX, num_graphs=2)
        torch.testing.assert_close(loss_fn.per_sample_loss, torch.tensor([1.0, 4.0]))


class TestHessianMatchingLossContract:
    """Shape, metadata, and serialization contract of the loss term."""

    def test_loss_declares_it_needs_evaluation_gradients(self) -> None:
        """The student's product is a derivative, so validation must keep grads."""
        assert HessianMatchingLoss().requires_eval_grad is True

    def test_composed_loss_inherits_the_gradient_requirement(self) -> None:
        """One curvature term forces gradient-enabled evaluation for the whole loss."""
        loss_fn = EnergyMSELoss(target_key="teacher_energy") + HessianMatchingLoss()
        assert loss_fn.requires_eval_grad() is True

    def test_default_keys_match_the_teacher_and_student_fields(self) -> None:
        """Defaults line up the teacher's stored product with the student's."""
        loss_fn = HessianMatchingLoss()
        assert loss_fn.target_key == "teacher_hvp"
        assert loss_fn.prediction_key == "predicted_hvp"

    def test_shape_mismatch_is_rejected(self) -> None:
        """Products of different node counts are refused rather than broadcast."""
        loss_fn = HessianMatchingLoss()
        with pytest.raises(ValueError, match="shape must match exactly"):
            loss_fn(
                torch.zeros(3, 3),
                torch.zeros(4, 3),
                batch_idx=_UNEVEN_BATCH_IDX,
                num_graphs=2,
            )

    def test_missing_graph_metadata_is_rejected(self) -> None:
        """The graph-balanced reduction names the metadata it needs."""
        loss_fn = HessianMatchingLoss()
        with pytest.raises(ValueError, match="batch_idx"):
            loss_fn(torch.zeros(4, 3), torch.zeros(4, 3))

    def test_spec_round_trip_rebuilds_an_equivalent_loss(self) -> None:
        """A JSON round-tripped spec rebuilds the loss with its configuration."""
        spec = loss_component_to_spec(
            HessianMatchingLoss(normalize_by_atom_count=False, ignore_nonfinite=False)
        )
        rebuilt = create_model_spec_from_json(
            json.loads(spec.model_dump_json())
        ).build()
        assert isinstance(rebuilt, HessianMatchingLoss)
        assert rebuilt.normalize_by_atom_count is False
        assert rebuilt.ignore_nonfinite is False

    def test_teacher_products_score_zero_against_teacher_labels(
        self, small_batch: Batch, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """Labels read back as predictions reproduce the teacher exactly."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["hessian"])
        _attach_teacher_labels(small_batch, scorer.label(small_batch))
        predictions = {"predicted_hvp": small_batch.teacher_hvp.clone()}
        out = compute_supervised_loss(
            ComposedLossFunction([HessianMatchingLoss()]),
            predictions,
            small_batch,
            step=0,
            epoch=0,
        )
        assert out["total_loss"].item() == pytest.approx(0.0)
