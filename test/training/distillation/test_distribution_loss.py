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
"""Tests for :mod:`nvalchemi.training.distillation.losses.distribution`."""

from __future__ import annotations

import json
import math

import pytest
import torch

from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.training._spec import create_model_spec_from_json
from nvalchemi.training.distillation import BoltzmannMatchingLoss
from nvalchemi.training.losses.composition import loss_component_to_spec

_TEMPERATURE = 300.0
"""Ensemble temperature every hand computation here is done at."""

_KT = KB_EV * _TEMPERATURE
"""Thermal energy in eV at :data:`_TEMPERATURE`."""

# Teacher-minus-student energies of two configurations, chosen so the teacher's
# self-normalized weights come out as the exact fractions 3/4 and 1/4.
_TWO_STATE_GAP = _KT * math.log(3.0)
"""Energy difference giving teacher weights of ``3/4`` and ``1/4``."""

_FORWARD_KL = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
"""``D_KL(p||q)`` for weights ``(3/4, 1/4)`` against a uniform pair."""

_REVERSE_KL = -0.5 * (math.log(1.5) + math.log(0.5))
"""``D_KL(q||p)`` for the same pair."""


def _two_state_energies() -> tuple[torch.Tensor, torch.Tensor]:
    """Return student and teacher energies whose weights are ``(3/4, 1/4)``."""
    return torch.zeros(2, 1), torch.tensor([[0.0], [_TWO_STATE_GAP]])


class TestBoltzmannMatchingLossValues:
    """Scalar values the beta-interpolated relative entropy produces."""

    @pytest.mark.parametrize(
        ("beta", "expected"),
        [
            pytest.param(0.0, _FORWARD_KL, id="forward"),
            pytest.param(1.0, _REVERSE_KL, id="reverse"),
            pytest.param(0.5, 0.5 * (_FORWARD_KL + _REVERSE_KL), id="symmetric"),
        ],
    )
    def test_value_matches_a_hand_computed_relative_entropy(
        self, beta: float, expected: float
    ) -> None:
        """Both endpoints and their midpoint reproduce the closed-form divergence."""
        pred, target = _two_state_energies()
        loss_fn = BoltzmannMatchingLoss(beta=beta, temperature=_TEMPERATURE)
        assert loss_fn(pred, target).item() == pytest.approx(expected, rel=1e-5)

    def test_matching_energies_give_zero(self) -> None:
        """A student reproducing the teacher's energies scores zero."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        energies = torch.tensor([[0.0], [0.1], [-0.2]])
        assert loss_fn(energies.clone(), energies).item() == pytest.approx(0.0)

    def test_constant_energy_offset_is_invisible(self) -> None:
        """Only relative populations matter, so a shifted student still scores zero."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        target = torch.tensor([[0.0], [0.1], [-0.2]])
        assert loss_fn(target + 5.0, target).item() == pytest.approx(0.0)

    def test_loss_is_non_negative_for_random_energies(self) -> None:
        """Both directions are relative entropies, so neither can go negative."""
        loss_fn = BoltzmannMatchingLoss(beta=0.5, temperature=_TEMPERATURE)
        for seed in range(5):
            generator = torch.Generator().manual_seed(seed)
            pred = torch.randn(6, 1, generator=generator) * _KT
            target = torch.randn(6, 1, generator=generator) * _KT
            assert loss_fn(pred, target).item() >= 0.0

    def test_a_single_configuration_carries_no_signal(self) -> None:
        """One sample is one distribution of one state, which always matches."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        assert loss_fn(torch.zeros(1, 1), torch.ones(1, 1)).item() == pytest.approx(0.0)

    def test_lower_temperature_sharpens_the_divergence(self) -> None:
        """The same energy gap is a larger population difference when it is colder."""
        pred, target = _two_state_energies()
        cold = BoltzmannMatchingLoss(temperature=_TEMPERATURE / 3.0)
        warm = BoltzmannMatchingLoss(temperature=_TEMPERATURE * 3.0)
        assert cold(pred, target).item() > warm(pred, target).item()

    def test_per_sample_loss_averages_to_the_scalar_loss(self) -> None:
        """The per-graph diagnostic decomposes the value it is published beside."""
        pred, target = _two_state_energies()
        loss_fn = BoltzmannMatchingLoss(beta=0.5, temperature=_TEMPERATURE)
        loss = loss_fn(pred, target)
        assert loss_fn.per_sample_loss.shape == (2,)
        assert loss_fn.per_sample_loss.mean().item() == pytest.approx(loss.item())

    def test_gradients_flow_to_the_student_energies(self) -> None:
        """The student's energies receive gradient, and only their spread matters."""
        target = torch.tensor([[0.0], [_TWO_STATE_GAP], [-_TWO_STATE_GAP]])
        pred = torch.zeros(3, 1, requires_grad=True)
        BoltzmannMatchingLoss(temperature=_TEMPERATURE)(pred, target).backward()
        assert pred.grad is not None
        assert float(pred.grad.sum()) == pytest.approx(0.0, abs=1e-4)


class TestBoltzmannMatchingLossMasking:
    """Which configurations enter the ensemble."""

    def test_nonfinite_target_drops_its_configuration(self) -> None:
        """A masked graph leaves the others normalized among themselves."""
        pred, target = _two_state_energies()
        pred = torch.cat([pred, torch.zeros(1, 1)])
        target = torch.cat([target, torch.full((1, 1), float("nan"))])
        loss_fn = BoltzmannMatchingLoss(beta=0.0, temperature=_TEMPERATURE)
        assert loss_fn(pred, target).item() == pytest.approx(_FORWARD_KL, rel=1e-5)

    def test_fully_masked_ensemble_contributes_zero(self) -> None:
        """No valid configuration is no distribution, which scores zero."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        target = torch.full((2, 1), float("nan"))
        assert loss_fn(torch.zeros(2, 1), target).item() == pytest.approx(0.0)


class TestBoltzmannMatchingLossContract:
    """Configuration, ensemble, and serialization contract of the loss term."""

    @pytest.mark.parametrize("beta", [-0.1, 1.5])
    def test_beta_outside_the_unit_interval_is_rejected(self, beta: float) -> None:
        """Beta interpolates two divergences and cannot extrapolate past them."""
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            BoltzmannMatchingLoss(beta=beta)

    def test_nonpositive_temperature_is_rejected(self) -> None:
        """A temperature that defines no ensemble fails at construction."""
        with pytest.raises(ValueError, match="must be positive Kelvin"):
            BoltzmannMatchingLoss(temperature=0.0)

    def test_mixed_system_sizes_are_rejected(self) -> None:
        """Energies of different systems are not one Boltzmann distribution."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        with pytest.raises(ValueError, match="graphs of different sizes"):
            loss_fn(
                torch.zeros(2, 1),
                torch.zeros(2, 1),
                num_nodes_per_graph=torch.tensor([3, 4]),
            )

    def test_uniform_system_sizes_are_accepted(self) -> None:
        """One system's replicas are exactly the ensemble the term is defined on."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        loss = loss_fn(
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            num_nodes_per_graph=torch.tensor([4, 4]),
        )
        assert loss.item() == pytest.approx(0.0)

    def test_loss_declares_it_needs_no_evaluation_gradients(self) -> None:
        """Total energies are a direct output, so validation can run no-grad."""
        assert BoltzmannMatchingLoss().requires_eval_grad is False

    def test_default_keys_read_the_teacher_and_student_energies(self) -> None:
        """The term needs no target of its own beyond the teacher's energies."""
        loss_fn = BoltzmannMatchingLoss()
        assert loss_fn.target_key == "teacher_energy"
        assert loss_fn.prediction_key == "predicted_energy"

    def test_reduced_energy_scale_is_the_thermal_energy(self) -> None:
        """Energies are reduced by ``k_B T`` in the toolkit's eV convention."""
        loss_fn = BoltzmannMatchingLoss(temperature=_TEMPERATURE)
        assert loss_fn.reduced_energy_scale == pytest.approx(_KT)

    def test_spec_round_trip_rebuilds_an_equivalent_loss(self) -> None:
        """A JSON round-tripped spec rebuilds the loss with its configuration."""
        spec = loss_component_to_spec(
            BoltzmannMatchingLoss(beta=0.25, temperature=500.0)
        )
        rebuilt = create_model_spec_from_json(
            json.loads(spec.model_dump_json())
        ).build()
        assert isinstance(rebuilt, BoltzmannMatchingLoss)
        assert rebuilt.beta == pytest.approx(0.25)
        assert rebuilt.temperature == pytest.approx(500.0)
