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
"""Tests for :mod:`nvalchemi.training.distillation.evaluation.throughput`."""

from __future__ import annotations

import pytest

from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators import NVE
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.training.distillation.evaluation import measure_throughput
from test.training.distillation.conftest import (
    _build_lattice_batch,
    _build_lj_teacher,
)

_LATTICE_ATOMS = 27
"""Atom count of the default 3x3x3 lattice."""


def _make_nve() -> NVE:
    """Return an NVE integrator over the Lennard-Jones teacher."""
    model = _build_lj_teacher()
    return NVE(
        model=model,
        dt=1.0,
        hooks=[
            NeighborListHook(
                config=model.model_config.neighbor_config,
                skin=1.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
    )


class TestMeasureThroughput:
    """Steady-state rates of a propagator over a fixed batch."""

    def test_rates_are_consistent_with_the_measured_window(self) -> None:
        """Every reported rate is the step rate rescaled by a known constant."""
        speed = measure_throughput(
            _make_nve(),
            _build_lattice_batch(),
            warmup_steps=2,
            measured_steps=5,
            timestep_fs=2.0,
        )
        assert speed.steps_per_second * speed.elapsed_seconds == pytest.approx(5.0)
        assert speed.atoms_per_second == pytest.approx(
            speed.steps_per_second * _LATTICE_ATOMS
        )
        assert speed.ns_per_day == pytest.approx(
            speed.steps_per_second * 2.0 * 86_400.0 / 1.0e6
        )

    def test_reported_sizes_describe_the_measured_batch(self) -> None:
        """Atom and graph counts are read at the start of the timed window."""
        speed = measure_throughput(
            _make_nve(), _build_lattice_batch(), warmup_steps=1, measured_steps=2
        )
        assert speed.num_atoms == _LATTICE_ATOMS
        assert speed.num_graphs == 1
        assert speed.device == "cpu"

    def test_ns_per_day_is_omitted_without_a_timestep(self) -> None:
        """A step has no physical duration until a timestep says what it is."""
        speed = measure_throughput(
            _make_nve(), _build_lattice_batch(), warmup_steps=0, measured_steps=2
        )
        assert speed.ns_per_day is None
        assert speed.warmup_steps == 0

    def test_warmup_and_measured_steps_both_advance_the_propagator(self) -> None:
        """The batch really is propagated for the whole warmup plus window."""
        dynamics = _make_nve()
        measure_throughput(
            dynamics, _build_lattice_batch(), warmup_steps=3, measured_steps=4
        )
        assert dynamics.step_count == 7

    @pytest.mark.parametrize(
        ("warmup_steps", "measured_steps"),
        [(2, 0), (-1, 5)],
        ids=["no-measured-steps", "negative-warmup"],
    )
    def test_invalid_step_counts_are_rejected(
        self, warmup_steps: int, measured_steps: int
    ) -> None:
        """A window with no steps, or a negative warmup, raises."""
        with pytest.raises(ValueError, match="measured_steps"):
            measure_throughput(
                _make_nve(),
                _build_lattice_batch(),
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            )

    def test_non_positive_timestep_is_rejected(self) -> None:
        """A zero timestep would report an infinite simulated rate."""
        with pytest.raises(ValueError, match="timestep_fs"):
            measure_throughput(
                _make_nve(), _build_lattice_batch(), measured_steps=2, timestep_fs=0.0
            )
