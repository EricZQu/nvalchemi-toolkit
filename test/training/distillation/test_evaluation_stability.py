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
"""Tests for :mod:`nvalchemi.training.distillation.evaluation.stability`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators import NVE
from nvalchemi.hooks import DynamicsContext
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.training.distillation.evaluation import (
    StabilityMonitor,
    compare_radial_distributions,
    extensivity_error,
    radial_distribution,
    total_momentum,
)
from test.training.conftest import _build_batch
from test.training.distillation.conftest import (
    _build_lattice_batch,
    _build_lattice_data,
    _build_lj_teacher,
    _build_pair_batch,
)

_ARGON_MASS = 39.948
"""Mass carried by every atom of the shared lattice builder."""

_LATTICE_ATOMS = 27
"""Atom count of the default 3x3x3 lattice."""


def _drive(monitor: StabilityMonitor, batch: Batch, energies: Sequence[float]) -> None:
    """Fire *monitor* once per scripted total energy, one step apart."""
    for step, energy in enumerate(energies):
        batch.energy = torch.full((batch.num_graphs, 1), energy)
        monitor(DynamicsContext(batch=batch, step_count=step), DynamicsStage.AFTER_STEP)


def _make_identified_batch(
    system_ids: Sequence[int], cells: Sequence[int] = (2, 2)
) -> Batch:
    """Return one lattice graph per entry of *system_ids*, tagged and sized to match."""
    structures = []
    for system_id, count in zip(system_ids, cells, strict=True):
        data = _build_lattice_data(cells=count)
        data.add_system_property("system_id", torch.tensor([[system_id]]))
        structures.append(data)
    return Batch.from_data_list(structures)


class _ChargeSumScorer:
    """Scorer summing a per-atom energy that reads an optional charge.

    Size-extensive by construction, and it defaults a missing charge to zero
    the way :class:`~nvalchemi.models.uma.UMAWrapper` defaults a missing tag,
    so any error the extensivity check reports comes from the supercell losing
    the field rather than from the model.
    """

    signals = frozenset({"energy"})

    def label(self, batch: Batch) -> dict[str, Any]:
        """Return each graph's summed ``1 + charge`` per-atom energy."""
        charges = getattr(batch, "charges", None)
        if charges is None:
            charges = torch.zeros(batch.num_nodes)
        per_atom = 1.0 + charges.reshape(-1)
        energy = per_atom.new_zeros(batch.num_graphs).index_add_(
            0, batch.batch_idx, per_atom
        )
        return {"teacher_energy": (energy.reshape(-1, 1),)}


def _make_nve(model: object, monitor: StabilityMonitor | None = None) -> NVE:
    """Return an NVE integrator with the Lennard-Jones neighbor-list hook."""
    hooks = [
        NeighborListHook(
            config=model.model_config.neighbor_config,
            skin=1.0,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    ]
    if monitor is not None:
        hooks.append(monitor)
    return NVE(model=model, dt=1.0, hooks=hooks)


class TestStabilityMonitor:
    """Drift and momentum metrics over a recorded trajectory."""

    def test_scripted_linear_drift_matches_the_analytic_rate(self) -> None:
        """A total energy rising by a fixed amount per step reports that slope."""
        monitor = StabilityMonitor(timestep_fs=2.0)
        batch = _build_lattice_batch()
        _drive(monitor, batch, [1.0 + 0.027 * step for step in range(11)])
        metrics = monitor.metrics()
        assert metrics.num_samples == 11
        assert metrics.energy_drift_per_atom == pytest.approx(0.27 / _LATTICE_ATOMS)
        assert metrics.energy_drift_per_atom_per_step == pytest.approx(0.001)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(500.0)

    def test_kinetic_energy_is_included_by_default(self) -> None:
        """Only the kinetic-aware monitor sees a constant-potential run heating up."""
        batch = _build_lattice_batch()
        total = StabilityMonitor()
        potential = StabilityMonitor(include_kinetic=False)
        for step in range(2):
            batch.velocities = torch.full((batch.num_nodes, 3), 0.1 * step)
            batch.energy = torch.ones(1, 1)
            for monitor in (total, potential):
                monitor(
                    DynamicsContext(batch=batch, step_count=step),
                    DynamicsStage.AFTER_STEP,
                )
        assert potential.metrics().energy_drift_per_atom == 0.0
        assert total.metrics().energy_drift_per_atom == pytest.approx(
            0.5 * _ARGON_MASS * 3.0 * 0.1**2
        )

    def test_momentum_drift_matches_the_scripted_velocity_change(self) -> None:
        """Momentum drift is the total mass times the velocity it drifted by."""
        monitor = StabilityMonitor()
        batch = _build_lattice_batch()
        for step in range(3):
            batch.velocities = torch.zeros(batch.num_nodes, 3)
            batch.velocities[:, 0] = 0.25 * step
            batch.energy = torch.zeros(1, 1)
            monitor(
                DynamicsContext(batch=batch, step_count=step), DynamicsStage.AFTER_STEP
            )
        expected = _ARGON_MASS * _LATTICE_ATOMS * 0.5
        assert monitor.metrics().max_momentum_drift == pytest.approx(expected, rel=1e-5)

    def test_a_single_sample_cannot_be_scored(self) -> None:
        """One recorded frame gives no interval to measure drift over."""
        monitor = StabilityMonitor()
        _drive(monitor, _build_lattice_batch(), [1.0])
        with pytest.raises(ValueError, match="at least two recorded samples"):
            monitor.metrics()

    def test_drift_rate_is_omitted_without_a_timestep(self) -> None:
        """Steps become nanoseconds only when a timestep says how long one is."""
        monitor = StabilityMonitor()
        _drive(monitor, _build_lattice_batch(), [1.0, 2.0])
        assert monitor.metrics().energy_drift_per_atom_per_ns is None

    def test_changing_graph_count_stops_recording_and_warns(self) -> None:
        """A batch that graduated systems is not folded into the same series."""
        monitor = StabilityMonitor()
        _drive(monitor, _build_lattice_batch(), [1.0, 2.0])
        graduated = _build_lattice_batch()
        graduated = Batch.from_data_list(graduated.to_data_list() * 2)
        with pytest.warns(UserWarning, match="stopped recording"):
            monitor(
                DynamicsContext(batch=graduated, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2

    def test_a_refill_of_differently_sized_systems_stops_recording(self) -> None:
        """Same graph count, different atom counts, is still a different series."""
        monitor = StabilityMonitor()
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([0, 1], cells=(2, 3))
        with pytest.warns(UserWarning, match="stopped recording"):
            monitor(
                DynamicsContext(batch=refilled, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2

    def test_a_shape_preserving_refill_stops_recording(self) -> None:
        """Fresh systems in the same slots break the series even at the same size."""
        monitor = StabilityMonitor()
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0])
        refilled = _make_identified_batch([2, 3])
        with pytest.warns(UserWarning, match="stopped recording"):
            monitor(
                DynamicsContext(batch=refilled, step_count=9), DynamicsStage.AFTER_STEP
            )
        assert monitor.metrics().num_samples == 2

    def test_the_same_systems_keep_being_recorded(self) -> None:
        """An unchanged inflight batch is not mistaken for a refilled one."""
        monitor = StabilityMonitor()
        _drive(monitor, _make_identified_batch([0, 1]), [1.0, 2.0, 3.0])
        assert monitor.metrics().num_samples == 3

    def test_a_symmetric_excursion_fits_a_zero_drift_rate(self) -> None:
        """A run that heats up and cools back down is scored as no net drift."""
        monitor = StabilityMonitor(timestep_fs=1.0)
        _drive(monitor, _build_lattice_batch(), [0.0, 2.0, 3.0, 2.0, 0.0])
        metrics = monitor.metrics()
        assert metrics.energy_drift_per_atom == pytest.approx(0.0)
        assert metrics.energy_drift_per_atom_per_ns == pytest.approx(0.0, abs=1e-9)

    def test_lattice_at_rest_holds_its_energy_through_an_nve_run(self) -> None:
        """A Lennard-Jones lattice at its minimum drifts by nothing measurable."""
        model = _build_lj_teacher()
        monitor = StabilityMonitor(frequency=2, timestep_fs=1.0)
        _make_nve(model, monitor).run(_build_lattice_batch(), n_steps=20)
        metrics = monitor.metrics()
        assert metrics.num_samples == 10
        assert metrics.energy_drift_per_atom_per_step < 1e-9
        assert metrics.max_momentum_drift < 1e-9

    def test_perturbed_lattice_conserves_energy_under_nve(self) -> None:
        """A moving, displaced lattice still conserves energy to MD tolerance."""
        model = _build_lj_teacher()
        monitor = StabilityMonitor(frequency=5, timestep_fs=1.0)
        _make_nve(model, monitor).run(
            _build_lattice_batch(speed=0.002, jitter=0.15), n_steps=50
        )
        assert monitor.metrics().energy_drift_per_atom_per_step < 1e-6

    def test_total_momentum_sums_mass_weighted_velocities_per_graph(self) -> None:
        """A batch at rest carries no momentum, one row per graph."""
        batch = _build_lattice_batch()
        torch.testing.assert_close(total_momentum(batch), torch.zeros(1, 3))
        batch.velocities = torch.ones(batch.num_nodes, 3)
        expected = torch.full((1, 3), _ARGON_MASS * _LATTICE_ATOMS)
        torch.testing.assert_close(total_momentum(batch), expected)


class TestExtensivity:
    """Energy scaling of a model across replicated cells."""

    def test_lennard_jones_supercell_energy_is_exactly_extensive(self) -> None:
        """Doubling the cell doubles the pair energy to floating-point precision."""
        metrics = extensivity_error(
            _build_lj_teacher(), _build_lattice_batch(), repeats=(2, 1, 1)
        )
        assert metrics.num_graphs == 1
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-9)
        assert metrics.max_relative_error == pytest.approx(0.0, abs=1e-6)

    def test_replication_along_every_axis_is_supported(self) -> None:
        """A 2x2x2 supercell is eight copies and eight times the energy."""
        metrics = extensivity_error(
            _build_lj_teacher(), _build_lattice_batch(cells=2), repeats=(2, 2, 2)
        )
        assert metrics.repeats == (2, 2, 2)
        assert metrics.mean_error_per_atom == pytest.approx(0.0, abs=1e-9)

    def test_a_model_reading_a_per_atom_field_still_sees_it_in_the_supercell(
        self,
    ) -> None:
        """A field the primitive cell carries is replicated, not defaulted away."""
        data = _build_lattice_data(cells=2)
        data.add_node_property("charges", torch.full((data.num_nodes,), 0.25))
        metrics = extensivity_error(
            _ChargeSumScorer(), Batch.from_data_list([data]), repeats=(2, 1, 1)
        )
        assert metrics.max_error_per_atom == pytest.approx(0.0, abs=1e-9)

    def test_a_field_that_does_not_scale_with_the_supercell_is_rejected(self) -> None:
        """A spin multiplicity has no k-fold value, so replication raises."""
        data = _build_lattice_data(cells=2)
        data.add_system_property("spin", torch.ones(1, 1))
        with pytest.raises(ValueError, match="is not defined"):
            extensivity_error(_build_lj_teacher(), Batch.from_data_list([data]))

    def test_non_periodic_structures_are_rejected(self) -> None:
        """Replicating a cluster is not defined, so it raises instead."""
        with pytest.raises(ValueError, match="no cell"):
            extensivity_error(_build_lj_teacher(), _build_batch())

    @pytest.mark.parametrize(
        "repeats",
        [(2, 1), (0, 1, 1), (-1, 1, 1)],
        ids=["too-short", "zero", "negative"],
    )
    def test_invalid_repeat_counts_are_rejected(self, repeats: tuple[int, ...]) -> None:
        """Replication factors must be three positive integers."""
        with pytest.raises(ValueError, match="three positive integers"):
            extensivity_error(
                _build_lj_teacher(), _build_lattice_batch(), repeats=repeats
            )


class TestRadialDistribution:
    """Pair correlation accumulated over frames."""

    def test_simple_cubic_lattice_has_six_nearest_neighbors(self) -> None:
        """Every atom of the lattice has exactly six neighbors inside 5 A."""
        rdf = radial_distribution(_build_lattice_batch(), r_max=5.0, num_bins=25)
        assert float(rdf.counts.sum()) == 6.0 * _LATTICE_ATOMS
        assert rdf.num_atoms == _LATTICE_ATOMS
        assert float(rdf.edges[int(rdf.g_r.argmax())]) == pytest.approx(3.8)

    def test_isolated_pair_integrates_to_one_neighbor(self) -> None:
        """The normalization reproduces the coordination number of a lone pair."""
        rdf = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        shells = (4.0 / 3.0) * torch.pi * (rdf.edges[1:].pow(3) - rdf.edges[:-1].pow(3))
        density = 2.0 / 20.0**3
        assert float((rdf.g_r * shells).sum() * density) == pytest.approx(1.0)

    def test_frames_are_averaged_over_graphs(self) -> None:
        """Two identical frames give the same curve as one, with twice the counts."""
        single = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        doubled = radial_distribution(
            Batch.from_data_list(_build_pair_batch(3.0).to_data_list() * 2),
            r_max=6.0,
            num_bins=12,
        )
        assert doubled.num_frames == 2
        assert float(doubled.counts.sum()) == 2.0 * float(single.counts.sum())
        torch.testing.assert_close(doubled.g_r, single.g_r)

    def test_frames_keep_the_neighbor_state_they_arrived_with(self) -> None:
        """The neighbor list built to count pairs is rolled back afterwards."""
        batch = _build_lattice_batch()
        radial_distribution(batch, r_max=5.0, num_bins=25)
        assert "neighbor_list" not in batch
        assert "neighbor_matrix" not in batch

    def test_non_periodic_frames_are_rejected(self) -> None:
        """Without a cell there is no density to normalize against."""
        with pytest.raises(ValueError, match="no cell"):
            radial_distribution(_build_batch())

    @pytest.mark.parametrize(
        ("r_max", "num_bins"), [(0.0, 10), (5.0, 0)], ids=["no-range", "no-bins"]
    )
    def test_degenerate_binning_is_rejected(self, r_max: float, num_bins: int) -> None:
        """A histogram needs a positive range and at least one bin."""
        with pytest.raises(ValueError, match="must be positive"):
            radial_distribution(_build_lattice_batch(), r_max=r_max, num_bins=num_bins)


class TestRadialDistributionComparison:
    """Scalar divergences between two pair correlation functions."""

    def test_a_curve_matches_itself_exactly(self) -> None:
        """Comparing a curve to itself gives zero on every measure."""
        rdf = radial_distribution(_build_lattice_batch(), r_max=5.0, num_bins=25)
        match = compare_radial_distributions(rdf, rdf)
        assert match.jensen_shannon == 0.0
        assert match.l1 == 0.0
        assert match.max_deviation == 0.0
        assert match.num_bins == 25

    def test_disjoint_histograms_reach_the_maximum_divergence(self) -> None:
        """Peaks in different bins have no overlap, which is one bit apart."""
        near = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        far = radial_distribution(_build_pair_batch(5.0), r_max=6.0, num_bins=12)
        assert compare_radial_distributions(near, far).jensen_shannon == pytest.approx(
            1.0
        )

    def test_curves_binned_differently_cannot_be_compared(self) -> None:
        """Two curves must share bin edges before their bins mean the same thing."""
        coarse = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        fine = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=24)
        with pytest.raises(ValueError, match="share bin edges"):
            compare_radial_distributions(coarse, fine)

    def test_a_curve_with_no_pairs_cannot_be_compared(self) -> None:
        """An empty histogram has no distribution to diverge from."""
        populated = radial_distribution(_build_pair_batch(3.0), r_max=6.0, num_bins=12)
        empty = radial_distribution(
            _build_pair_batch(15.0, cell_length=60.0), r_max=6.0, num_bins=12
        )
        with pytest.raises(ValueError, match="must hold pairs"):
            compare_radial_distributions(populated, empty)
