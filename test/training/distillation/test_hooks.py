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
"""Tests for :mod:`nvalchemi.training.distillation.hooks`."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.dynamics.optimizers.fire import FIRE
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training.distillation import InProcessTeacherScorer, TeacherLabelHook
from nvalchemi.training.distillation.scoring import TeacherLabels
from test.dynamics.conftest import make_dynamics_context
from test.training.conftest import _build_batch, _build_demo_model

_TEACHER_CUTOFF = 5.0
"""Cutoff of the neighbor list the ephemeral-key tests put on the live batch."""


def _make_batch(device: str = "cpu", n_systems: int = 2) -> Batch:
    """Return a batch carrying the fields a propagator needs, on *device*."""
    return _build_batch(n_systems=n_systems).to(device)


def _make_model(device: str = "cpu"):
    """Return a freshly-seeded demo model placed on *device*."""
    model = _build_demo_model()
    return model.to(device) if device != "cpu" else model


def _make_dynamics(device: str = "cpu", n_steps: int = 1) -> DemoDynamics:
    """Return demo dynamics over a freshly-seeded demo model on *device*."""
    return DemoDynamics(
        _make_model(device), n_steps=n_steps, dt=0.5, device_type=device
    )


def _make_scorer(
    device: str = "cpu", signals: tuple[str, ...] = ("energy", "forces")
) -> InProcessTeacherScorer:
    """Return an in-process scorer over a freshly-seeded demo teacher."""
    return InProcessTeacherScorer(_make_model(device), signals)


class _RecordingScorer:
    """Scorer counting its calls and returning constant labels."""

    def __init__(self, signals: frozenset[str] | None = None) -> None:
        self.signals = (
            signals if signals is not None else frozenset({"energy", "forces"})
        )
        self.calls = 0

    def label(self, batch: Batch) -> TeacherLabels:
        """Return zero-valued labels for every requested signal."""
        self.calls += 1
        return {
            "teacher_energy": (
                torch.zeros(batch.num_graphs, 1, device=batch.device),
                "system",
            ),
            "teacher_forces": (
                torch.zeros(batch.num_nodes, 3, device=batch.device),
                "node",
            ),
        }


class _ForeignFieldScorer:
    """Scorer that tries to write the propagator's own force field."""

    signals = frozenset({"unregistered"})

    def label(self, batch: Batch) -> TeacherLabels:
        """Return a label aimed at ``forces`` instead of ``teacher_forces``."""
        return {"forces": (torch.zeros(batch.num_nodes, 3), "node")}


class TestTeacherLabelHookLabeling:
    def test_labels_land_at_the_level_each_signal_declares(self, device: str) -> None:
        """Energy lands in the system group and forces in the atoms group."""
        batch = _make_batch(device)
        dynamics = _make_dynamics(device)
        hook = TeacherLabelHook(_make_scorer(device))

        hook(make_dynamics_context(batch, dynamics), DynamicsStage.AFTER_STEP)

        assert batch.keys["system"] >= {"teacher_energy"}
        assert batch.keys["node"] >= {"teacher_forces"}
        assert batch.teacher_energy.shape == (batch.num_graphs, 1)
        assert batch.teacher_forces.shape == (batch.num_nodes, 3)

    def test_student_energy_and_forces_are_untouched(self, device: str) -> None:
        """The values driving the propagator survive a teacher pass unchanged."""
        batch = _make_batch(device)
        dynamics = _make_dynamics(device)
        student_energy = batch.energy.clone()
        student_forces = batch.forces.clone()

        TeacherLabelHook(_make_scorer(device))(
            make_dynamics_context(batch, dynamics), DynamicsStage.AFTER_STEP
        )

        torch.testing.assert_close(batch.energy, student_energy)
        torch.testing.assert_close(batch.forces, student_forces)
        assert not torch.allclose(batch.teacher_forces, student_forces)

    def test_positions_requires_grad_is_restored(self, device: str) -> None:
        """The autograd teacher leaves ``positions`` as the propagator left it."""
        batch = _make_batch(device)
        dynamics = _make_dynamics(device)
        assert not batch.positions.requires_grad

        TeacherLabelHook(_make_scorer(device))(
            make_dynamics_context(batch, dynamics), DynamicsStage.AFTER_STEP
        )

        assert not batch.positions.requires_grad
        assert not batch.teacher_forces.requires_grad

    def test_already_labeled_frame_skips_the_teacher(self, device: str) -> None:
        """A second call on a labeled frame costs no teacher pass."""
        batch = _make_batch(device)
        dynamics = _make_dynamics(device)
        scorer = _RecordingScorer()
        hook = TeacherLabelHook(scorer)
        ctx = make_dynamics_context(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)
        hook(ctx, DynamicsStage.AFTER_STEP)

        assert scorer.calls == 1

    def test_unknown_signal_name_always_relabels(self, device: str) -> None:
        """A scorer with no field mapping cannot be short-circuited."""
        batch = _make_batch(device)
        dynamics = _make_dynamics(device)
        scorer = _RecordingScorer(signals=frozenset({"energy", "custom"}))
        hook = TeacherLabelHook(scorer)
        ctx = make_dynamics_context(batch, dynamics)

        hook(ctx, DynamicsStage.AFTER_STEP)
        hook(ctx, DynamicsStage.AFTER_STEP)

        assert scorer.calls == 2

    def test_label_outside_the_teacher_namespace_raises(self, device: str) -> None:
        """A label aimed at a propagator field is rejected, not written."""
        batch = _make_batch(device)
        dynamics = _make_dynamics(device)
        student_forces = batch.forces.clone()
        hook = TeacherLabelHook(_ForeignFieldScorer())

        with pytest.raises(ValueError, match="teacher_\\*"):
            hook(make_dynamics_context(batch, dynamics), DynamicsStage.AFTER_STEP)

        torch.testing.assert_close(batch.forces, student_forces)


class TestTeacherLabelHookSink:
    def test_captured_frame_drops_neighbor_and_bookkeeping_keys(
        self, device: str
    ) -> None:
        """The sink copy carries labels but nothing run-local."""
        batch = _make_batch(device)
        compute_neighbors(
            batch,
            config=NeighborConfig(
                cutoff=_TEACHER_CUTOFF, format=NeighborListFormat.MATRIX
            ),
        )
        batch.add_key(
            "status",
            [torch.zeros(1, 1, dtype=torch.long) for _ in range(batch.num_graphs)],
            level="system",
        )
        dynamics = _make_dynamics(device)
        sink = HostMemory(capacity=100)

        TeacherLabelHook(_make_scorer(device), sink=sink)(
            make_dynamics_context(batch, dynamics), DynamicsStage.AFTER_STEP
        )

        stored = sink.read()
        assert stored.num_graphs == batch.num_graphs
        assert "teacher_forces" in stored
        assert "teacher_energy" in stored
        for key in ("neighbor_matrix", "num_neighbors", "status", "system_id"):
            assert key not in stored

    def test_live_batch_keeps_its_neighbor_state(self, device: str) -> None:
        """Stripping happens on a copy, so downstream hooks see an intact batch."""
        batch = _make_batch(device)
        compute_neighbors(batch, config=NeighborConfig(cutoff=_TEACHER_CUTOFF))
        dynamics = _make_dynamics(device)
        sink = HostMemory(capacity=100)

        TeacherLabelHook(_make_scorer(device), sink=sink)(
            make_dynamics_context(batch, dynamics), DynamicsStage.AFTER_STEP
        )

        assert "neighbor_list" in batch
        assert "neighbor_list" not in sink.read()

    def test_registered_step_counters_are_stripped(self, device: str) -> None:
        """Bookkeeping keys registered by a fused stage are stripped too."""
        BaseDynamics.register_bookkeeping_key(
            "n_steps_counter_0",
            lambda n, dev: torch.zeros(n, 1, dtype=torch.long, device=dev),
        )
        batch = _make_batch(device)
        batch.add_key(
            "n_steps_counter_0",
            [torch.zeros(1, 1, dtype=torch.long) for _ in range(batch.num_graphs)],
            level="system",
        )
        sink = HostMemory(capacity=100)

        TeacherLabelHook(_make_scorer(device), sink=sink)(
            make_dynamics_context(batch, _make_dynamics(device)),
            DynamicsStage.AFTER_STEP,
        )

        assert "n_steps_counter_0" not in sink.read()


class TestTeacherLabelHookInDynamics:
    def test_langevin_run_labels_on_the_hook_frequency(self, device: str) -> None:
        """Four Langevin steps at ``frequency=2`` cost two teacher passes."""
        batch = _make_batch(device)
        scorer = _RecordingScorer()
        dynamics = NVTLangevin(
            _make_model(device),
            dt=0.5,
            temperature=300.0,
            friction=0.01,
            n_steps=4,
            hooks=[TeacherLabelHook(scorer, frequency=2)],
            device_type=device,
        )

        dynamics.run(batch)

        assert scorer.calls == 2
        assert dynamics.step_count == 4

    def test_relaxation_propagator_is_labeled_the_same_way(self, device: str) -> None:
        """A FIRE relaxation is on-policy generation too — nothing assumes MD."""
        batch = _make_batch(device)
        sink = HostMemory(capacity=100)
        dynamics = FIRE(
            _make_model(device),
            dt=0.1,
            n_steps=2,
            hooks=[TeacherLabelHook(_make_scorer(device), sink=sink)],
            device_type=device,
        )

        dynamics.run(batch)

        assert batch.teacher_forces.shape == (batch.num_nodes, 3)
        assert len(sink) == 2 * batch.num_graphs

    def test_labeled_frames_survive_a_chunked_run(self, device: str) -> None:
        """Resuming a run relabels the fresh state rather than the stale one."""
        batch = _make_batch(device)
        scorer = _make_scorer(device)
        dynamics = _make_dynamics(device)
        dynamics.register_hook(TeacherLabelHook(scorer))

        dynamics.run(batch, n_steps=1)
        first = batch.teacher_forces.clone()
        dynamics.run(batch, n_steps=1)

        assert dynamics.step_count == 2
        assert not torch.allclose(batch.teacher_forces, first)
