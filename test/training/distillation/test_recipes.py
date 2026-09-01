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
"""Tests for recipe serialization and restart of the on-policy segment loop."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
)
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InProcessTeacherScorer,
    OnPolicyConfig,
    label_dataset,
)
from nvalchemi.training.distillation._restart import _OnPolicyRestartHook
from nvalchemi.training.distillation.config import (
    _dynamics_from_spec_dict,
    _dynamics_spec_dict,
)
from nvalchemi.training.hooks import CheckpointHook
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import _build_direct_force_teacher

_ATOMS_PER_SYSTEM = 4
"""Atoms in every synthetic system, so the segments stay small."""

_SEED_ELEMENT = 1
"""Atomic number tagging the structures the propagator generates from."""

_REFERENCE_ELEMENT = 6
"""Atomic number tagging the structures the anchor supplies."""

_SEGMENT_STEPS = 3
"""Propagator steps one segment generates, as every recipe here sets it."""

_LANGEVIN = {
    "cls_path": "nvalchemi.dynamics.integrators.nvt_langevin.NVTLangevin",
    "kwargs": {
        "dt": 0.5,
        "temperature": 300.0,
        "friction": 0.01,
        "random_seed": 7,
    },
}
"""Propagator reference every recipe here builds its segment loop from."""


def _make_system(
    atomic_number: int, seed: int, *, predictions: bool = False
) -> AtomicData:
    """Return one system tagged by *atomic_number*.

    ``predictions=True`` carries the ``energy`` and ``forces`` a propagator
    reads on its first step and the labeling hook strips again, which is the
    shape a seed structure has and a replay frame — and so the anchor — has not.
    """
    generator = torch.Generator().manual_seed(seed)
    predicted = (
        {"energy": torch.zeros(1, 1), "forces": torch.zeros(_ATOMS_PER_SYSTEM, 3)}
        if predictions
        else {}
    )
    return AtomicData(
        positions=torch.randn(_ATOMS_PER_SYSTEM, 3, generator=generator),
        atomic_numbers=torch.full(
            (_ATOMS_PER_SYSTEM,), atomic_number, dtype=torch.long
        ),
        atomic_masses=torch.ones(_ATOMS_PER_SYSTEM),
        **predicted,
    )


def _make_batch(
    atomic_number: int, n_systems: int, base_seed: int, *, predictions: bool = False
) -> Batch:
    """Return a batch of systems all tagged by *atomic_number*."""
    return Batch.from_data_list(
        [
            _make_system(atomic_number, base_seed + index, predictions=predictions)
            for index in range(n_systems)
        ]
    )


def _make_scorer(teacher: BaseModelMixin) -> InProcessTeacherScorer:
    """Return an energy-and-forces scorer over *teacher*."""
    return InProcessTeacherScorer(teacher, ("energy", "forces"))


def _make_store(
    store: Path,
    scorer: InProcessTeacherScorer,
    element: int,
    n_systems: int,
    seed: int,
    *,
    predictions: bool = False,
) -> Dataset:
    """Return a teacher-labeled Zarr store a recipe can name by path."""
    label_dataset(
        InMemoryDataset(
            in_memory_batch=_make_batch(
                element, n_systems, seed, predictions=predictions
            )
        ),
        scorer,
        store,
        batch_size=4,
    )
    return Dataset(reader=AtomicDataZarrReader(store))


def _make_recipe(seed_store: Path, **overrides: Any) -> dict[str, Any]:
    """Return an on-policy recipe seeded from *seed_store*."""
    recipe: dict[str, Any] = {
        "dynamics": json.loads(json.dumps(_LANGEVIN)),
        "teacher_scorer": {
            "teacher": "teacher",
            "signals": ["energy", "forces"],
            "cast_to": None,
        },
        "seed_dataset": {"path": str(seed_store), "device": "cpu"},
        "replay_ratio": 0.5,
        "steps_per_segment": 2,
        "batch_size": 4,
        "segment_steps": _SEGMENT_STEPS,
        "label_frequency": 1,
        "replay_capacity": None,
        "replay_eviction": "fifo",
        "replay_device": None,
        "seed": 0,
        "weight_sync_frequency": 1,
    }
    recipe.update(overrides)
    return recipe


def _make_strategy(
    tmp_path: Path,
    *,
    student: BaseModelMixin,
    teacher: BaseModelMixin,
    num_steps: int,
    hooks: list[Any] | None = None,
) -> DistillationStrategy:
    """Return an on-policy strategy whose segment loop came from a recipe."""
    scorer = _make_scorer(teacher)
    seed_store = tmp_path / "seeds.zarr"
    anchor_store = tmp_path / "anchor.zarr"
    if not seed_store.exists():
        _make_store(seed_store, scorer, _SEED_ELEMENT, 4, 500, predictions=True)
        _make_store(anchor_store, scorer, _REFERENCE_ELEMENT, 8, 700)
    return DistillationStrategy(
        models={"student": student, "teacher": teacher},
        optimizer_configs={
            "student": [
                OptimizerConfig(
                    optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2}
                )
            ]
        },
        loss_fn=EnergyMSELoss(target_key="teacher_energy")
        + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
        num_steps=num_steps,
        hooks=list(hooks or []),
        reference_dataset=Dataset(reader=AtomicDataZarrReader(anchor_store)),
        on_policy=OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=student, teacher=teacher
        ),
    )


class _ScalarPropagator(BaseDynamics):
    """Propagator keeping every constructor argument on a same-named attribute."""

    def __init__(
        self,
        model: BaseModelMixin,
        dt: float = 0.25,
        precision: torch.dtype = torch.float32,
        staging: torch.device = torch.device("cpu"),
    ) -> None:
        """Record the knobs a recipe carries."""
        super().__init__(model=model)
        self.dt = dt
        self.precision = precision
        self.staging = staging

    def pre_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch

    def post_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch


class _PrivateKnobPropagator(BaseDynamics):
    """Propagator storing a defaulted constructor argument under a private name."""

    def __init__(
        self,
        model: BaseModelMixin,
        dt: float = 0.25,
        temperature: float = 300.0,
    ) -> None:
        """Keep the timestep exposed and hide the temperature."""
        super().__init__(model=model)
        self.dt = dt
        self._temperature = temperature

    def pre_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch

    def post_update(self, batch: Batch) -> Batch:
        """Return *batch* untouched."""
        return batch


class TestOnPolicyRecipeRoundTrip:
    def test_a_recipe_built_config_serializes_back_to_its_recipe(
        self, tmp_path: Path
    ) -> None:
        """Rebuilding from a recipe and re-serializing returns the same recipe."""
        teacher = _build_direct_force_teacher(seed=2)
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        recipe = _make_recipe(seed_store)

        config = OnPolicyConfig.from_spec_dict(
            recipe, student=_build_demo_model(), teacher=teacher
        )

        assert config.to_spec_dict(teacher=teacher) == recipe
        assert isinstance(config.dynamics, NVTLangevin)

    def test_the_rebuilt_propagator_holds_the_supplied_student(
        self, tmp_path: Path
    ) -> None:
        """On-policy data is only on-policy because the propagator holds the student."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )

        config = OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=student, teacher=teacher
        )

        assert config.dynamics.model is student
        assert config.teacher_scorer.teacher is teacher

    def test_a_strategy_spec_carries_the_recipe_and_the_anchor(
        self, tmp_path: Path
    ) -> None:
        """A whole on-policy run survives to_spec_dict as references."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )

        spec = json.loads(json.dumps(strategy.to_spec_dict()))

        assert spec["on_policy"]["dynamics"] == _LANGEVIN
        assert spec["reference_dataset"]["path"].endswith("anchor.zarr")
        assert spec["teacher_signals"] is None

    def test_the_round_trip_rebuilds_a_running_on_policy_strategy(
        self, tmp_path: Path
    ) -> None:
        """The rebuilt strategy generates rather than falling back to offline."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        spec = json.loads(json.dumps(strategy.to_spec_dict()))

        rebuilt = DistillationStrategy.from_spec_dict(
            spec,
            models={"student": _build_demo_model(), "teacher": teacher},
        )

        assert rebuilt.on_policy is not None
        assert rebuilt.on_policy.dynamics.model is rebuilt.models["student"]
        assert rebuilt.reference_dataset is not None
        rebuilt.run()
        assert rebuilt.step_count == 2
        assert len(rebuilt.replay_buffer) > 0

    def test_an_in_memory_seed_dataset_names_what_to_do(self, tmp_path: Path) -> None:
        """A dataset a path cannot name is refused with the fix in the message."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        config = OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=student, teacher=teacher
        )
        config.seed_dataset = InMemoryDataset(
            in_memory_batch=_make_batch(_SEED_ELEMENT, 2, 500)
        )

        with pytest.raises(ValueError, match="holding its samples in memory"):
            config.to_spec_dict(teacher=teacher)

    def test_a_hand_built_propagator_is_omitted_with_its_reason(
        self, tmp_path: Path
    ) -> None:
        """A propagator that hides its constructor arguments cannot be described."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        config = OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=student, teacher=teacher
        )
        config.dynamics = NVTLangevin(
            student, dt=0.5, temperature=300.0, friction=0.01, random_seed=7
        )

        with pytest.raises(ValueError, match="cannot be read back"):
            config.to_spec_dict(teacher=teacher)

    def test_a_scorer_over_another_teacher_is_refused(self, tmp_path: Path) -> None:
        """A recipe names the teacher by role, so a second one cannot be described."""
        teacher = _build_direct_force_teacher(seed=2)
        student = _build_demo_model()
        seed_store = tmp_path / "seeds.zarr"
        _make_store(
            seed_store, _make_scorer(teacher), _SEED_ELEMENT, 4, 500, predictions=True
        )
        config = OnPolicyConfig.from_spec_dict(
            _make_recipe(seed_store), student=student, teacher=teacher
        )

        with pytest.raises(ValueError, match="not the strategy's"):
            config.to_spec_dict(teacher=_build_direct_force_teacher(seed=5))


class TestIntrospectedPropagatorRecipes:
    def test_a_dtype_argument_round_trips_through_json(self) -> None:
        """A torch.dtype kwarg travels as its name and comes back as a dtype."""
        student = _build_demo_model()
        propagator = _ScalarPropagator(student, dt=0.5, precision=torch.float64)

        spec = _dynamics_spec_dict(propagator)
        rebuilt = _dynamics_from_spec_dict(json.loads(json.dumps(spec)), student)

        assert spec["kwargs"]["precision"] == "float64"
        assert rebuilt.precision is torch.float64

    def test_a_device_argument_round_trips_through_json(self) -> None:
        """A torch.device kwarg travels as its name and comes back as a device."""
        student = _build_demo_model()
        propagator = _ScalarPropagator(student, staging=torch.device("cpu"))

        spec = _dynamics_spec_dict(propagator)
        rebuilt = _dynamics_from_spec_dict(json.loads(json.dumps(spec)), student)

        assert spec["kwargs"]["staging"] == "cpu"
        assert rebuilt.staging == torch.device("cpu")

    def test_the_student_is_not_reported_as_a_dropped_collaborator(self) -> None:
        """The one collaborator a rebuild rebinds is not warned about."""
        propagator = _ScalarPropagator(_build_demo_model(), dt=0.5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _dynamics_spec_dict(propagator)

        assert not [w for w in caught if "runtime objects" in str(w.message)]

    def test_a_privately_stored_defaulted_argument_is_refused(self) -> None:
        """A knob no attribute exposes is named, not rebuilt at the library default."""
        propagator = _PrivateKnobPropagator(
            _build_demo_model(), dt=0.5, temperature=1500.0
        )

        with pytest.raises(ValueError, match="does not expose its"):
            _dynamics_spec_dict(propagator)


class TestOnPolicyRestart:
    def test_a_restored_run_continues_the_same_trajectory(self, tmp_path: Path) -> None:
        """Resuming from a checkpoint reaches the trajectory an unbroken run does."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        uninterrupted = _make_strategy(
            tmp_path / "whole",
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
        )
        uninterrupted.run()

        torch.manual_seed(0)
        interrupted = _make_strategy(
            tmp_path / "split",
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "split" / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        resumed = _make_strategy(
            tmp_path / "split",
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
        )
        resumed.restore_checkpoint(tmp_path / "split" / "ckpt")
        resumed.run()

        assert resumed.step_count == uninterrupted.step_count == 4
        assert (
            resumed.on_policy.dynamics.step_count
            == uninterrupted.on_policy.dynamics.step_count
        )
        torch.testing.assert_close(
            resumed._on_policy_state.positions,
            uninterrupted._on_policy_state.positions,
        )

    def test_a_checkpoint_inside_a_segment_resumes_the_trajectory_it_held(
        self, tmp_path: Path
    ) -> None:
        """A mid-training-phase checkpoint carries the propagator state anyway."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=4,
            hooks=[CheckpointHook(tmp_path / "ckpt", step_interval=1)],
        )
        interrupted.run()

        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        resumed.restore_checkpoint(tmp_path / "ckpt", checkpoint_index=0)
        bundle = next(
            hook for hook in resumed.hooks if isinstance(hook, _OnPolicyRestartHook)
        )._restored
        resumed.run()

        assert int(bundle["dynamics_step_count"]) == _SEGMENT_STEPS
        assert resumed.step_count == 4
        assert resumed.on_policy.dynamics.step_count == 3 * _SEGMENT_STEPS
        assert interrupted.on_policy.dynamics.step_count == 2 * _SEGMENT_STEPS

    def test_a_restored_run_keeps_the_frames_it_generated(self, tmp_path: Path) -> None:
        """The replay buffer travels with the checkpoint instead of restarting empty."""
        torch.manual_seed(0)
        teacher = _build_direct_force_teacher(seed=2)
        interrupted = _make_strategy(
            tmp_path,
            student=_build_demo_model(),
            teacher=teacher,
            num_steps=2,
            hooks=[CheckpointHook(tmp_path / "ckpt", epoch_interval=1)],
        )
        interrupted.run()
        stored = len(interrupted.replay_buffer)

        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=4
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        resumed.run()

        assert stored > 0
        assert len(resumed.replay_buffer) > stored

    def test_a_run_that_never_generated_restarts_by_seeding(
        self, tmp_path: Path
    ) -> None:
        """A checkpoint taken outside a segment loop carries no trajectory."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )

        strategy.save_checkpoint(tmp_path / "ckpt")
        resumed = _make_strategy(
            tmp_path, student=_build_demo_model(), teacher=teacher, num_steps=2
        )
        resumed.restore_checkpoint(tmp_path / "ckpt")
        resumed.run()

        assert resumed.step_count == 2
        assert resumed.on_policy.dynamics.step_count == _SEGMENT_STEPS
