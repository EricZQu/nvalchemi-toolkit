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
"""Tests for the on-policy segment loop of :class:`DistillationStrategy`."""

from __future__ import annotations

import json
import time
import warnings
from typing import Any
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import TrainContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
    ValidationConfig,
)
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InProcessTeacherScorer,
    OnPolicyConfig,
    TeacherLabelHook,
)
from nvalchemi.training.distillation._labels import _attach_teacher_labels
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import _build_direct_force_teacher

_SEED_ELEMENT = 1
"""Atomic number tagging every structure the propagator generates from."""

_REFERENCE_ELEMENT = 6
"""Atomic number tagging every structure that comes from the reference dataset."""

_ATOMS_PER_SYSTEM = 4
"""Atoms in every synthetic system, so batches stay small and comparable."""

_LANGEVIN_KWARGS: dict[str, Any] = {
    "dt": 0.5,
    "temperature": 300.0,
    "friction": 0.01,
    "random_seed": 7,
}
"""Thermostat settings shared by every propagator built here."""


def _make_system(atomic_number: int, seed: int) -> AtomicData:
    """Return one system tagged by *atomic_number*, carrying the propagator's keys."""
    generator = torch.Generator().manual_seed(seed)
    return AtomicData(
        positions=torch.randn(_ATOMS_PER_SYSTEM, 3, generator=generator),
        atomic_numbers=torch.full(
            (_ATOMS_PER_SYSTEM,), atomic_number, dtype=torch.long
        ),
        atomic_masses=torch.ones(_ATOMS_PER_SYSTEM),
        energy=torch.zeros(1, 1),
        forces=torch.zeros(_ATOMS_PER_SYSTEM, 3),
    )


def _make_batch(atomic_number: int, n_systems: int, base_seed: int) -> Batch:
    """Return a batch of *n_systems* systems all tagged by *atomic_number*."""
    return Batch.from_data_list(
        [_make_system(atomic_number, base_seed + index) for index in range(n_systems)]
    )


def _make_seed_dataset(n_systems: int = 4, base_seed: int = 500) -> InMemoryDataset:
    """Return the structures the generated trajectories start from."""
    return InMemoryDataset(
        in_memory_batch=_make_batch(_SEED_ELEMENT, n_systems, base_seed)
    )


def _make_reference_dataset(
    scorer: InProcessTeacherScorer, n_systems: int = 8, base_seed: int = 700
) -> InMemoryDataset:
    """Return a teacher-labeled anchor dataset with the generated frames' schema."""
    frames = _make_batch(_REFERENCE_ELEMENT, n_systems, base_seed)
    _attach_teacher_labels(frames, scorer.label(frames))
    return InMemoryDataset(in_memory_batch=frames)


def _make_scorer(teacher: BaseModelMixin) -> InProcessTeacherScorer:
    """Return an energy-and-forces scorer over *teacher*."""
    return InProcessTeacherScorer(teacher, ("energy", "forces"))


def _make_loss() -> Any:
    """Return the energy-plus-forces teacher objective the loop trains on."""
    return EnergyMSELoss(target_key="teacher_energy") + ForceMSELoss(
        target_key="teacher_forces", normalize_by_atom_count=True
    )


def _make_optimizer_configs() -> dict[str, list[OptimizerConfig]]:
    """Return the Adam configuration the student is optimized with."""
    return {
        "student": [
            OptimizerConfig(
                optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2}
            )
        ]
    }


def _make_on_policy_strategy(
    *,
    student: BaseModelMixin | None = None,
    teacher: BaseModelMixin | None = None,
    num_steps: int = 12,
    steps_per_segment: int = 4,
    segment_steps: int = 3,
    label_frequency: int = 1,
    replay_ratio: float = 0.5,
    batch_size: int = 4,
    device: str = "cpu",
    config_overrides: dict[str, Any] | None = None,
    **overrides: Any,
) -> DistillationStrategy:
    """Return a runnable on-policy strategy over independently seeded demo models."""
    student = _build_demo_model() if student is None else student
    teacher = _build_direct_force_teacher(seed=2) if teacher is None else teacher
    scorer = _make_scorer(teacher)
    config_kwargs: dict[str, Any] = {
        "dynamics": NVTLangevin(student, **_LANGEVIN_KWARGS),
        "teacher_scorer": scorer,
        "seed_dataset": _make_seed_dataset(),
        "replay_ratio": replay_ratio,
        "steps_per_segment": steps_per_segment,
        "batch_size": batch_size,
        "segment_steps": segment_steps,
        "label_frequency": label_frequency,
    }
    config_kwargs.update(config_overrides or {})
    kwargs: dict[str, Any] = {
        "models": {"student": student, "teacher": teacher},
        "optimizer_configs": _make_optimizer_configs(),
        "loss_fn": _make_loss(),
        "num_steps": num_steps,
        "devices": [torch.device(device)],
        "reference_dataset": None
        if replay_ratio == 1.0
        else _make_reference_dataset(scorer),
        "on_policy": OnPolicyConfig(**config_kwargs),
    }
    kwargs.update(overrides)
    return DistillationStrategy(**kwargs)


def _graph_tags(batch: Batch) -> list[int]:
    """Return the atomic number tagging each graph of *batch*."""
    return [
        int(batch.atomic_numbers[batch.batch_idx == index][0])
        for index in range(batch.num_graphs)
    ]


class _RecordingBatchHook:
    """Record the loss and the per-graph source tags of every training batch."""

    frequency = 1
    stage = TrainingStage.AFTER_BATCH

    def __init__(self) -> None:
        """Start with an empty trace."""
        self.losses: list[float] = []
        self.tags: list[list[int]] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Append the loss and composition of the batch just trained on."""
        self.losses.append(float(ctx.loss))
        self.tags.append(_graph_tags(ctx.batch))


class TestOnPolicySegmentLoop:
    def test_three_segments_train_on_labeled_generated_frames(
        self, device: str
    ) -> None:
        """A full run reaches its step target on generated, teacher-labeled data."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(device=device, hooks=[recorder])

        strategy.run()

        assert strategy.step_count == 12
        assert strategy.epoch_count == 3
        assert len(recorder.losses) == 12
        assert strategy.on_policy.dynamics.step_count == 9

    def test_generated_frames_land_in_the_buffer_with_the_label_schema(self) -> None:
        """Every propagated step is labeled and stored under one frozen schema."""
        strategy = _make_on_policy_strategy()

        strategy.run()

        buffer = strategy.replay_buffer
        assert buffer is not None
        assert "node.teacher_forces" in buffer.schema
        assert "system.teacher_energy" in buffer.schema
        assert len(buffer) == 3 * 3 * len(strategy.on_policy.seed_dataset)

    def test_every_batch_holds_the_configured_mixture(self) -> None:
        """A replay ratio of one half puts two generated frames in a batch of four."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(hooks=[recorder])

        strategy.run()

        for tags in recorder.tags:
            assert len(tags) == 4
            assert tags.count(_SEED_ELEMENT) == 2
            assert tags.count(_REFERENCE_ELEMENT) == 2

    def test_loss_falls_across_the_segments(self) -> None:
        """Twelve steps against the teacher's energies and forces lower the loss."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(hooks=[recorder])

        strategy.run()

        assert sum(recorder.losses[-4:]) < sum(recorder.losses[:4])

    def test_the_student_generates_from_its_updated_weights(self) -> None:
        """The propagator holds the trained module, so generation follows training."""
        strategy = _make_on_policy_strategy()
        before = strategy.models["student"].model.projection.weight.detach().clone()

        strategy.run()

        assert strategy.on_policy.dynamics.model is strategy.models["student"]
        assert not torch.allclose(
            strategy.models["student"].model.projection.weight, before
        )

    def test_replay_only_runs_need_no_reference_dataset(self) -> None:
        """A ratio of one draws every sample from the buffer the run itself filled."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(
            replay_ratio=1.0, num_steps=6, hooks=[recorder]
        )

        strategy.run()

        assert strategy.step_count == 6
        assert all(set(tags) == {_SEED_ELEMENT} for tags in recorder.tags)

    def test_labeling_hook_is_removed_from_the_caller_propagator(self) -> None:
        """The loop leaves the propagator as it found it, so a rerun labels once."""
        strategy = _make_on_policy_strategy(num_steps=4)

        strategy.run()

        assert strategy.on_policy.dynamics.hooks == []

    def test_teacher_weights_survive_the_run(self) -> None:
        """Generation and training both leave the frozen teacher untouched."""
        strategy = _make_on_policy_strategy(num_steps=4)
        before = [
            parameter.detach().clone()
            for parameter in strategy.models["teacher"].parameters()
        ]

        strategy.run()

        for parameter, snapshot in zip(
            strategy.models["teacher"].parameters(), before, strict=True
        ):
            torch.testing.assert_close(parameter, snapshot)


class TestOnPolicySegmentAccounting:
    def test_final_partial_segment_trains_only_the_remainder(self) -> None:
        """A step target that is not a multiple of the segment never overshoots."""
        recorder = _RecordingBatchHook()
        strategy = _make_on_policy_strategy(
            num_steps=7, steps_per_segment=3, hooks=[recorder]
        )

        strategy.run()

        assert strategy.step_count == 7
        assert len(recorder.losses) == 7
        assert strategy.epoch_count == 3

    def test_reaching_the_target_returns_without_generating(self) -> None:
        """A run already at its target neither propagates nor trains."""
        strategy = _make_on_policy_strategy(num_steps=4)
        strategy.step_count = 4

        strategy.run()

        assert strategy.on_policy.dynamics.step_count == 0
        assert strategy.replay_buffer is None

    def test_the_last_frame_of_a_segment_is_labeled_off_cadence(self) -> None:
        """A cadence that skips the segment's end still stores its final frame."""
        strategy = _make_on_policy_strategy(
            num_steps=3, steps_per_segment=1, segment_steps=3, label_frequency=10
        )

        strategy.run()

        seeds = len(strategy.on_policy.seed_dataset)
        assert strategy.on_policy.dynamics.step_count == 9
        assert len(strategy.replay_buffer) == 4 * seeds

    def test_an_early_exiting_propagator_still_produces_a_segment(self) -> None:
        """A chunk that converges out short is read from ``step_count``, not assumed."""
        student = _build_demo_model()
        strategy = _make_on_policy_strategy(
            student=student,
            num_steps=4,
            steps_per_segment=2,
            segment_steps=5,
            label_frequency=10,
            config_overrides={
                "dynamics": NVTLangevin(
                    student,
                    convergence_hook=ConvergenceHook.from_fmax(threshold=1e6),
                    **_LANGEVIN_KWARGS,
                )
            },
        )

        strategy.run()

        assert strategy.step_count == 4
        assert strategy.on_policy.dynamics.step_count == 2
        assert len(strategy.replay_buffer) == 2 * len(strategy.on_policy.seed_dataset)

    def test_capacity_bounds_the_buffer_across_segments(self) -> None:
        """A bounded buffer keeps the newest frames instead of growing forever."""
        strategy = _make_on_policy_strategy(
            num_steps=8, config_overrides={"replay_capacity": 6}
        )

        strategy.run()

        assert len(strategy.replay_buffer) == 6


class TestChunkedPropagatorResume:
    def _run_langevin(self, chunks: tuple[int, ...]) -> tuple[Batch, NVTLangevin]:
        """Return the state and propagator after running *chunks* back to back."""
        state = _make_batch(_SEED_ELEMENT, 3, base_seed=500)
        dynamics = NVTLangevin(_build_demo_model(), **_LANGEVIN_KWARGS)
        for n_steps in chunks:
            state = dynamics.run(state, n_steps=n_steps)
        return state, dynamics

    def test_two_chunks_reproduce_one_run_exactly(self) -> None:
        """Segmenting a trajectory changes nothing: the thermostat RNG is counter-based."""
        one_shot, one_shot_dynamics = self._run_langevin((6,))
        chunked, chunked_dynamics = self._run_langevin((3, 3))

        torch.testing.assert_close(
            chunked.positions, one_shot.positions, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            chunked.velocities, one_shot.velocities, rtol=0.0, atol=0.0
        )
        assert chunked_dynamics.step_count == one_shot_dynamics.step_count == 6

    def test_uneven_chunks_reproduce_one_run_exactly(self) -> None:
        """Resume exactness does not depend on the chunks being the same length."""
        one_shot, _ = self._run_langevin((6,))
        chunked, _ = self._run_langevin((1, 4, 1))

        torch.testing.assert_close(
            chunked.positions, one_shot.positions, rtol=0.0, atol=0.0
        )

    def test_labeling_does_not_perturb_the_trajectory(self) -> None:
        """A teacher pass between steps leaves the propagated state bit-identical."""
        unlabeled, _ = self._run_langevin((3, 3))
        labeled = _make_batch(_SEED_ELEMENT, 3, base_seed=500)
        dynamics = NVTLangevin(_build_demo_model(), **_LANGEVIN_KWARGS)
        dynamics.register_hook(
            TeacherLabelHook(_make_scorer(_build_direct_force_teacher(seed=2)))
        )

        for _ in range(2):
            labeled = dynamics.run(labeled, n_steps=3)

        torch.testing.assert_close(
            labeled.positions, unlabeled.positions, rtol=0.0, atol=0.0
        )


class TestOnPolicyDirectForceTeacher:
    def test_a_direct_force_teacher_trains_a_conservative_student(self) -> None:
        """Forces from a teacher head distill into a student whose forces are a gradient."""
        recorder = _RecordingBatchHook()
        student = _build_demo_model()
        strategy = _make_on_policy_strategy(
            student=student,
            teacher=_build_direct_force_teacher(seed=3),
            hooks=[recorder],
        )

        strategy.run()

        assert strategy.models["teacher"].model_config.autograd_outputs == frozenset()
        assert "forces" in student.model_config.autograd_outputs
        assert strategy.step_count == 12
        assert sum(recorder.losses[-4:]) < sum(recorder.losses[:4])


class TestOnPolicyValidation:
    def test_step_cadence_validation_fires_inside_the_segments(self) -> None:
        """Validation runs mid-segment and reads a summary from unlabeled data."""
        strategy = _make_on_policy_strategy(
            num_steps=8,
            validation_config=ValidationConfig(
                validation_data=[_make_batch(_REFERENCE_ELEMENT, 2, base_seed=900)],
                every_n_steps=4,
            ),
        )

        strategy.run()

        assert strategy.step_count == 8
        assert strategy.last_validation is not None
        assert strategy.last_validation["total_loss"] > 0.0

    def test_prelabeled_training_batches_are_never_labeled_twice(self) -> None:
        """Generated and reference frames arrive labeled, so the seam skips them all."""
        strategy = _make_on_policy_strategy(num_steps=8)

        with patch.object(
            strategy.teacher_scorer, "label", wraps=strategy.teacher_scorer.label
        ) as spy:
            strategy.run()

        assert strategy.step_count == 8
        assert spy.call_count == 0

    def test_unlabeled_validation_data_is_labeled_by_the_training_seam(self) -> None:
        """The one batch the run cannot pre-label is the validation set."""
        strategy = _make_on_policy_strategy(
            num_steps=8,
            validation_config=ValidationConfig(
                validation_data=[_make_batch(_REFERENCE_ELEMENT, 2, base_seed=900)],
                every_n_steps=4,
            ),
        )

        with patch.object(
            strategy.teacher_scorer, "label", wraps=strategy.teacher_scorer.label
        ) as spy:
            strategy.run()

        assert spy.call_count >= 1

    def test_epoch_cadence_validation_follows_the_segments(self) -> None:
        """One segment is one epoch, so an epoch cadence fires at segment boundaries."""
        strategy = _make_on_policy_strategy(
            num_steps=8,
            validation_config=ValidationConfig(
                validation_data=[_make_batch(_REFERENCE_ELEMENT, 2, base_seed=900)],
                every_n_epochs=1,
            ),
        )

        strategy.run()

        assert strategy.epoch_count == 2
        assert strategy.last_validation is not None


class TestOnPolicyValidationContract:
    def test_a_propagator_over_another_module_is_rejected(self) -> None:
        """On-policy data requires the propagator to hold the trained student itself."""
        with pytest.raises(ValueError, match="models\\['student'\\]"):
            _make_on_policy_strategy(
                config_overrides={
                    "dynamics": NVTLangevin(_build_demo_model(), **_LANGEVIN_KWARGS)
                }
            )

    def test_epoch_sized_runs_are_rejected(self) -> None:
        """Segments build their own loaders, so there is no epoch to convert."""
        with pytest.raises(ValueError, match="sized in optimizer steps"):
            _make_on_policy_strategy(num_steps=None, num_epochs=2)

    def test_a_partial_ratio_without_a_reference_dataset_is_rejected(self) -> None:
        """Mixing in reference data requires a reference dataset to mix from."""
        with pytest.raises(ValueError, match="reference_dataset is required"):
            _make_on_policy_strategy(reference_dataset=None)

    def test_a_zero_replay_ratio_is_rejected(self) -> None:
        """Generating frames no batch ever draws is offline training with extra steps."""
        with pytest.raises(ValueError, match="drop on_policy"):
            _make_on_policy_strategy(replay_ratio=0.0)

    def test_a_reference_dataset_without_on_policy_is_rejected(self) -> None:
        """Offline distillation trains on the dataloader, not on the anchor field."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.raises(ValueError, match="read only by the segment loop"):
            DistillationStrategy(
                models={"student": _build_demo_model(), "teacher": teacher},
                optimizer_configs=_make_optimizer_configs(),
                loss_fn=_make_loss(),
                num_steps=2,
                reference_dataset=_make_reference_dataset(_make_scorer(teacher)),
            )

    def test_a_dataloader_is_rejected_in_on_policy_mode(self) -> None:
        """The segment loop owns its loader, so a caller's would be silently dropped."""
        strategy = _make_on_policy_strategy(num_steps=2)
        with pytest.raises(ValueError, match="builds its own loader"):
            strategy.run([_make_batch(_SEED_ELEMENT, 2, base_seed=800)])

    def test_offline_mode_still_requires_a_dataloader(self) -> None:
        """Without a segment loop there is nothing to train on but the caller's batches."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = DistillationStrategy(
            models={"student": _build_demo_model(), "teacher": teacher},
            optimizer_configs=_make_optimizer_configs(),
            loss_fn=_make_loss(),
            num_steps=2,
        )
        with pytest.raises(ValueError, match="run\\(dataloader=None\\)"):
            strategy.run()

    def test_a_narrower_generation_scorer_warns(self) -> None:
        """Frames missing a signal the loss reads get scored twice, so the loop says so."""
        teacher = _build_direct_force_teacher(seed=2)
        with pytest.warns(UserWarning, match="scored twice"):
            _make_on_policy_strategy(
                teacher=teacher,
                config_overrides={
                    "teacher_scorer": InProcessTeacherScorer(teacher, ("energy",))
                },
            )


class TestOnPolicySerialization:
    def test_serializing_an_on_policy_strategy_warns(self) -> None:
        """A spec cannot describe a live propagator, so the omission is announced."""
        strategy = _make_on_policy_strategy(num_steps=2)

        with pytest.warns(UserWarning, match="omitted from the spec"):
            spec = strategy.to_spec_dict()

        assert "on_policy" not in spec
        assert "reference_dataset" not in spec

    def test_the_round_trip_rebuilds_an_offline_strategy(self) -> None:
        """A rebuilt strategy runs offline until the on-policy pieces are re-supplied."""
        strategy = _make_on_policy_strategy(num_steps=2)
        with pytest.warns(UserWarning, match="omitted from the spec"):
            spec = json.loads(json.dumps(strategy.to_spec_dict()))

        teacher = _build_direct_force_teacher(seed=2)
        rebuilt = DistillationStrategy.from_spec_dict(
            spec, models={"student": _build_demo_model(), "teacher": teacher}
        )

        assert rebuilt.on_policy is None
        assert rebuilt.reference_dataset is None
        rebuilt.run([_make_batch(_REFERENCE_ELEMENT, 2, base_seed=900)])
        assert rebuilt.step_count == 2

    def test_an_offline_strategy_serializes_without_warning(self) -> None:
        """The warning is about the on-policy fields, not about distillation."""
        teacher = _build_direct_force_teacher(seed=2)
        strategy = DistillationStrategy(
            models={"student": _build_demo_model(), "teacher": teacher},
            optimizer_configs=_make_optimizer_configs(),
            loss_fn=_make_loss(),
            num_steps=2,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = strategy.to_spec_dict()

        assert spec["label_missing"] is True
        assert not [w for w in caught if "omitted from the spec" in str(w.message)]


@pytest.mark.slow
class TestOnPolicyLabelingOverhead:
    def _time_segment(self, dynamics: NVTLangevin, n_steps: int) -> float:
        """Return the wall-clock seconds a warmed-up segment of *n_steps* takes."""
        dynamics.run(_make_batch(_SEED_ELEMENT, 4, base_seed=500), n_steps=2)
        state = _make_batch(_SEED_ELEMENT, 4, base_seed=500)
        start = time.perf_counter()
        dynamics.run(state, n_steps=n_steps)
        return time.perf_counter() - start

    def test_labeling_overhead_stays_within_budget(self) -> None:
        """Labeling every step of a segment costs a bounded multiple of generating it."""
        student = _build_demo_model()
        scorer = _make_scorer(_build_direct_force_teacher(seed=2))
        labeled = NVTLangevin(student, **_LANGEVIN_KWARGS)
        labeled.register_hook(TeacherLabelHook(scorer))
        captured = NVTLangevin(student, **_LANGEVIN_KWARGS)
        captured.register_hook(
            TeacherLabelHook(scorer, sink=HostMemory(capacity=10_000))
        )

        bare_seconds = self._time_segment(NVTLangevin(student, **_LANGEVIN_KWARGS), 40)
        labeled_seconds = self._time_segment(labeled, 40)
        captured_seconds = self._time_segment(captured, 40)

        print(
            f"generation {bare_seconds:.3f}s; labeling every step "
            f"{labeled_seconds / bare_seconds:.2f}x; labeling and capturing "
            f"{captured_seconds / bare_seconds:.2f}x"
        )
        assert labeled_seconds / bare_seconds < 20.0
        assert captured_seconds / bare_seconds < 40.0
