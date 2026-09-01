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
"""Tests for teacher-by-reference checkpoints of :mod:`nvalchemi.training.distillation`."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
)
from nvalchemi.training._checkpoint import _model_fingerprint
from nvalchemi.training._spec import BaseSpec, create_model_spec
from nvalchemi.training.distillation import DistillationStrategy
from test.training.conftest import _build_batch, _build_demo_model
from test.training.distillation.conftest import (
    _build_direct_force_model,
    _build_direct_force_teacher,
    _DirectForceTeacher,
)


class _SourcedTeacher(_DirectForceTeacher):
    """Direct-force teacher that names the file its weights were loaded from."""

    def __init__(self, model: Any, source: str) -> None:
        """Record the source alongside the wrapped model."""
        super().__init__(model)
        self.source = source

    def checkpoint_spec(self) -> BaseSpec:
        """Return the factory spec that reloads this teacher from its source."""
        return create_model_spec(_teacher_from_file, path=self.source)


def _teacher_from_file(path: str) -> _SourcedTeacher:
    """Rebuild the direct-force teacher stored at *path*."""
    teacher = _SourcedTeacher(_build_direct_force_model(seed=2), path)
    teacher.load_state_dict(torch.load(path, weights_only=True))
    return teacher


def _write_teacher(tmp_path: Path, *, seed: int = 2) -> _SourcedTeacher:
    """Return a teacher whose weights sit in a file it can be reloaded from."""
    source = tmp_path / "teacher.pt"
    teacher = _SourcedTeacher(_build_direct_force_model(seed=seed), str(source))
    torch.save(teacher.state_dict(), source)
    return teacher


def _make_strategy(teacher: Any, *, num_steps: int = 2) -> DistillationStrategy:
    """Return an offline distillation strategy over demo models."""
    return DistillationStrategy(
        models={"student": _build_demo_model(), "teacher": teacher},
        optimizer_configs={
            "student": [
                OptimizerConfig(
                    optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2}
                )
            ]
        },
        loss_fn=EnergyMSELoss(target_key="teacher_energy")
        + ForceMSELoss(target_key="teacher_forces"),
        num_steps=num_steps,
    )


def _batches(count: int = 4) -> list[Batch]:
    """Return a short stream of unlabeled training batches."""
    return [_build_batch(seed=index) for index in range(count)]


def _manifest(root: Path) -> dict[str, Any]:
    """Return the parsed checkpoint manifest under *root*."""
    return json.loads((root / "manifest.json").read_text())


class TestTeacherByReferenceCheckpoints:
    def test_a_sourced_teacher_is_recorded_instead_of_serialized(
        self, tmp_path: Path
    ) -> None:
        """The teacher contributes a manifest reference and no weight file."""
        strategy = _make_strategy(_write_teacher(tmp_path))
        root = tmp_path / "checkpoints"

        strategy.save_checkpoint(root)

        manifest = _manifest(root)
        assert set(manifest["model_references"]) == {"teacher"}
        assert (root / "models" / "student" / "checkpoints" / "0.pt").is_file()
        assert not (root / "models" / "teacher" / "checkpoints" / "0.pt").exists()
        assert (root / "models" / "teacher" / "spec.json").is_file()

    def test_the_reference_names_the_source_and_fingerprints_the_weights(
        self, tmp_path: Path
    ) -> None:
        """A reader can see what the teacher rebuilds from without loading it."""
        teacher = _write_teacher(tmp_path)
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"

        strategy.save_checkpoint(root)

        reference = _manifest(root)["model_references"]["teacher"]
        assert reference["rebuild"] == "spec"
        assert "_teacher_from_file" in reference["source"]
        assert str(tmp_path / "teacher.pt") in reference["source"]
        assert reference["fingerprint"] == _model_fingerprint(teacher)

    def test_restoring_reloads_the_teacher_from_its_source(
        self, tmp_path: Path
    ) -> None:
        """A rebuilt strategy holds the teacher's own weights, not fresh ones."""
        teacher = _write_teacher(tmp_path)
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)

        restored = DistillationStrategy.load_checkpoint(
            root, training_fn="nvalchemi.training.distillation.default_distillation_fn"
        )

        restored_teacher = restored.models["teacher"]
        assert restored_teacher is not teacher
        for key, tensor in teacher.state_dict().items():
            torch.testing.assert_close(restored_teacher.state_dict()[key], tensor)

    def test_a_swapped_teacher_source_is_refused(self, tmp_path: Path) -> None:
        """A source that no longer holds the checkpointed weights fails loudly."""
        teacher = _write_teacher(tmp_path)
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)
        torch.save(
            _SourcedTeacher(
                _build_direct_force_model(seed=11), teacher.source
            ).state_dict(),
            teacher.source,
        )

        with pytest.raises(ValueError, match="stored by reference"):
            DistillationStrategy.load_checkpoint(
                root,
                training_fn="nvalchemi.training.distillation.default_distillation_fn",
            )

    def test_a_teacher_without_a_source_is_serialized_inline_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        """The fallback keeps working, and says what it costs."""
        strategy = _make_strategy(_build_direct_force_teacher(seed=2))
        root = tmp_path / "checkpoints"

        with pytest.warns(UserWarning, match="names no source to reload from"):
            strategy.save_checkpoint(root)

        assert _manifest(root)["model_references"] == {}
        assert (root / "models" / "teacher" / "checkpoints" / "0.pt").is_file()

    def test_the_inline_warning_is_raised_once_per_strategy(
        self, tmp_path: Path
    ) -> None:
        """A periodic checkpoint does not repeat the size warning every write."""
        strategy = _make_strategy(_build_direct_force_teacher(seed=2))
        root = tmp_path / "checkpoints"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            strategy.save_checkpoint(root)
            strategy.save_checkpoint(root)

        matching = [w for w in caught if "names no source" in str(w.message)]
        assert len(matching) == 1

    def test_a_restarted_run_trains_on_identically(self, tmp_path: Path) -> None:
        """Interrupting and restoring a run reaches the weights it would have."""
        torch.manual_seed(0)
        uninterrupted = _make_strategy(_write_teacher(tmp_path), num_steps=4)
        uninterrupted.run(_batches())

        torch.manual_seed(0)
        interrupted = _make_strategy(_write_teacher(tmp_path), num_steps=2)
        interrupted.run(_batches(2))
        root = tmp_path / "checkpoints"
        interrupted.save_checkpoint(root)
        resumed = DistillationStrategy.load_checkpoint(
            root, training_fn="nvalchemi.training.distillation.default_distillation_fn"
        )
        resumed.num_steps = 4
        resumed.run(_batches()[2:])

        assert resumed.step_count == uninterrupted.step_count == 4
        expected = uninterrupted.models["student"].state_dict()
        for key, tensor in resumed.models["student"].state_dict().items():
            torch.testing.assert_close(tensor, expected[key])
