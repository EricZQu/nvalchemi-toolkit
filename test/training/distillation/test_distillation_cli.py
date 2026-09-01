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
"""Tests for the distillation recipe CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.models.demo import DemoModelWrapper
from nvalchemi.training import save_checkpoint
from nvalchemi.training._spec import create_model_spec
from nvalchemi.training.cli import main
from nvalchemi.training.distillation import InProcessTeacherScorer, label_dataset
from nvalchemi.training.distillation.cli import DistillationJobSpec, _load_recipe
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import (
    _build_direct_force_model,
    _DirectForceTeacher,
)
from test.training.distillation.test_recipes import _make_batch

pytestmark = pytest.mark.cli

_STUDENT_PATH = "test.training.distillation.test_distillation_cli.build_cli_student"
"""Dotted path a recipe under test constructs its student from."""


def build_cli_student(hidden_dim: int = 8) -> DemoModelWrapper:
    """Return the demo student a recipe under test constructs."""
    del hidden_dim
    return _build_demo_model()


def build_cli_teacher() -> _DirectForceTeacher:
    """Return the direct-force teacher a recipe under test distills from."""
    return _DirectForceTeacher(_build_direct_force_model(seed=2))


def _combined_output(result: Any) -> str:
    """Return stdout and stderr from a Click test result."""
    return result.output + getattr(result, "stderr", "")


def _write_teacher_checkpoint(root: Path) -> Path:
    """Return a native checkpoint directory holding a rebuildable teacher."""
    teacher = build_cli_teacher()
    save_checkpoint(
        root,
        models={"teacher": (teacher, create_model_spec(build_cli_teacher))},
    )
    return root


def _write_labeled_store(
    store: Path,
    element: int,
    n_systems: int,
    seed: int,
    *,
    predictions: bool = False,
) -> Path:
    """Return a teacher-labeled Zarr store the recipe trains, seeds, or scores on."""
    label_dataset(
        InMemoryDataset(
            in_memory_batch=_make_batch(
                element, n_systems, seed, predictions=predictions
            )
        ),
        InProcessTeacherScorer(build_cli_teacher(), ("energy", "forces")),
        store,
        batch_size=4,
    )
    return store


def _write_recipe(tmp_path: Path, **overrides: Any) -> Path:
    """Write a runnable offline recipe to disk and return its path."""
    checkpoint = _write_teacher_checkpoint(tmp_path / "teacher-ckpt")
    dataset = _write_labeled_store(tmp_path / "labeled.zarr", 6, 8, 700)
    job = DistillationJobSpec.template(
        mode="offline",
        tier="small",
        dataset=str(dataset),
        output_dir=str(tmp_path / "run"),
        teacher_model="native-checkpoint",
        teacher_checkpoint=str(checkpoint),
        student_cls_path=_STUDENT_PATH,
        num_steps=2,
        device="cpu",
    )
    payload = job.model_dump(mode="json", exclude_none=True)
    payload["student"]["spec"]["kwargs"] = {"hidden_dim": 8}
    for key, value in overrides.items():
        payload[key] = value
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _write_on_policy_recipe(
    tmp_path: Path, *, anchors: int = 1, **overrides: Any
) -> Path:
    """Write a runnable on-policy recipe whose segment loop stays small."""
    checkpoint = _write_teacher_checkpoint(tmp_path / "teacher-ckpt")
    seeds = _write_labeled_store(tmp_path / "seeds.zarr", 1, 4, 500, predictions=True)
    stores = [
        str(_write_labeled_store(tmp_path / f"anchor{index}.zarr", 6, 8, 700 + index))
        for index in range(anchors)
    ]
    job = DistillationJobSpec.template(
        mode="on-policy",
        tier="small",
        dataset=stores[0],
        output_dir=str(tmp_path / "run"),
        teacher_model="native-checkpoint",
        teacher_checkpoint=str(checkpoint),
        student_cls_path=_STUDENT_PATH,
        num_steps=2,
        device="cpu",
        seed_dataset=str(seeds),
    )
    payload = job.model_dump(mode="json", exclude_none=True)
    payload["student"]["spec"]["kwargs"] = {"hidden_dim": 8}
    payload["on_policy"].update(
        {
            "replay_ratio": 0.5,
            "steps_per_segment": 2,
            "batch_size": 4,
            "segment_steps": 3,
            "label_frequency": 1,
            "replay_capacity": None,
        }
    )
    if anchors > 1:
        payload["dataset"] = {
            "paths": stores,
            "format": "alchemi-zarr-multidataset",
            "batch_size": 4,
        }
    for key, value in overrides.items():
        payload[key] = value
    path = tmp_path / "onpolicy.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


class TestRecipeScaffolds:
    def test_init_writes_a_validated_offline_recipe(self, tmp_path: Path) -> None:
        """``distill init`` writes a recipe that loads back through validation."""
        output = tmp_path / "recipe.json"

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "init",
                "--dataset",
                "data/labeled.zarr",
                "--output-dir",
                "runs/distill",
                "--teacher-model",
                "mace",
                "--teacher-id",
                "small-0b",
                "--out",
                str(output),
            ],
        )

        assert result.exit_code == 0, _combined_output(result)
        job = _load_recipe(output)
        assert job.mode == "offline"
        assert job.student.tier == "small"
        assert job.on_policy is None
        assert job.strategy["optimizer_configs"].keys() == {"student"}

    def test_the_tiers_are_sizes_rather_than_architectures(
        self, tmp_path: Path
    ) -> None:
        """Every tier writes the same knobs at a different width and depth."""
        widths = {}
        for tier in ("small", "base", "large"):
            output = tmp_path / f"{tier}.json"
            result = CliRunner().invoke(
                main,
                [
                    "distill",
                    "init",
                    "--tier",
                    tier,
                    "--dataset",
                    "data/labeled.zarr",
                    "--output-dir",
                    f"runs/{tier}",
                    "--out",
                    str(output),
                ],
            )
            assert result.exit_code == 0, _combined_output(result)
            widths[tier] = _load_recipe(output).student.spec["kwargs"]

        assert widths["small"].keys() == widths["large"].keys()
        assert widths["small"]["hidden_dim"] < widths["large"]["hidden_dim"]
        assert widths["small"]["num_layers"] < widths["large"]["num_layers"]

    def test_init_writes_the_segment_loop_for_an_on_policy_recipe(
        self, tmp_path: Path
    ) -> None:
        """An on-policy scaffold carries a propagator, a scorer, and a mixture."""
        output = tmp_path / "recipe.json"

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "init",
                "--mode",
                "on-policy",
                "--dataset",
                "data/anchor.zarr",
                "--seed-dataset",
                "data/seeds.zarr",
                "--output-dir",
                "runs/onpolicy",
                "--out",
                str(output),
            ],
        )

        assert result.exit_code == 0, _combined_output(result)
        recipe = _load_recipe(output).on_policy
        assert "cls_path" in recipe["dynamics"]
        assert recipe["teacher_scorer"]["signals"] == ["energy", "forces"]
        assert recipe["seed_dataset"]["path"] == "data/seeds.zarr"

    def test_schema_dumps_the_recipe_envelope(self) -> None:
        """``distill schema`` prints the JSON schema recipes are validated against."""
        result = CliRunner().invoke(main, ["distill", "schema"])

        assert result.exit_code == 0, _combined_output(result)
        schema = json.loads(result.output)
        assert schema["title"] == "DistillationJobSpec"
        assert {"mode", "teacher", "student", "strategy"} <= set(schema["properties"])


class TestRecipeValidation:
    def test_an_on_policy_recipe_without_a_segment_loop_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """The mode and the on_policy block have to agree."""
        path = _write_recipe(tmp_path, mode="on-policy")

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        assert "on-policy recipes need an on_policy block" in _combined_output(result)

    def test_an_optimized_teacher_is_rejected(self, tmp_path: Path) -> None:
        """The teacher is frozen by omission, and the CLI says so before the run."""
        path = _write_recipe(tmp_path)
        payload = json.loads(path.read_text())
        payload["strategy"]["optimizer_configs"]["teacher"] = payload["strategy"][
            "optimizer_configs"
        ]["student"]
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        assert "frozen by omission" in _combined_output(result)

    def test_a_strategy_construction_error_surfaces_cleanly(
        self, tmp_path: Path
    ) -> None:
        """The strategy's own contract errors reach the user as CLI errors."""
        path = _write_recipe(tmp_path)
        payload = json.loads(path.read_text())
        payload["strategy"]["loss_fn_spec"]["components"][0]["target_key"] = (
            "teacher_stress"
        )
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "run", str(path)])

        assert result.exit_code != 0
        assert "strategy could not be built" in _combined_output(result)


class TestOnPolicyPreflight:
    def test_a_knob_outside_its_range_fails_at_report(self, tmp_path: Path) -> None:
        """A replay_ratio the config forbids is refused before a model is built."""
        path = _write_on_policy_recipe(tmp_path)
        payload = json.loads(path.read_text())
        payload["on_policy"]["replay_ratio"] = 1.5
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        assert "on_policy knobs are invalid" in _combined_output(result)

    def test_a_reserved_knob_fails_at_report(self, tmp_path: Path) -> None:
        """The eviction policy the config holds back is refused at pre-flight."""
        path = _write_on_policy_recipe(tmp_path)
        payload = json.loads(path.read_text())
        payload["on_policy"]["replay_eviction"] = "uncertainty"
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        assert "reserved for committee-based" in _combined_output(result)

    def test_an_unknown_knob_fails_at_report(self, tmp_path: Path) -> None:
        """A misspelled key is an error rather than a silently ignored setting."""
        path = _write_on_policy_recipe(tmp_path)
        payload = json.loads(path.read_text())
        payload["on_policy"]["replay_ratios"] = 0.5
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        assert "on_policy knobs are invalid" in _combined_output(result)

    def test_a_recipe_without_a_seed_store_fails_at_report(
        self, tmp_path: Path
    ) -> None:
        """The CLI has no sampler, so the seed store is required rather than optional."""
        path = _write_on_policy_recipe(tmp_path)
        payload = json.loads(path.read_text())
        payload["on_policy"]["seed_dataset"] = None
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        assert "on_policy.seed_dataset" in _combined_output(result)

    def test_a_recipe_missing_an_optional_knob_still_reports(
        self, tmp_path: Path
    ) -> None:
        """A knob the config defaults is rendered at its default, not a traceback."""
        path = _write_on_policy_recipe(tmp_path)
        payload = json.loads(path.read_text())
        del payload["on_policy"]["segment_steps"]
        del payload["on_policy"]["label_frequency"]
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        output = _combined_output(result)
        assert result.exit_code == 0, output
        assert "100 generated steps" in output

    def test_an_offline_recipe_carrying_a_bundled_segment_loop_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """A pasted on-policy strategy bundle contradicts mode='offline'."""
        path = _write_recipe(tmp_path)
        payload = json.loads(path.read_text())
        payload["strategy"]["on_policy"] = {"replay_ratio": 0.5}
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        assert "while mode='offline'" in _combined_output(result)


class TestRecipeReport:
    def test_report_renders_signals_mixture_and_bars(self, tmp_path: Path) -> None:
        """The report answers what the teacher is asked for and what has to pass."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
                "thresholds": {"max_forces_mae": 0.5},
            },
        )

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        output = _combined_output(result)
        assert result.exit_code == 0, output
        assert "energy, forces" in output
        assert "offline" in output
        assert "max_forces_mae" in output

    def test_report_shows_the_on_policy_batch_composition(self, tmp_path: Path) -> None:
        """An on-policy report says how each training batch is composed."""
        output_path = tmp_path / "recipe.json"
        CliRunner().invoke(
            main,
            [
                "distill",
                "init",
                "--mode",
                "on-policy",
                "--dataset",
                "data/anchor.zarr",
                "--output-dir",
                "runs/onpolicy",
                "--out",
                str(output_path),
            ],
        )

        result = CliRunner().invoke(
            main, ["distill", "spec", "report", str(output_path)]
        )

        output = _combined_output(result)
        assert result.exit_code == 0, output
        assert "6 anchor + 2 generated" in output


class TestRecipeExecution:
    def test_run_trains_the_student_of_an_offline_recipe(self, tmp_path: Path) -> None:
        """``spec run`` builds both models and the strategy, and takes its steps."""
        path = _write_recipe(tmp_path)
        checkpoint_dir = tmp_path / "run" / "checkpoints"
        payload = json.loads(path.read_text())
        payload["student"]["hooks"] = [
            {
                "spec": {
                    "cls_path": "nvalchemi.training.hooks.checkpoint.CheckpointHook",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "checkpoint_dir": str(checkpoint_dir),
                    "step_interval": 1,
                    "async_save": False,
                }
            }
        ]
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(
            main, ["distill", "spec", "run", str(path), "--no-report"]
        )

        assert result.exit_code == 0, _combined_output(result)
        assert (checkpoint_dir / "manifest.json").is_file()

    def test_run_generates_and_trains_an_on_policy_recipe(self, tmp_path: Path) -> None:
        """``spec run`` drives the segment loop rather than a dataloader."""
        path = _write_on_policy_recipe(tmp_path)

        result = CliRunner().invoke(
            main, ["distill", "spec", "run", str(path), "--no-report"]
        )

        assert result.exit_code == 0, _combined_output(result)

    def test_a_multi_store_anchor_runs_on_policy(self, tmp_path: Path) -> None:
        """An anchor named by dataset.paths is opened, not silently dropped."""
        path = _write_on_policy_recipe(tmp_path, anchors=2)

        result = CliRunner().invoke(
            main, ["distill", "spec", "run", str(path), "--no-report"]
        )

        assert result.exit_code == 0, _combined_output(result)

    def test_report_checks_every_store_a_multi_store_recipe_names(
        self, tmp_path: Path
    ) -> None:
        """Pre-flight existence checks reach dataset.paths, not only dataset.path."""
        path = _write_on_policy_recipe(tmp_path, anchors=2)
        payload = json.loads(path.read_text())
        payload["dataset"]["paths"][1] = str(tmp_path / "absent.zarr")
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        output = _combined_output(result)
        assert result.exit_code == 0, output
        assert "dataset.paths[1]" in output

    def test_resume_continues_an_interrupted_run(self, tmp_path: Path) -> None:
        """``spec resume`` restarts from a checkpoint and the recipe that wrote it."""
        path = _write_recipe(tmp_path)
        checkpoint_dir = tmp_path / "run" / "checkpoints"
        payload = json.loads(path.read_text())
        payload["student"]["hooks"] = [
            {
                "spec": {
                    "cls_path": "nvalchemi.training.hooks.checkpoint.CheckpointHook",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "checkpoint_dir": str(checkpoint_dir),
                    "step_interval": 1,
                    "async_save": False,
                }
            }
        ]
        path.write_text(json.dumps(payload))
        assert (
            CliRunner()
            .invoke(main, ["distill", "spec", "run", str(path), "--no-report"])
            .exit_code
            == 0
        )

        result = CliRunner().invoke(
            main,
            ["distill", "spec", "resume", str(checkpoint_dir), "--spec", str(path)],
        )

        assert result.exit_code == 0, _combined_output(result)

    def test_resume_reports_an_unreadable_checkpoint_cleanly(
        self, tmp_path: Path
    ) -> None:
        """A directory holding no checkpoint is a CLI error, not a traceback."""
        path = _write_recipe(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()

        result = CliRunner().invoke(
            main, ["distill", "spec", "resume", str(empty), "--spec", str(path)]
        )

        assert result.exit_code != 0
        assert "could not be restored" in _combined_output(result)

    def test_evaluate_reports_a_verdict_and_exits_on_a_missed_bar(
        self, tmp_path: Path
    ) -> None:
        """The acceptance report is rendered, exported, and gated on."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
                "batch_size": 4,
                "thresholds": {"max_forces_mae": 1e-9},
            },
        )
        student_checkpoint = tmp_path / "student-ckpt"
        save_checkpoint(
            student_checkpoint,
            models={
                "student": (
                    build_cli_student(),
                    create_model_spec(build_cli_student, hidden_dim=8),
                )
            },
        )
        report_path = tmp_path / "acceptance.json"

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "evaluate",
                str(path),
                "--student-checkpoint",
                str(student_checkpoint),
                "--json-out",
                str(report_path),
            ],
        )

        assert result.exit_code == 1, _combined_output(result)
        report = json.loads(report_path.read_text())
        assert report["accepted"] is False
        assert report["students"][0]["accuracy"]["forces_mae"] > 0.0

    def test_evaluate_accepts_a_student_that_clears_its_bars(
        self, tmp_path: Path
    ) -> None:
        """A cleared bar exits zero, which is what a sweep gates on."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
                "thresholds": {"max_forces_mae": 1e6},
            },
        )
        student_checkpoint = tmp_path / "student-ckpt"
        save_checkpoint(
            student_checkpoint,
            models={
                "student": (
                    build_cli_student(),
                    create_model_spec(build_cli_student, hidden_dim=8),
                )
            },
        )

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "evaluate",
                str(path),
                "--student-checkpoint",
                str(student_checkpoint),
            ],
        )

        assert result.exit_code == 0, _combined_output(result)
        assert "ACCEPT" in _combined_output(result)

    def test_a_recipe_without_bars_reports_numbers_and_no_verdict(
        self, tmp_path: Path
    ) -> None:
        """`--holdout` scores a bar-less recipe instead of refusing to run."""
        path = _write_recipe(tmp_path)
        student_checkpoint = tmp_path / "student-ckpt"
        save_checkpoint(
            student_checkpoint,
            models={
                "student": (
                    build_cli_student(),
                    create_model_spec(build_cli_student, hidden_dim=8),
                )
            },
        )
        report_path = tmp_path / "acceptance.json"

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "evaluate",
                str(path),
                "--student-checkpoint",
                str(student_checkpoint),
                "--holdout",
                str(tmp_path / "labeled.zarr"),
                "--json-out",
                str(report_path),
            ],
        )

        assert result.exit_code == 0, _combined_output(result)
        report = json.loads(report_path.read_text())
        assert report["accepted"] is True
        assert report["students"][0]["accuracy"]["forces_mae"] > 0.0

    def test_evaluate_without_a_holdout_says_what_to_add(self, tmp_path: Path) -> None:
        """A recipe with neither an evaluation section nor `--holdout` is refused."""
        path = _write_recipe(tmp_path)
        student_checkpoint = tmp_path / "student-ckpt"
        save_checkpoint(
            student_checkpoint,
            models={
                "student": (
                    build_cli_student(),
                    create_model_spec(build_cli_student, hidden_dim=8),
                )
            },
        )

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "evaluate",
                str(path),
                "--student-checkpoint",
                str(student_checkpoint),
            ],
        )

        assert result.exit_code != 0
        assert "no evaluation section" in _combined_output(result)


def test_the_distill_group_is_registered_on_the_training_entry_point() -> None:
    """The recipe CLI is a subgroup of `nvalchemi-training`, as the trainer's is."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, _combined_output(result)
    assert "distill" in result.output
