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

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from click.testing import CliRunner

from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.models.demo import DemoModelWrapper
from nvalchemi.training import save_checkpoint
from nvalchemi.training._spec import create_model_spec
from nvalchemi.training.cli import main
from nvalchemi.training.distillation import InProcessTeacherScorer, label_dataset
from nvalchemi.training.distillation import cli as distillation_cli
from nvalchemi.training.distillation.cli import DistillationJobSpec, _load_recipe
from nvalchemi.training.distillation.evaluation import (
    AcceptanceThresholds,
    measured_bars,
)
from nvalchemi.training.distillation.evaluation.accuracy import AccuracyMetrics
from nvalchemi.training.hooks.ddp import DDPHook
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


def _write_student_checkpoint(root: Path) -> Path:
    """Return a native checkpoint directory holding a rebuildable student."""
    save_checkpoint(
        root,
        models={
            "student": (
                build_cli_student(),
                create_model_spec(build_cli_student, hidden_dim=8),
            )
        },
    )
    return root


def _manifest_index(checkpoint_dir: Path) -> int:
    """Return the latest checkpoint index a run's manifest records."""
    return json.loads((checkpoint_dir / "manifest.json").read_text())[
        "checkpoint_index"
    ]


def _ema_hook_spec() -> dict[str, Any]:
    """Return a runtime hook that is not the one a checkpoint_dir needs."""
    return {
        "spec": {
            "cls_path": "nvalchemi.training.hooks.ema.EMAHook",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    }


def _write_recipe(tmp_path: Path, **overrides: Any) -> Path:
    """Write a runnable offline recipe to disk and return its path.

    The scaffold's own hooks are carried through, so a recipe written here
    checkpoints into ``run/checkpoints`` the way ``distill init`` leaves it.
    """
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


def _drop_the_loss_spec(payload: dict[str, Any]) -> None:
    """Remove a spec key the strategy bundle is required to carry."""
    del payload["strategy"]["loss_fn_spec"]


def _set_both_durations(payload: dict[str, Any]) -> None:
    """Size the run in epochs as well as steps, which the XOR forbids."""
    payload["strategy"]["num_epochs"] = 1


def _optimize_the_teacher(payload: dict[str, Any]) -> None:
    """Configure an optimizer for the model the strategy freezes by omission."""
    payload["strategy"]["optimizer_configs"]["teacher"] = payload["strategy"][
        "optimizer_configs"
    ]["student"]


def _optimize_nobody(payload: dict[str, Any]) -> None:
    """Rename the student's optimizer so nothing configures the student."""
    payload["strategy"]["optimizer_configs"] = {
        "critic": payload["strategy"]["optimizer_configs"]["student"]
    }


def _size_the_segment_loop_in_epochs(payload: dict[str, Any]) -> None:
    """Drop the step budget an on-policy run is sized in."""
    payload["strategy"]["num_steps"] = None
    payload["strategy"]["num_epochs"] = 1


def _ask_for_validation(payload: dict[str, Any]) -> None:
    """Add a cadence to a recipe whose dataset names no validation store."""
    payload["validation"] = {"every_n_epochs": 1}


_STRATEGY_GUARDS: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("offline", _drop_the_loss_spec, "missing required DistillationStrategy spec key"),
    ("offline", _set_both_durations, "exactly one of num_epochs or num_steps"),
    ("offline", _optimize_the_teacher, "frozen by omission"),
    ("offline", _optimize_nobody, "must configure the student"),
    ("on-policy", _size_the_segment_loop_in_epochs, "sized in optimizer steps"),
    ("offline", _ask_for_validation, "requires dataset.validation_path"),
]
"""One case per guard in ``DistillationJobSpec._validate_strategy``."""

_DEFAULT_QUANTITIES = ["energy", "forces"]
"""Quantities an ``EvaluationSpec`` compares unless the recipe names others."""

_SCORED_QUANTITIES = ["energy", "forces", "stress"]
"""Quantities a recipe compares to earn every accuracy bar there is."""

_MEASURABLE_BARS = sorted(
    measured_bars("accuracy", accuracy_quantities=_SCORED_QUANTITIES)
)
"""Bars a holdout pass over ``_SCORED_QUANTITIES`` fills."""

_UNMEASURABLE_BARS: list[tuple[str, float | bool]] = sorted(
    (bar, True if AcceptanceThresholds.model_fields[bar].annotation is bool else 0.5)
    for bar in set(AcceptanceThresholds.model_fields)
    - measured_bars("accuracy", accuracy_quantities=_DEFAULT_QUANTITIES)
)
"""Bars a default recipe may not carry, with a value moving each off its default."""


def _holdout_accuracy() -> AccuracyMetrics:
    """Return the accuracy metrics a holdout pass hands ``distill evaluate``."""
    return AccuracyMetrics(
        name="student",
        num_graphs=2,
        num_atoms=8,
        energy_mae=0.1,
        energy_rmse=0.1,
        energy_per_atom_mae=0.01,
        energy_per_atom_rmse=0.01,
        forces_mae=0.02,
        forces_rmse=0.02,
        stress_mae=0.03,
        stress_rmse=0.03,
        force_cosine_mean=0.9,
        force_cosine_aggregate=0.95,
    )


def _reject_json_constant(token: str) -> float:
    """Raise on the ``NaN``/``Infinity`` tokens plain JSON has no room for."""
    raise ValueError(f"{token} is not a JSON value.")


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

    def test_init_scaffolds_the_hook_that_writes_the_checkpoint_dir(
        self, tmp_path: Path
    ) -> None:
        """The scaffold carries the CheckpointHook its checkpoint_dir needs."""
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
                "--num-steps",
                "500",
                "--out",
                str(output),
            ],
        )

        assert result.exit_code == 0, _combined_output(result)
        job = _load_recipe(output)
        (hook,) = job.student.hooks
        assert hook.spec.cls_path == distillation_cli._CHECKPOINT_HOOK_PATH
        assert hook.spec.model_extra["checkpoint_dir"] == job.output.checkpoint_dir
        assert hook.spec.model_extra["step_interval"] == 50

    def test_an_on_policy_scaffold_refuses_to_seed_from_the_anchor(
        self, tmp_path: Path
    ) -> None:
        """Without --seed-dataset the scaffold is refused rather than written."""
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
                "--output-dir",
                "runs/onpolicy",
                "--out",
                str(output),
            ],
        )

        assert result.exit_code != 0
        message = _combined_output(result)
        assert "--seed-dataset" in message
        assert "anchor" in message
        assert not output.exists()

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


class TestStrategyValidation:
    @pytest.mark.parametrize(
        ("mode", "mutate", "message"),
        _STRATEGY_GUARDS,
        ids=[mutate.__name__.strip("_") for _, mutate, _ in _STRATEGY_GUARDS],
    )
    def test_a_strategy_guard_refuses_the_recipe(
        self,
        tmp_path: Path,
        mode: str,
        mutate: Callable[[dict[str, Any]], None],
        message: str,
    ) -> None:
        """Every guard on the strategy bundle fails the recipe at `spec report`."""
        write = _write_recipe if mode == "offline" else _write_on_policy_recipe
        path = write(tmp_path)
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        assert message in _combined_output(result)


class TestAcceptanceBars:
    @pytest.mark.parametrize(
        ("bar", "value"), _UNMEASURABLE_BARS, ids=[bar for bar, _ in _UNMEASURABLE_BARS]
    )
    def test_a_bar_evaluate_never_measures_is_refused(
        self, tmp_path: Path, bar: str, value: float | bool
    ) -> None:
        """A bar with no measurement behind it is refused when the recipe is read."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "thresholds": {bar: value},
            },
        )

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        message = _combined_output(result)
        assert "does not measure" in message
        assert bar in message

    def test_the_accuracy_bars_are_accepted(self, tmp_path: Path) -> None:
        """Every bar the holdout pass fills passes validation and is reported."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "quantities": _SCORED_QUANTITIES,
                "thresholds": {bar: 0.5 for bar in _MEASURABLE_BARS},
            },
        )

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        output = _combined_output(result)
        assert result.exit_code == 0, output
        assert "max_forces_mae" in output

    def test_a_stress_bar_the_default_quantities_never_compare_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A holdout scored on energy and forces fills no stress bar."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "thresholds": {"max_stress_mae": 0.002},
            },
        )

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        message = _combined_output(result)
        assert "max_stress_mae" in message
        assert "'forces'" in message

    def test_a_stress_bar_is_accepted_once_stress_is_compared(
        self, tmp_path: Path
    ) -> None:
        """Naming the quantity is what turns the same bar from refused into gated."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "quantities": _SCORED_QUANTITIES,
                "thresholds": {"max_stress_mae": 0.002},
            },
        )

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        output = _combined_output(result)
        assert result.exit_code == 0, output
        assert "max_stress_mae" in output

    def test_a_force_bar_is_refused_when_only_energy_is_compared(
        self, tmp_path: Path
    ) -> None:
        """Narrowing the quantities narrows the bars, not only the reported rows."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "quantities": ["energy"],
                "thresholds": {"max_forces_mae": 0.05},
            },
        )

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        assert result.exit_code != 0
        message = _combined_output(result)
        assert "max_forces_mae" in message
        assert "'energy'" in message


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

    def test_a_scaffolded_recipe_reports_no_pre_flight_issues(
        self, tmp_path: Path
    ) -> None:
        """The scaffold's own checkpoint intent is complete rather than warned about."""
        path = _write_recipe(tmp_path)

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        output = _combined_output(result)
        assert result.exit_code == 0, output
        assert "output.checkpoint_dir" not in output

    @pytest.mark.parametrize("hooks", [[], [_ema_hook_spec()]], ids=["none", "ema"])
    def test_a_recipe_without_a_checkpoint_hook_is_warned_about(
        self, tmp_path: Path, hooks: list[dict[str, Any]]
    ) -> None:
        """A hook that is not a CheckpointHook does not silence the warning."""
        path = _write_recipe(tmp_path)
        payload = json.loads(path.read_text())
        payload["student"]["hooks"] = hooks
        path.write_text(json.dumps(payload))

        result = CliRunner().invoke(main, ["distill", "spec", "report", str(path)])

        output = _combined_output(result)
        assert result.exit_code == 0, output
        assert "output.checkpoint_dir" in output

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
                "--seed-dataset",
                "data/seeds.zarr",
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

    def test_resume_rebuilds_the_hooks_the_recipe_declares(
        self, tmp_path: Path
    ) -> None:
        """A resumed run keeps checkpointing: the manifest index advances again."""
        path = _write_recipe(tmp_path)
        checkpoint_dir = tmp_path / "run" / "checkpoints"
        assert (
            CliRunner()
            .invoke(main, ["distill", "spec", "run", str(path), "--no-report"])
            .exit_code
            == 0
        )
        before = _manifest_index(checkpoint_dir)

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "spec",
                "resume",
                str(checkpoint_dir),
                "--spec",
                str(path),
                "--checkpoint-index",
                "0",
            ],
        )

        assert result.exit_code == 0, _combined_output(result)
        assert _manifest_index(checkpoint_dir) == before + 1

    def test_resume_restarts_from_the_requested_checkpoint_index(
        self, tmp_path: Path
    ) -> None:
        """The last index has no steps left to take; an earlier one has."""
        path = _write_recipe(tmp_path)
        checkpoint_dir = tmp_path / "run" / "checkpoints"
        assert (
            CliRunner()
            .invoke(main, ["distill", "spec", "run", str(path), "--no-report"])
            .exit_code
            == 0
        )
        completed = _manifest_index(checkpoint_dir)

        latest = CliRunner().invoke(
            main,
            ["distill", "spec", "resume", str(checkpoint_dir), "--spec", str(path)],
        )
        assert latest.exit_code == 0, _combined_output(latest)
        after_latest = _manifest_index(checkpoint_dir)
        earlier = CliRunner().invoke(
            main,
            [
                "distill",
                "spec",
                "resume",
                str(checkpoint_dir),
                "--spec",
                str(path),
                "--checkpoint-index",
                "0",
            ],
        )

        assert earlier.exit_code == 0, _combined_output(earlier)
        assert after_latest == completed
        assert _manifest_index(checkpoint_dir) == completed + 1

    def test_resume_under_a_recipe_of_the_other_mode_is_a_clean_error(
        self, tmp_path: Path
    ) -> None:
        """A recipe whose mode the checkpoint contradicts is a CLI error."""
        offline = tmp_path / "offline"
        on_policy = tmp_path / "on-policy"
        offline.mkdir()
        on_policy.mkdir()
        path = _write_recipe(offline)
        checkpoint_dir = offline / "run" / "checkpoints"
        assert (
            CliRunner()
            .invoke(main, ["distill", "spec", "run", str(path), "--no-report"])
            .exit_code
            == 0
        )
        mismatched = _write_on_policy_recipe(on_policy)

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "spec",
                "resume",
                str(checkpoint_dir),
                "--spec",
                str(mismatched),
            ],
        )

        assert result.exit_code != 0
        assert "the run could not be started" in _combined_output(result)

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


class TestEvaluateStudent:
    def test_the_scaffolded_flow_runs_and_then_gates_its_own_student(
        self, tmp_path: Path
    ) -> None:
        """`init` -> `spec run` -> `evaluate` needs nothing the scaffold omits."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
                "thresholds": {"max_forces_mae": 1e6},
            },
        )
        checkpoint_dir = tmp_path / "run" / "checkpoints"

        run = CliRunner().invoke(
            main, ["distill", "spec", "run", str(path), "--no-report"]
        )
        result = CliRunner().invoke(
            main,
            [
                "distill",
                "evaluate",
                str(path),
                "--student-checkpoint",
                str(checkpoint_dir),
            ],
        )

        assert run.exit_code == 0, _combined_output(run)
        assert (checkpoint_dir / "manifest.json").is_file()
        assert result.exit_code == 0, _combined_output(result)
        assert "ACCEPT" in _combined_output(result)

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
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")
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
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")

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
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")
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
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")

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

    def test_the_holdout_option_wins_over_the_recipe(self, tmp_path: Path) -> None:
        """`--holdout` replaces the recipe's store rather than being replaced by it."""
        absent = tmp_path / "absent.zarr"
        path = _write_recipe(
            tmp_path,
            evaluation={"holdout_path": str(absent), "targets": "teacher"},
        )
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")
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
        assert (
            json.loads(report_path.read_text())["students"][0]["accuracy"]["forces_mae"]
            > 0.0
        )

    @pytest.mark.parametrize(
        ("override", "field"),
        [(False, "evaluation.holdout_path"), (True, "--holdout")],
        ids=["recipe", "option"],
    )
    def test_a_holdout_that_is_not_on_disk_is_named(
        self, tmp_path: Path, override: bool, field: str
    ) -> None:
        """A store that does not exist is a CLI error naming where it came from."""
        absent = tmp_path / "absent.zarr"
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr" if override else absent),
                "targets": "teacher",
            },
        )
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "evaluate",
                str(path),
                "--student-checkpoint",
                str(student_checkpoint),
                *(["--holdout", str(absent)] if override else []),
            ],
        )

        assert result.exit_code != 0
        message = _combined_output(result)
        assert field in message
        assert str(absent) in message

    def test_reference_targets_score_against_the_store_s_own_labels(
        self, tmp_path: Path
    ) -> None:
        """`targets='reference'` skips the teacher, and with it its cosine rows."""
        holdout = _write_labeled_store(
            tmp_path / "reference.zarr", 6, 8, 900, predictions=True
        )
        path = _write_recipe(
            tmp_path,
            evaluation={"holdout_path": str(holdout), "targets": "reference"},
        )
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")
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

        assert result.exit_code == 0, _combined_output(result)
        accuracy = json.loads(report_path.read_text())["students"][0]["accuracy"]
        assert accuracy["forces_mae"] > 0.0
        assert "force_cosine_aggregate" not in accuracy

    def test_reference_targets_over_an_unlabeled_store_are_refused(
        self, tmp_path: Path
    ) -> None:
        """An anchor carries no labels of its own, and the CLI says so."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "reference",
            },
        )
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")

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
        message = _combined_output(result)
        assert "carries no target the evaluation asked for" in message
        assert "'reference'" in message

    def test_the_recipe_s_quantities_narrow_the_report(self, tmp_path: Path) -> None:
        """A recipe asking for forces alone gets no energy rows back."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
                "quantities": ["forces"],
            },
        )
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")
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

        assert result.exit_code == 0, _combined_output(result)
        accuracy = json.loads(report_path.read_text())["students"][0]["accuracy"]
        assert accuracy["forces_mae"] > 0.0
        assert "energy_mae" not in accuracy

    def test_the_batch_size_reaches_the_holdout_loader(self, tmp_path: Path) -> None:
        """The recipe sizes the holdout loader, and `--batch-size` overrides it."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
                "batch_size": 4,
            },
        )
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")
        command = [
            "distill",
            "evaluate",
            str(path),
            "--student-checkpoint",
            str(student_checkpoint),
        ]

        with patch.object(
            distillation_cli,
            "_build_dataloader",
            wraps=distillation_cli._build_dataloader,
        ) as loader:
            from_recipe = CliRunner().invoke(main, command)
            recipe_size = loader.call_args.kwargs["batch_size"]
            overridden = CliRunner().invoke(main, [*command, "--batch-size", "2"])
            option_size = loader.call_args.kwargs["batch_size"]

        assert from_recipe.exit_code == 0, _combined_output(from_recipe)
        assert overridden.exit_code == 0, _combined_output(overridden)
        assert recipe_size == 4
        assert option_size == 2

    def test_a_student_checkpoint_holding_no_manifest_is_a_clean_error(
        self, tmp_path: Path
    ) -> None:
        """A plain directory passes the parser, so the load is what has to report it."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
            },
        )
        empty = tmp_path / "not-a-checkpoint"
        empty.mkdir()

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "evaluate",
                str(path),
                "--student-checkpoint",
                str(empty),
            ],
        )

        assert result.exit_code != 0
        message = _combined_output(result)
        assert "could not be read" in message
        assert str(empty) in message
        assert "--checkpoint-index" in message

    def test_a_checkpoint_index_the_run_never_wrote_is_a_clean_error(
        self, tmp_path: Path
    ) -> None:
        """An index past the last saved one names the index rather than a traceback."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
            },
        )
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")

        result = CliRunner().invoke(
            main,
            [
                "distill",
                "evaluate",
                str(path),
                "--student-checkpoint",
                str(student_checkpoint),
                "--checkpoint-index",
                "99",
            ],
        )

        assert result.exit_code != 0
        message = _combined_output(result)
        assert "could not be read" in message
        assert "99" in message

    def test_a_report_the_bars_cannot_form_is_a_clean_error(
        self, tmp_path: Path
    ) -> None:
        """The report's own contract errors reach the user as CLI errors."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
            },
        )
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")

        with patch.object(
            distillation_cli,
            "build_acceptance_report",
            side_effect=ValueError("no student of the family carries drafter metrics"),
        ):
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
        message = _combined_output(result)
        assert "acceptance report could not be formed" in message
        assert "drafter metrics" in message

    @pytest.mark.parametrize(
        ("value", "token"),
        [(math.nan, "nan"), (math.inf, "inf"), (-math.inf, "-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_a_nonfinite_metric_is_exported_as_a_token_json_can_hold(
        self, tmp_path: Path, value: float, token: str
    ) -> None:
        """A metric json cannot spell is named as a string, so the export parses."""
        path = _write_recipe(
            tmp_path,
            evaluation={
                "holdout_path": str(tmp_path / "labeled.zarr"),
                "targets": "teacher",
            },
        )
        student_checkpoint = _write_student_checkpoint(tmp_path / "student-ckpt")
        report_path = tmp_path / "acceptance.json"
        metrics = dataclasses.replace(_holdout_accuracy(), force_cosine_aggregate=value)

        with patch.object(distillation_cli, "evaluate_accuracy", return_value=metrics):
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

        assert result.exit_code == 0, _combined_output(result)
        report = json.loads(
            report_path.read_text(), parse_constant=_reject_json_constant
        )
        assert report["students"][0]["accuracy"]["force_cosine_aggregate"] == token


def test_the_distill_group_is_registered_on_the_training_entry_point() -> None:
    """The recipe CLI is a subgroup of `nvalchemi-training`, as the trainer's is."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, _combined_output(result)
    assert "distill" in result.output


class _FakeManager:
    """Distributed manager reporting a fixed world size, rank, and device."""

    def __init__(
        self, *, world_size: int = 2, rank: int = 0, device: str = "cpu"
    ) -> None:
        """Report a world of *world_size* ranks, seen from *rank* on *device*."""
        self.world_size = world_size
        self.rank = rank
        self.global_rank = rank
        self.local_rank = rank
        self.device = torch.device(device)
        self.broadcast_buffers = False
        self.find_unused_parameters = True

    def is_initialized(self) -> bool:
        """Report communication as established for any multi-rank world."""
        return self.world_size > 1


class _RecordingDDP(torch.nn.Module):
    """Data-parallel stand-in wrapping a model without a process group."""

    def __init__(self, module: torch.nn.Module, **kwargs: Any) -> None:  # noqa: ARG002
        """Wrap *module* the way ``DistributedDataParallel`` would."""
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward the pass a real wrapper would all-reduce the gradients of."""
        return self.module(*args, **kwargs)


def _write_cuda_recipe(tmp_path: Path, device: str) -> Path:
    """Write an offline recipe whose strategy is pinned to *device*."""
    path = _write_recipe(tmp_path)
    payload = json.loads(path.read_text())
    payload["strategy"]["devices"] = [device]
    path.write_text(json.dumps(payload))
    return path


def _optimizer_state_devices(strategy: Any) -> set[torch.device]:
    """Return every device the strategy's optimizer state tensors sit on."""
    return {
        value.device
        for optimizer in strategy._optimizers
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    }


def _run_as_rank(args: list[str], manager: _FakeManager) -> tuple[Any, list[Any]]:
    """Invoke the distill CLI as a rank of *manager*'s world, capturing its strategy.

    ``DDPHook`` makes the rank's GPU current for the rest of the process, which
    would leave an index-less ``"cuda"`` resolving to it in every later test, so
    the device this process was on is restored on the way out.
    """
    executed: list[Any] = []
    original = distillation_cli._execute_strategy

    def execute(job: Any, strategy: Any, stack: Any, **kwargs: Any) -> None:
        executed.append(strategy)
        original(job, strategy, stack, **kwargs)

    pins_a_gpu = manager.device.type == "cuda" and torch.cuda.is_available()
    pinned = torch.cuda.current_device() if pins_a_gpu else None
    try:
        with (
            patch.object(
                distillation_cli, "_setup_distributed_manager", lambda enabled: manager
            ),
            patch.object(distillation_cli, "_execute_strategy", execute),
            patch.object(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP),
        ):
            result = CliRunner().invoke(main, args)
    finally:
        if pinned is not None:
            torch.cuda.set_device(pinned)
    return result, executed


class TestDistributedRecipeExecution:
    def test_run_attaches_a_ddp_hook_and_the_manager_it_was_given(
        self, tmp_path: Path
    ) -> None:
        """``spec run --distributed`` wires the manager and the hook onto the strategy."""
        path = _write_recipe(tmp_path)
        manager = _FakeManager()

        result, executed = _run_as_rank(
            ["distill", "spec", "run", str(path), "--no-report", "--distributed"],
            manager,
        )

        assert result.exit_code == 0, _combined_output(result)
        strategy = executed[0]
        assert strategy.distributed_manager is manager
        assert any(isinstance(hook, DDPHook) for hook in strategy.hooks)

    def test_the_ddp_backend_reaches_the_hook(self, tmp_path: Path) -> None:
        """``--ddp-backend`` is the backend the attached hook was built with."""
        path = _write_recipe(tmp_path)

        result, executed = _run_as_rank(
            [
                "distill",
                "spec",
                "run",
                str(path),
                "--no-report",
                "--distributed",
                "--ddp-backend",
                "gloo",
            ],
            _FakeManager(),
        )

        assert result.exit_code == 0, _combined_output(result)
        backends = [
            hook.backend for hook in executed[0].hooks if isinstance(hook, DDPHook)
        ]
        assert backends == ["gloo"]

    def test_no_distributed_under_a_world_of_two_attaches_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit refusal outranks the ``WORLD_SIZE`` the launcher exported."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        path = _write_recipe(tmp_path)
        executed: list[Any] = []
        original = distillation_cli._execute_strategy

        def execute(job: Any, strategy: Any, stack: Any, **kwargs: Any) -> None:
            executed.append(strategy)
            original(job, strategy, stack, **kwargs)

        with patch.object(distillation_cli, "_execute_strategy", execute):
            result = CliRunner().invoke(
                main,
                [
                    "distill",
                    "spec",
                    "run",
                    str(path),
                    "--no-report",
                    "--no-distributed",
                ],
            )

        assert result.exit_code == 0, _combined_output(result)
        assert executed[0].distributed_manager is None
        assert not any(isinstance(hook, DDPHook) for hook in executed[0].hooks)

    def test_an_on_policy_recipe_under_a_world_of_two_is_a_clean_error(
        self, tmp_path: Path
    ) -> None:
        """The segment loop's single-process rule surfaces as a CLI error."""
        path = _write_on_policy_recipe(tmp_path)

        result, _ = _run_as_rank(
            ["distill", "spec", "run", str(path), "--no-report", "--distributed"],
            _FakeManager(),
        )

        assert result.exit_code != 0
        output = _combined_output(result)
        assert "the run could not be started" in output
        assert "single-process for now" in output

    def test_a_resuming_rank_defaults_the_load_device_to_its_own(self) -> None:
        """An omitted ``--map-location`` becomes the device this rank runs on."""
        manager = _FakeManager(rank=1, device="cuda:1")

        assert distillation_cli._restart_map_location(manager, None) == "cuda:1"

    def test_a_single_process_resume_keeps_the_map_location_it_was_given(self) -> None:
        """Outside a multi-rank launch the flag is passed through untouched."""
        assert distillation_cli._restart_map_location(None, "cpu") == "cpu"
        assert (
            distillation_cli._restart_map_location(
                _FakeManager(world_size=1, device="cuda:0"), "cpu"
            )
            == "cpu"
        )

    def test_a_map_location_that_is_not_this_ranks_device_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Loading onto another rank's device would hang the world, so it is refused."""
        path = _write_recipe(tmp_path)
        checkpoint_dir = tmp_path / "run" / "checkpoints"
        assert (
            CliRunner()
            .invoke(main, ["distill", "spec", "run", str(path), "--no-report"])
            .exit_code
            == 0
        )

        result, _ = _run_as_rank(
            [
                "distill",
                "spec",
                "resume",
                str(checkpoint_dir),
                "--spec",
                str(path),
                "--distributed",
                "--map-location",
                "cuda:0",
            ],
            _FakeManager(rank=1, device="cuda:1"),
        )

        assert result.exit_code != 0
        output = _combined_output(result)
        assert "is not this rank's device" in output
        assert "Tensors of the same index must be on the same device" in output

    @pytest.mark.multigpu
    def test_a_resuming_rank_lands_every_optimizer_tensor_on_its_own_device(
        self, tmp_path: Path
    ) -> None:
        """A restart named for this rank leaves nothing behind on rank zero's GPU."""
        path = _write_cuda_recipe(tmp_path, "cuda:0")
        checkpoint_dir = tmp_path / "run" / "checkpoints"
        written, _ = _run_as_rank(
            ["distill", "spec", "run", str(path), "--no-report", "--distributed"],
            _FakeManager(rank=0, device="cuda:0"),
        )
        assert written.exit_code == 0, _combined_output(written)

        result, executed = _run_as_rank(
            [
                "distill",
                "spec",
                "resume",
                str(checkpoint_dir),
                "--spec",
                str(path),
                "--checkpoint-index",
                "0",
                "--distributed",
            ],
            _FakeManager(rank=1, device="cuda:1"),
        )

        assert result.exit_code == 0, _combined_output(result)
        resumed = executed[0]
        assert _optimizer_state_devices(resumed) == {torch.device("cuda", 1)}
        assert resumed.step_count == 2
