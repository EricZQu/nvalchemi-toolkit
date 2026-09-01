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
"""Tests for :mod:`nvalchemi.training.distillation.evaluation.report`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rich.console import Console

from nvalchemi.training.distillation.evaluation import (
    AcceptanceReport,
    AcceptanceThresholds,
    AccuracyMetrics,
    DrafterMetrics,
    StabilityMetrics,
    StudentEvaluation,
    ThroughputMetrics,
    build_acceptance_report,
)


def _make_accuracy(
    name: str = "student", forces_mae: float = 0.02, energy_mae: float = 0.001
) -> AccuracyMetrics:
    """Return accuracy metrics with the two fields the gates read."""
    return AccuracyMetrics(
        name=name,
        num_graphs=4,
        num_atoms=40,
        energy_per_atom_mae=energy_mae,
        forces_mae=forces_mae,
        force_cosine_mean=0.99,
    )


def _make_stability(drift: float = 0.001) -> StabilityMetrics:
    """Return stability metrics with a chosen per-nanosecond drift."""
    return StabilityMetrics(
        num_samples=10,
        first_step=0,
        last_step=90,
        energy_drift_per_atom=1e-4,
        energy_drift_per_atom_per_step=1e-6,
        energy_drift_per_atom_per_ns=drift,
        max_momentum_drift=1e-8,
        timestep_fs=1.0,
    )


def _make_throughput(atoms_per_second: float = 2.0e6) -> ThroughputMetrics:
    """Return throughput metrics with a chosen atoms-per-second rate."""
    return ThroughputMetrics(
        steps_per_second=atoms_per_second / 1000.0,
        atoms_per_second=atoms_per_second,
        ns_per_day=50.0,
        num_atoms=1000,
        num_graphs=1,
        warmup_steps=5,
        measured_steps=20,
        elapsed_seconds=0.5,
        device="cpu",
    )


def _make_student(
    name: str = "student",
    forces_mae: float = 0.02,
    atoms_per_second: float = 2.0e6,
    **kwargs: object,
) -> StudentEvaluation:
    """Return a fully measured student evaluation with overridable slots."""
    return StudentEvaluation(
        name=name,
        accuracy=_make_accuracy(name, forces_mae=forces_mae),
        stability=_make_stability(),
        throughput=_make_throughput(atoms_per_second),
        **kwargs,
    )


def _render(report: AcceptanceReport) -> str:
    """Return the report's Rich rendering as plain text."""
    console = Console(width=200, record=True, force_terminal=False)
    console.print(report)
    return console.export_text()


class TestAcceptanceThresholds:
    """Validation of the bars themselves."""

    def test_every_bar_defaults_to_unset(self) -> None:
        """A default threshold set tests nothing and accepts everyone."""
        report = build_acceptance_report([_make_student()])
        assert report.accepted
        assert report.verdicts[0].checks == ()

    def test_unknown_bar_is_rejected(self) -> None:
        """The threshold model forbids fields it does not know how to check."""
        with pytest.raises(ValidationError):
            AcceptanceThresholds(max_dipole_mae=0.1)

    def test_negative_bar_is_rejected(self) -> None:
        """An error bar has to be a positive number."""
        with pytest.raises(ValidationError):
            AcceptanceThresholds(max_forces_mae=-1.0)


class TestAcceptanceVerdicts:
    """Per-student verdicts against configured bars."""

    def test_a_student_inside_every_bar_is_accepted(self) -> None:
        """Clearing accuracy, stability, and speed bars accepts the student."""
        report = build_acceptance_report(
            [_make_student()],
            AcceptanceThresholds(
                max_forces_mae=0.05,
                max_energy_per_atom_mae=0.005,
                min_force_cosine=0.95,
                max_energy_drift_per_atom_per_ns=0.01,
                min_atoms_per_second=1.0e6,
            ),
        )
        assert report.accepted
        assert all(check.passed for check in report.verdicts[0].checks)

    def test_a_student_outside_one_bar_is_rejected(self) -> None:
        """One failed check is enough to reject."""
        report = build_acceptance_report(
            [_make_student(forces_mae=0.2)],
            AcceptanceThresholds(max_forces_mae=0.05, min_atoms_per_second=1.0e6),
        )
        assert not report.accepted
        failed = [check for check in report.verdicts[0].checks if not check.passed]
        assert [check.name for check in failed] == ["forces_mae"]

    def test_a_bar_with_no_measurement_behind_it_fails(self) -> None:
        """An unmeasured metric fails its bar rather than skipping it."""
        report = build_acceptance_report(
            [StudentEvaluation(name="student", accuracy=_make_accuracy())],
            AcceptanceThresholds(min_atoms_per_second=1.0e6),
        )
        check = report.verdicts[0].checks[0]
        assert not check.passed
        assert check.detail == "not measured"
        assert check.value is None

    def test_minimum_bars_compare_in_the_other_direction(self) -> None:
        """A throughput floor passes when the measurement is above it."""
        report = build_acceptance_report(
            [_make_student(atoms_per_second=5.0e5)],
            AcceptanceThresholds(min_atoms_per_second=1.0e6),
        )
        check = report.verdicts[0].checks[0]
        assert check.comparison == ">="
        assert not check.passed


class TestFromScratchGate:
    """The distilled student has to beat its equal-size from-scratch twin."""

    def test_a_student_beating_the_baseline_passes_the_gate(self) -> None:
        """Lower error than the from-scratch student clears the gate."""
        report = build_acceptance_report(
            [
                _make_student(
                    forces_mae=0.02,
                    baseline_accuracy=_make_accuracy("scratch", forces_mae=0.05),
                )
            ],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        check = report.verdicts[0].checks[0]
        assert check.name == "from_scratch_ratio"
        assert check.value == pytest.approx(1.0)
        assert check.passed

    def test_losing_on_any_shared_metric_fails_the_gate(self) -> None:
        """The worst shared metric decides, so winning on energy is not enough."""
        baseline = AccuracyMetrics(
            name="scratch",
            num_graphs=4,
            num_atoms=40,
            energy_per_atom_mae=0.01,
            forces_mae=0.01,
        )
        report = build_acceptance_report(
            [_make_student(forces_mae=0.02, baseline_accuracy=baseline)],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        assert report.verdicts[0].checks[0].value == pytest.approx(2.0)
        assert not report.accepted

    def test_a_demanded_margin_tightens_the_gate(self) -> None:
        """A margin below one demands the distilled student win by that factor."""
        report = build_acceptance_report(
            [
                _make_student(
                    forces_mae=0.02,
                    baseline_accuracy=_make_accuracy("scratch", forces_mae=0.025),
                )
            ],
            AcceptanceThresholds(
                require_from_scratch_baseline=True, from_scratch_margin=0.5
            ),
        )
        assert not report.accepted

    def test_a_missing_baseline_fails_the_gate(self) -> None:
        """The gate cannot be satisfied by simply not running the baseline."""
        report = build_acceptance_report(
            [_make_student()],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        check = report.verdicts[0].checks[0]
        assert not check.passed
        assert check.detail == "no from-scratch baseline supplied"

    def test_a_baseline_sharing_no_metric_fails_the_gate(self) -> None:
        """Two evaluations measured on different quantities cannot be compared."""
        baseline = AccuracyMetrics(name="scratch", num_graphs=4, num_atoms=40)
        report = build_acceptance_report(
            [_make_student(baseline_accuracy=baseline)],
            AcceptanceThresholds(require_from_scratch_baseline=True),
        )
        assert report.verdicts[0].checks[0].detail.startswith("baseline shares no")

    def test_the_gate_is_off_unless_it_is_asked_for(self) -> None:
        """A baseline that is present but unrequested produces no check."""
        report = build_acceptance_report(
            [_make_student(baseline_accuracy=_make_accuracy("scratch"))]
        )
        assert report.verdicts[0].checks == ()


class TestParetoTable:
    """Speed against accuracy across a family of students."""

    def test_dominated_students_are_left_off_the_front(self) -> None:
        """A student both slower and less accurate than another is dominated."""
        report = build_acceptance_report(
            [
                _make_student("small", forces_mae=0.04, atoms_per_second=4.0e6),
                _make_student("medium", forces_mae=0.02, atoms_per_second=2.0e6),
                _make_student("dominated", forces_mae=0.05, atoms_per_second=1.0e6),
            ]
        )
        assert report.pareto_front == ("small", "medium")

    def test_students_without_a_speed_measurement_are_not_ranked(self) -> None:
        """The trade-off needs both axes, so an unmeasured student is skipped."""
        report = build_acceptance_report(
            [
                _make_student("measured"),
                StudentEvaluation(name="accuracy-only", accuracy=_make_accuracy()),
            ]
        )
        assert report.pareto_front == ("measured",)

    def test_the_table_lists_every_student_with_its_verdict(self) -> None:
        """The rendered table carries every student, ranked or not."""
        report = build_acceptance_report(
            [
                _make_student("small", forces_mae=0.04, atoms_per_second=4.0e6),
                _make_student("large", forces_mae=0.2, atoms_per_second=1.0e6),
            ],
            AcceptanceThresholds(max_forces_mae=0.05),
        )
        rendered = _render(report)
        assert "small" in rendered
        assert "large" in rendered
        assert "REJECT" in rendered


class TestDrafterRows:
    """The deferred speculative-MD rows of the report."""

    def test_drafter_rows_are_omitted_when_no_student_carries_them(self) -> None:
        """The speculative table is left out entirely, not rendered empty."""
        assert "Speculative MD" not in _render(
            build_acceptance_report([_make_student()])
        )

    def test_drafter_rows_render_once_metrics_are_supplied(self) -> None:
        """A student carrying drafter metrics gets its acceptance-rate row."""
        report = build_acceptance_report(
            [
                _make_student(
                    drafter=DrafterMetrics(
                        acceptance_rate=0.8, speculative_speedup=2.5, draft_steps=4
                    )
                )
            ]
        )
        rendered = _render(report)
        assert "Speculative MD" in rendered
        assert "0.8" in rendered

    def test_an_acceptance_rate_bar_is_checked_against_the_drafter(self) -> None:
        """The drafter bar behaves like every other minimum bar."""
        report = build_acceptance_report(
            [_make_student(drafter=DrafterMetrics(acceptance_rate=0.4))],
            AcceptanceThresholds(min_drafter_acceptance_rate=0.6),
        )
        assert not report.accepted
        assert report.verdicts[0].checks[0].name == "drafter_acceptance_rate"


class TestReportExports:
    """Plain-dictionary and scalar exports of a finished report."""

    def test_to_dict_carries_thresholds_students_and_verdicts(self) -> None:
        """The export is a plain structure with every measured section in it."""
        report = build_acceptance_report(
            [_make_student()], AcceptanceThresholds(max_forces_mae=0.05)
        )
        exported = report.to_dict()
        assert exported["accepted"] is True
        assert exported["thresholds"]["max_forces_mae"] == 0.05
        student = exported["students"][0]
        assert student["accuracy"]["forces_mae"] == 0.02
        assert student["throughput"]["num_atoms"] == 1000
        assert student["verdict"]["checks"][0]["name"] == "forces_mae"

    def test_scalars_are_flat_and_numeric(self) -> None:
        """Every exported scalar is a float keyed by student, group, and metric."""
        report = build_acceptance_report([_make_student()])
        scalars = report.scalars()
        assert scalars["student/accepted"] == 1.0
        assert scalars["student/accuracy/forces_mae"] == 0.02
        assert scalars["student/stability/energy_drift_per_atom_per_ns"] == 0.001
        assert all(isinstance(value, float) for value in scalars.values())

    def test_unmeasured_sections_are_left_out_of_the_export(self) -> None:
        """A student with only accuracy exports only accuracy."""
        report = build_acceptance_report(
            [StudentEvaluation(name="student", accuracy=_make_accuracy())]
        )
        assert "throughput" not in report.to_dict()["students"][0]

    def test_the_verdict_table_names_every_check(self) -> None:
        """Each applied bar becomes a row of the acceptance table."""
        report = build_acceptance_report(
            [_make_student()],
            AcceptanceThresholds(max_forces_mae=0.05, min_atoms_per_second=1.0e6),
        )
        rendered = _render(report)
        assert "forces_mae" in rendered
        assert "atoms_per_second" in rendered
        assert "ACCEPT" in rendered


class TestReportConstruction:
    """Guards on building a report at all."""

    def test_an_empty_family_is_rejected(self) -> None:
        """There is nothing to decide without a student."""
        with pytest.raises(ValueError, match="At least one student"):
            build_acceptance_report([])

    def test_duplicate_student_names_are_rejected(self) -> None:
        """Names key the exports, so two students cannot share one."""
        with pytest.raises(ValueError, match="must be unique"):
            build_acceptance_report([_make_student(), _make_student()])
