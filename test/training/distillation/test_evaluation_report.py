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

import dataclasses
import json

import pytest
from pydantic import ValidationError
from rich.console import Console

from nvalchemi.training.distillation.evaluation import (
    AcceptanceReport,
    AcceptanceThresholds,
    AccuracyMetrics,
    DrafterMetrics,
    ExtensivityMetrics,
    RDFComparison,
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
        force_cosine_aggregate=0.99,
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


def _make_rdf(
    jensen_shannon: float = 0.02, pair: tuple[int, int] | None = None
) -> RDFComparison:
    """Return a structural comparison at a chosen species resolution."""
    return RDFComparison(
        jensen_shannon=jensen_shannon,
        l1=0.3,
        max_deviation=0.1,
        num_bins=24,
        pair=pair,
    )


def _make_extensivity() -> ExtensivityMetrics:
    """Return an energy-scaling result over a doubled cell."""
    return ExtensivityMetrics(
        repeats=(2, 1, 1),
        num_graphs=2,
        max_error_per_atom=1e-6,
        mean_error_per_atom=5e-7,
        max_relative_error=1e-8,
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


def _make_student_on(name: str, **workload: int) -> StudentEvaluation:
    """Return a student whose speed was measured on a chosen ``(atoms, graphs)``."""
    return StudentEvaluation(
        name=name,
        accuracy=_make_accuracy(name),
        throughput=dataclasses.replace(_make_throughput(), **workload),
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

    def test_the_structure_bar_says_it_read_a_species_blind_curve(self) -> None:
        """A pooled g(r) is labelled as one, so the bar is not read as more."""
        report = build_acceptance_report(
            [_make_student(rdf=_make_rdf())],
            AcceptanceThresholds(max_rdf_jensen_shannon=0.1),
        )
        assert report.verdicts[0].checks[0].detail == "species-blind total g(r)"
        assert "species-blind" in _render(report)

    def test_the_structure_bar_names_the_species_pair_it_resolved(self) -> None:
        """A partial g_ab(r) is reported as the observable it actually is."""
        report = build_acceptance_report(
            [_make_student(rdf=_make_rdf(pair=(11, 17)))],
            AcceptanceThresholds(max_rdf_jensen_shannon=0.1),
        )
        check = report.verdicts[0].checks[0]
        assert check.detail == "partial g(r) of atomic numbers [11, 17]"
        assert check.passed

    def test_the_force_cosine_bar_reads_the_magnitude_weighted_alignment(self) -> None:
        """The bar reads the aggregate, not the mean the low-force tail dominates."""
        accuracy = AccuracyMetrics(
            name="student",
            num_graphs=2,
            num_atoms=54,
            forces_mae=0.004,
            force_cosine_mean=0.51,
            force_cosine_aggregate=0.97,
        )
        report = build_acceptance_report(
            [StudentEvaluation(name="student", accuracy=accuracy)],
            AcceptanceThresholds(min_force_cosine=0.9),
        )
        check = report.verdicts[0].checks[0]
        assert report.verdicts[0].accepted is True
        assert check.name == "force_cosine_aggregate"
        assert check.value == 0.97

    @pytest.mark.parametrize(
        ("thresholds", "name"),
        [
            (AcceptanceThresholds(max_forces_mae=0.02), "forces_mae"),
            (AcceptanceThresholds(min_atoms_per_second=2.0e6), "atoms_per_second"),
        ],
        ids=["at-the-maximum", "at-the-minimum"],
    )
    def test_a_measurement_sitting_exactly_on_its_bar_passes(
        self, thresholds: AcceptanceThresholds, name: str
    ) -> None:
        """Both directions are inclusive, so meeting a bar exactly clears it."""
        report = build_acceptance_report([_make_student()], thresholds)
        check = report.verdicts[0].checks[0]
        assert check.name == name
        assert check.value == check.limit
        assert check.passed is True

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

    def test_two_students_measured_alike_both_stay_on_the_front(self) -> None:
        """Domination needs a strict win somewhere, so a tie knocks nobody off."""
        report = build_acceptance_report(
            [
                _make_student("twin-a", forces_mae=0.02, atoms_per_second=2.0e6),
                _make_student("twin-b", forces_mae=0.02, atoms_per_second=2.0e6),
            ]
        )
        assert report.pareto_front == ("twin-a", "twin-b")

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


class TestThroughputComparability:
    """A speed column ranks students only when one batch produced every number."""

    def test_students_timed_on_different_atom_counts_are_rejected(self) -> None:
        """The rate scales with the batch, so two batches produce no ranking."""
        with pytest.raises(ValueError, match="different batches"):
            build_acceptance_report(
                [
                    _make_student_on("small", num_atoms=1000),
                    _make_student_on("large", num_atoms=64000),
                ]
            )

    def test_the_same_atoms_split_into_different_graph_counts_are_rejected(
        self,
    ) -> None:
        """Graph count moves the rate at a fixed atom count, so it is checked too."""
        with pytest.raises(ValueError, match="different batches"):
            build_acceptance_report(
                [
                    _make_student_on("one-graph", num_graphs=1),
                    _make_student_on("many-graphs", num_graphs=64),
                ]
            )

    def test_a_lone_student_is_compared_against_nothing(self) -> None:
        """A family of one has no second workload to disagree with."""
        report = build_acceptance_report([_make_student_on("only", num_atoms=64000)])
        assert report.pareto_front == ("only",)

    def test_an_unmeasured_student_does_not_count_as_a_second_workload(self) -> None:
        """A student with no throughput at all leaves the measured ones comparable."""
        report = build_acceptance_report(
            [
                _make_student_on("measured", num_atoms=64000),
                StudentEvaluation(name="unmeasured", accuracy=_make_accuracy()),
            ]
        )
        assert report.pareto_front == ("measured",)

    def test_the_pareto_table_names_the_workload_every_rate_was_measured_on(
        self,
    ) -> None:
        """The speed column carries the batch behind it, so it cannot be misread."""
        rendered = _render(
            build_acceptance_report([_make_student_on("only", num_atoms=64000)])
        )
        assert "Atoms/graphs" in rendered
        assert "64,000 / 1" in rendered


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

    def test_a_mixed_family_gates_only_the_students_that_draft(self) -> None:
        """The bar skips the plain students of a sweep it was never aimed at."""
        report = build_acceptance_report(
            [
                _make_student("student-s"),
                _make_student("student-m", forces_mae=0.03),
                _make_student(
                    "drafter-xs", drafter=DrafterMetrics(acceptance_rate=0.8)
                ),
            ],
            AcceptanceThresholds(min_drafter_acceptance_rate=0.6, max_forces_mae=0.05),
        )
        assert [check.name for check in report.verdicts[0].checks] == ["forces_mae"]
        assert report.verdicts[2].checks[-1].name == "drafter_acceptance_rate"
        assert report.accepted
        assert report.scalars()["student-m/accepted"] == 1.0

    def test_a_drafter_below_the_bar_still_fails_in_a_mixed_family(self) -> None:
        """Scoping the bar to the drafters does not soften it for a drafter."""
        report = build_acceptance_report(
            [
                _make_student("student-s"),
                _make_student(
                    "drafter-xs", drafter=DrafterMetrics(acceptance_rate=0.4)
                ),
            ],
            AcceptanceThresholds(min_drafter_acceptance_rate=0.6),
        )
        assert report.verdicts[0].accepted
        assert not report.verdicts[1].accepted
        assert not report.accepted

    def test_a_drafter_bar_on_a_family_without_a_drafter_is_rejected(self) -> None:
        """A bar scoped away to nothing is a caller mistake, not a silent pass."""
        with pytest.raises(ValueError, match="no student of the family carries"):
            build_acceptance_report(
                [_make_student()],
                AcceptanceThresholds(min_drafter_acceptance_rate=0.6),
            )


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
        report = build_acceptance_report([_make_student(num_parameters=1234)])
        scalars = report.scalars()
        assert scalars["student/accepted"] == 1.0
        assert scalars["student/accuracy/forces_mae"] == 0.02
        assert scalars["student/stability/energy_drift_per_atom_per_ns"] == 0.001
        assert scalars["student/num_parameters"] == 1234.0
        assert all(isinstance(value, float) for value in scalars.values())
        assert "student/name" not in scalars

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


class TestMeasurementRoundTrip:
    """Rebuilding evaluations from the exports separate jobs wrote."""

    def test_a_fully_measured_student_survives_a_json_round_trip(self) -> None:
        """Every nested measurement comes back as the object it was exported from."""
        student = _make_student(
            "student-l",
            extensivity=_make_extensivity(),
            rdf=_make_rdf(pair=(11, 17)),
            drafter=DrafterMetrics(acceptance_rate=0.8, draft_steps=4),
            baseline_accuracy=_make_accuracy("scratch"),
            num_parameters=1234,
        )
        rebuilt = StudentEvaluation.from_dict(json.loads(json.dumps(student.to_dict())))
        assert rebuilt == student
        assert rebuilt.extensivity.repeats == (2, 1, 1)
        assert rebuilt.rdf.pair == (11, 17)

    def test_an_unmeasured_slot_comes_back_unmeasured(self) -> None:
        """Fields the export drops rebuild as ``None`` rather than as zeros."""
        student = StudentEvaluation(name="student", accuracy=_make_accuracy())
        rebuilt = StudentEvaluation.from_dict(student.to_dict())
        assert rebuilt == student
        assert rebuilt.throughput is None
        assert rebuilt.accuracy.stress_mae is None

    def test_a_student_entry_of_a_report_rebuilds_without_its_verdict(self) -> None:
        """Verdicts are formed from the bars of the report being built, not carried."""
        report = build_acceptance_report(
            [_make_student()], AcceptanceThresholds(max_forces_mae=0.05)
        )
        exported = report.to_dict()["students"][0]
        assert StudentEvaluation.from_dict(exported) == report.evaluations[0]

    def test_a_family_aggregated_from_exports_decides_the_same_way(self) -> None:
        """One job per student and one report at the end reaches the live verdicts."""
        thresholds = AcceptanceThresholds(
            max_forces_mae=0.05, min_atoms_per_second=1.0e6
        )
        family = [
            _make_student("student-s"),
            _make_student("student-m", forces_mae=0.2),
        ]
        rebuilt = [
            StudentEvaluation.from_dict(json.loads(json.dumps(student.to_dict())))
            for student in family
        ]
        assert (
            build_acceptance_report(rebuilt, thresholds).to_dict()
            == build_acceptance_report(family, thresholds).to_dict()
        )

    def test_a_key_the_measurement_does_not_declare_is_rejected(self) -> None:
        """An export written by another version fails where it is read."""
        exported = _make_accuracy().to_dict() | {"energy_mape": 0.1}
        with pytest.raises(ValueError, match="cannot be rebuilt from a mapping"):
            AccuracyMetrics.from_dict(exported)

    def test_a_missing_required_field_is_rejected(self) -> None:
        """A truncated export cannot be rebuilt into a partly-defaulted object."""
        exported = _make_accuracy().to_dict()
        del exported["num_atoms"]
        with pytest.raises(ValueError, match="missing the required"):
            AccuracyMetrics.from_dict(exported)


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
