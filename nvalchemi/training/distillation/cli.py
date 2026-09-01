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
"""Click interface for authoring, reviewing, and running distillation recipes.

The group registers on the training entry point as ``nvalchemi-training
distill`` — and on the ``nvalchemi-distill`` alias — beside the ``train`` and
``finetune`` groups it mirrors. A recipe is one JSON file validated by
:class:`DistillationJobSpec`: where the teacher and the student come from, what
data they see, the strategy bundle, the on-policy segment loop when there is
one, and the acceptance bars an evaluation is read against.

The student tiers the scaffold offers are size templates and nothing more —
``small``, ``base``, and ``large`` name a width and a depth for whatever
architecture the recipe points ``student.spec`` at, because a distillation
recipe is about the size of the student rather than its family.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

import click
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nvalchemi._serialization import _import_callable
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import load_checkpoint
from nvalchemi.training._spec import create_model_spec
from nvalchemi.training.cli import (
    DatasetSpec,
    OutputSpec,
    RuntimeHookSpec,
    SourceSpec,
    ValidationSpec,
    _attach_validation_config,
    _build_checked_hook,
    _build_dataloader,
    _build_supported_source_model,
    _path_exists,
    _primary_strategy_device,
    _write_or_print,
    console,
)
from nvalchemi.training.distillation.config import (
    OnPolicyConfig,
    _dataset_from_spec_dict,
)
from nvalchemi.training.distillation.evaluation import (
    AcceptanceThresholds,
    StudentEvaluation,
    build_acceptance_report,
    evaluate_accuracy,
)
from nvalchemi.training.distillation.evaluation.accuracy import AccuracyQuantity
from nvalchemi.training.distillation.scoring import _SIGNAL_SPECS
from nvalchemi.training.distillation.strategy import DistillationStrategy
from nvalchemi.training.losses.composition import (
    ComposedLossFunction,
    loss_component_to_spec,
)
from nvalchemi.training.losses.terms import EnergyMSELoss, ForceMSELoss
from nvalchemi.training.optimizers import OptimizerConfig

DistillationMode: TypeAlias = Literal["offline", "on-policy"]
StudentTier: TypeAlias = Literal["small", "base", "large"]

_STUDENT_TIERS: dict[str, dict[str, int]] = {
    "small": {"hidden_dim": 64, "num_layers": 2, "num_radial": 8},
    "base": {"hidden_dim": 128, "num_layers": 3, "num_radial": 8},
    "large": {"hidden_dim": 256, "num_layers": 4, "num_radial": 12},
}
"""Size templates a scaffold writes into the student spec, by tier name."""

_TIERS: tuple[StudentTier, ...] = ("small", "base", "large")

_DISTILL_EPILOG = (
    "A recipe is one JSON file: teacher, student, data, strategy, and — for "
    "on-policy runs — the segment loop. Author it with `distill init`, review "
    "it with `distill spec report`, start it with `distill spec run`, and gate "
    "the result with `distill evaluate`.\n\n"
    "Examples:\n\n"
    "Scaffold an offline recipe against a teacher-labeled store:\n\n"
    "  nvalchemi-training distill init --tier small --teacher-model mace --teacher-id small-0b --dataset data/labeled.zarr --output-dir runs/distill --out recipe.json\n\n"
    "Review and then run it:\n\n"
    "  nvalchemi-training distill spec report recipe.json\n\n"
    "  nvalchemi-training distill spec run recipe.json\n\n"
    "Score the trained student against a holdout:\n\n"
    "  nvalchemi-training distill evaluate recipe.json --student-checkpoint runs/distill/checkpoints\n"
)


class EvaluationSpec(BaseModel):
    """Holdout set and acceptance bars a recipe is gated on.

    ``EvaluationSpec`` is the optional ``evaluation`` member of
    :class:`DistillationJobSpec`, read by ``distill evaluate`` rather than by
    the run itself: a recipe therefore carries the bars it was meant to clear,
    and gating a trained student is one command against the same file.

    Examples
    --------
    ::

        EvaluationSpec(
            holdout_path="data/holdout.zarr",
            targets="teacher",
            thresholds={"max_forces_mae": 0.05},
        )
    """

    model_config = ConfigDict(extra="forbid")

    holdout_path: Annotated[
        str,
        Field(description="Held-out dataset the student is scored over."),
    ]
    targets: Annotated[
        Literal["reference", "teacher"],
        Field(
            default="teacher",
            description=(
                "Whether errors are measured against the holdout's own labels "
                "or against the teacher's."
            ),
        ),
    ] = "teacher"
    quantities: list[AccuracyQuantity] = Field(
        default_factory=lambda: ["energy", "forces"],
        description="Quantities the accuracy evaluation compares.",
    )
    batch_size: Annotated[
        int | None,
        Field(default=None, ge=1, description="Batch size of the holdout loader."),
    ] = None
    thresholds: AcceptanceThresholds = Field(
        default_factory=AcceptanceThresholds,
        description="Acceptance bars the verdict is formed against.",
    )


class StudentSpec(BaseModel):
    """Where the student comes from, and at what size.

    ``StudentSpec`` is the ``student`` member of :class:`DistillationJobSpec`.
    A student is normally constructed rather than loaded, so the common form is
    ``spec``: a ``{"cls_path": ..., "kwargs": {...}}`` reference naming the
    constructor and the arguments it is called with, the same shape the
    segment loop names its propagator by. ``tier`` records which size template
    those arguments came from, purely so a report and a sweep can say which
    tier a run belongs to; it selects a size, never an architecture.
    ``source`` loads a student from a checkpoint instead, for a run that
    continues from existing weights.

    Examples
    --------
    ::

        StudentSpec(
            tier="small",
            spec={"cls_path": "my_package.MyMLIP", "kwargs": {"hidden_dim": 64}},
        )
    """

    model_config = ConfigDict(extra="forbid")

    tier: Annotated[
        StudentTier | None,
        Field(description="Size template the student's arguments came from."),
    ] = None
    spec: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Constructor reference building the student, as cls_path plus kwargs."
            )
        ),
    ] = None
    source: Annotated[
        SourceSpec | None,
        Field(description="Checkpoint or supported wrapper to load the student from."),
    ] = None
    hooks: list[RuntimeHookSpec] = Field(
        default_factory=list,
        description=(
            "Runtime hooks attached to the training strategy, serialized as "
            "BaseSpec JSON objects. Attached at execution time rather than "
            "stored in the strategy bundle."
        ),
    )

    @model_validator(mode="after")
    def _validate_student_source(self) -> Self:
        """Require exactly one way to obtain the student, named the way it builds."""
        if (self.spec is None) == (self.source is None):
            raise ValueError(
                "student needs exactly one of spec or source: spec constructs a "
                "fresh student, source loads one from a checkpoint."
            )
        if self.spec is not None and "cls_path" not in self.spec:
            raise ValueError(
                "student.spec names the constructor by cls_path, with its "
                "arguments under kwargs."
            )
        return self


class DistillationJobSpec(BaseModel):
    """Top-level envelope describing one distillation recipe.

    The file ``distill spec report`` reads and ``distill spec run`` executes.
    ``mode`` selects offline distillation over a teacher-labeled store or the
    on-policy segment loop; ``teacher`` and ``student`` say where the two models
    come from; ``dataset`` names the training store — the labeled dataset
    offline, the anchor on-policy; ``strategy`` is the JSON-ready
    :meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
    bundle carrying optimizers, loss, devices, and duration; ``on_policy`` is
    the segment-loop recipe an on-policy run needs; and ``evaluation`` records
    the bars ``distill evaluate`` gates on.

    Examples
    --------
    A minimal offline recipe:

    .. code-block:: json

        {
          "mode": "offline",
          "teacher": {"model": "mace", "model_id": "small-0b"},
          "student": {"tier": "small", "spec": {"cls_path": "my_package.MyMLIP"}},
          "dataset": {"path": "data/labeled.zarr"},
          "output": {"run_dir": "runs/distill"},
          "strategy": {"...": "DistillationStrategy.to_spec_dict()"}
        }

    Notes
    -----
    Validation is pre-flight in the strict sense: the strategy bundle is
    deserialized with the same helpers the runtime uses, and an ``on_policy``
    recipe is checked for the keys the segment loop reads, so a misconfigured
    recipe fails at ``spec report`` rather than after a teacher has been loaded
    onto a GPU. What it cannot check without building models — that the loss's
    teacher targets are signals the teacher can produce, say — the strategy's
    own constructor checks at ``spec run``, and the CLI reports it as a clean
    error rather than a traceback.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(description="Human-readable recipe name.")] = (
        "distillation-job"
    )
    mode: Annotated[
        DistillationMode,
        Field(description="Offline distillation, or the on-policy segment loop."),
    ]
    teacher: Annotated[SourceSpec, Field(description="Where the teacher comes from.")]
    student: Annotated[StudentSpec, Field(description="Where the student comes from.")]
    dataset: Annotated[
        DatasetSpec,
        Field(description="Training store: the labeled dataset, or the anchor."),
    ]
    output: Annotated[OutputSpec, Field(description="Output path intent.")]
    validation: Annotated[
        ValidationSpec | None,
        Field(description="Optional validation cadence for CLI execution."),
    ] = None
    on_policy: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Segment-loop recipe produced by OnPolicyConfig.to_spec_dict(). "
                "Required by, and only read in, on-policy mode."
            )
        ),
    ] = None
    evaluation: Annotated[
        EvaluationSpec | None,
        Field(description="Holdout and acceptance bars for `distill evaluate`."),
    ] = None
    strategy: Annotated[
        dict[str, Any],
        Field(
            description=(
                "JSON-ready bundle produced by DistillationStrategy.to_spec_dict()."
            )
        ),
    ]
    notes: Annotated[
        str | None,
        Field(description="Optional notes rendered in the report."),
    ] = None

    @model_validator(mode="after")
    def _validate_mode(self) -> Self:
        """Require the segment-loop recipe exactly when the mode asks for one."""
        if self.mode == "on-policy" and self.on_policy is None:
            raise ValueError(
                "on-policy recipes need an on_policy block: the segment loop "
                "generates its own batches and has no dataloader to fall back "
                "on."
            )
        if self.mode == "offline" and self.on_policy is not None:
            raise ValueError(
                "offline recipes train on the dataset they name, so an "
                "on_policy block would never be read; set mode='on-policy' to "
                "use it."
            )
        return self

    @model_validator(mode="after")
    def _validate_strategy(self) -> Self:
        """Deserialize the strategy bundle with the runtime's own helpers."""
        missing = [
            key
            for key in ("optimizer_configs", "devices", "loss_fn_spec")
            if key not in self.strategy
        ]
        if missing:
            raise ValueError(
                f"strategy is missing required DistillationStrategy spec key(s) "
                f"{missing}."
            )
        num_epochs = self.strategy.get("num_epochs")
        num_steps = self.strategy.get("num_steps")
        if (num_epochs is None) == (num_steps is None):
            raise ValueError(
                "strategy must set exactly one of num_epochs or num_steps."
            )
        if self.mode == "on-policy" and num_steps is None:
            raise ValueError(
                "on-policy distillation is sized in optimizer steps, because "
                "every segment builds its own loader; set strategy.num_steps."
            )
        optimizers = strategy_spec._optimizer_configs_from_spec(
            self.strategy["optimizer_configs"]
        )
        if "teacher" in optimizers:
            raise ValueError(
                "the teacher is frozen by omission, so strategy."
                "optimizer_configs must not configure it."
            )
        if "student" not in optimizers:
            raise ValueError(
                "strategy.optimizer_configs must configure the student; got "
                f"{sorted(optimizers)!r}."
            )
        strategy_spec._devices_from_spec(self.strategy["devices"])
        strategy_spec._loss_fn_from_spec(self.strategy["loss_fn_spec"])
        strategy_spec._training_fn_from_spec(self.strategy, None)
        if self.validation is not None and self.dataset.validation_path is None:
            raise ValueError(
                "validation cadence requires dataset.validation_path to be set."
            )
        return self

    @model_validator(mode="after")
    def _validate_on_policy_recipe(self) -> Self:
        """Check the segment-loop recipe for the keys the loop reads."""
        if self.on_policy is None:
            return self
        required = ("dynamics", "teacher_scorer", "replay_ratio", "steps_per_segment")
        missing = [key for key in required if key not in self.on_policy]
        if missing:
            raise ValueError(f"on_policy is missing required key(s) {missing}.")
        if "cls_path" not in self.on_policy["dynamics"]:
            raise ValueError(
                "on_policy.dynamics names the propagator by cls_path, with its "
                "constructor arguments under kwargs; the student is bound at "
                "build time and must not be named."
            )
        signals = self.on_policy["teacher_scorer"].get("signals")
        unsupported = sorted(set(signals or ()) - set(_SIGNAL_SPECS))
        if not signals or unsupported:
            raise ValueError(
                "on_policy.teacher_scorer.signals must name teacher signals "
                f"from {sorted(_SIGNAL_SPECS)!r}; got {signals!r}."
            )
        return self

    @classmethod
    def template(
        cls,
        *,
        mode: DistillationMode,
        tier: StudentTier,
        dataset: str,
        output_dir: str,
        teacher_model: str,
        teacher_id: str | None = None,
        teacher_checkpoint: str | None = None,
        student_cls_path: str = "my_package.my_module.MyStudentModel",
        lr: float = 1e-4,
        num_steps: int = 1000,
        device: str = "cuda",
        seed_dataset: str | None = None,
        validation_path: str | None = None,
        holdout_path: str | None = None,
    ) -> Self:
        """Build a validated scaffold for a distillation recipe.

        Parameters
        ----------
        mode : {"offline", "on-policy"}
            Which loop the recipe describes.
        tier : {"small", "base", "large"}
            Size template written into the student's constructor arguments.
        dataset : str
            Training store: the teacher-labeled dataset offline, the anchor
            on-policy.
        output_dir : str
            Run directory; checkpoints are scaffolded beneath it.
        teacher_model : str
            Teacher source family, as in the training CLI.
        teacher_id : str | None, optional
            Teacher model id for a supported wrapper. Default ``None``.
        teacher_checkpoint : str | None, optional
            Teacher checkpoint path. Default ``None``.
        student_cls_path : str, optional
            Dotted path of the student constructor the tier sizes.
        lr : float, optional
            Student learning rate. Default ``1e-4``.
        num_steps : int, optional
            Optimizer steps to run. Default ``1000``.
        device : str, optional
            Strategy device string. Default ``"cuda"``.
        seed_dataset : str | None, optional
            Store the on-policy loop seeds its trajectories from. Default
            ``None``, which reuses *dataset*.
        validation_path : str | None, optional
            Validation store. Default ``None``.
        holdout_path : str | None, optional
            Holdout store recorded in the evaluation section. Default ``None``.

        Returns
        -------
        DistillationJobSpec
            Validated scaffold ready to be edited and reported on.
        """
        teacher: dict[str, Any] = {"model": teacher_model}
        if teacher_id is not None:
            teacher["model_id"] = teacher_id
        if teacher_checkpoint is not None:
            teacher["checkpoint_path"] = teacher_checkpoint
        dataset_payload: dict[str, Any] = {"path": dataset, "format": "alchemi-zarr"}
        if validation_path is not None:
            dataset_payload["validation_path"] = validation_path
        return cls(
            name=f"{tier}-student-{mode}-distillation",
            mode=mode,
            teacher=teacher,
            student={
                "tier": tier,
                "spec": {
                    "cls_path": student_cls_path,
                    "kwargs": dict(_STUDENT_TIERS[tier]),
                },
            },
            dataset=dataset_payload,
            output={
                "run_dir": output_dir,
                "checkpoint_dir": str(Path(output_dir) / "checkpoints"),
            },
            validation=(None if validation_path is None else {"every_n_epochs": 1}),
            on_policy=(
                None
                if mode == "offline"
                else _on_policy_template(seed_dataset or dataset, device)
            ),
            evaluation=(
                None
                if holdout_path is None
                else {"holdout_path": holdout_path, "targets": "teacher"}
            ),
            strategy=_default_distillation_strategy_spec(
                lr=lr, num_steps=num_steps, device=device
            ),
        )


def _on_policy_template(seed_dataset: str, device: str) -> dict[str, Any]:
    """Return a segment-loop recipe scaffold seeded from *seed_dataset*."""
    return {
        "dynamics": {
            "cls_path": "nvalchemi.dynamics.integrators.nvt_langevin.NVTLangevin",
            "kwargs": {
                "dt": 0.5,
                "temperature": 300.0,
                "friction": 0.01,
                "random_seed": 42,
            },
        },
        "teacher_scorer": {
            "teacher": "teacher",
            "signals": ["energy", "forces"],
            "cast_to": None,
        },
        "seed_dataset": {"path": seed_dataset, "device": device},
        "replay_ratio": 0.25,
        "steps_per_segment": 32,
        "batch_size": 8,
        "segment_steps": 50,
        "label_frequency": 10,
        "replay_capacity": 8192,
        "replay_eviction": "fifo",
        "replay_device": None,
        "seed": 0,
        "weight_sync_frequency": 1,
    }


def _default_distillation_strategy_spec(
    *, lr: float, num_steps: int, device: str
) -> dict[str, Any]:
    """Return a strategy bundle matching energies and forces against the teacher."""
    loss_fn = ComposedLossFunction(
        [
            EnergyMSELoss(target_key="teacher_energy"),
            ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
        ],
        weights=[1.0, 10.0],
        normalize_weights=False,
    )
    loss_fn_spec = create_model_spec(
        type(loss_fn),
        components=[loss_component_to_spec(comp) for comp in loss_fn.components],
        weights=list(loss_fn._weights),
        normalize_weights=loss_fn.normalize_weights,
        dtype_policy=loss_fn.dtype_policy,
    )
    optimizer_config = OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": lr, "weight_decay": 1e-6},
    )
    return {
        "optimizer_configs": {"student": [optimizer_config.to_spec().model_dump()]},
        "num_epochs": None,
        "num_steps": num_steps,
        "epoch_step_modifier": 1.0,
        "devices": [device],
        "loss_fn_spec": loss_fn_spec.model_dump(),
        "model_specs": {},
        "single_model_input": False,
        "training_fn": (
            "nvalchemi.training.distillation.strategy.default_distillation_fn"
        ),
        "teacher_signals": None,
        "label_missing": True,
    }


def _load_recipe(path: Path) -> DistillationJobSpec:
    """Load and validate a distillation recipe from JSON."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Could not parse {path}: {exc}") from exc
    try:
        return DistillationJobSpec.model_validate(raw)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc


def _recipe_paths(job: DistillationJobSpec) -> list[tuple[str, str]]:
    """Return the local paths a recipe references, keyed by field."""
    checks: list[tuple[str, str | None]] = [
        ("dataset.path", job.dataset.path),
        ("dataset.validation_path", job.dataset.validation_path),
        ("teacher.checkpoint_path", job.teacher.checkpoint_path),
    ]
    if job.student.source is not None:
        checks.append(
            ("student.source.checkpoint_path", job.student.source.checkpoint_path)
        )
    if job.on_policy is not None and job.on_policy.get("seed_dataset"):
        checks.append(
            ("on_policy.seed_dataset.path", job.on_policy["seed_dataset"]["path"])
        )
    if job.evaluation is not None:
        checks.append(("evaluation.holdout_path", job.evaluation.holdout_path))
    return [(field, value) for field, value in checks if value is not None]


def _derived_teacher_signals(job: DistillationJobSpec) -> list[str]:
    """Return the teacher signals the recipe's loss targets imply."""
    fields = {spec.field: name for name, spec in _SIGNAL_SPECS.items()}
    loss_fn = strategy_spec._loss_fn_from_spec(job.strategy["loss_fn_spec"])
    signals = {
        fields[key]
        for component in loss_fn.components
        if (key := getattr(component, "target_key", None)) in fields
    }
    return sorted(signals)


def _mixture_rows(job: DistillationJobSpec) -> list[tuple[str, str]]:
    """Return the composition of one training batch, as label/value rows."""
    if job.on_policy is None:
        return [("mixture", "every sample from the labeled dataset (offline)")]
    ratio = float(job.on_policy["replay_ratio"])
    batch_size = int(job.on_policy.get("batch_size", 8))
    replay = int(round(ratio * batch_size))
    return [
        ("replay_ratio", f"{ratio:g}"),
        ("batch composition", f"{batch_size - replay} anchor + {replay} generated"),
        ("segment", f"{job.on_policy['segment_steps']} generated steps"),
        ("label cadence", f"every {job.on_policy['label_frequency']} steps"),
        ("training per segment", f"{job.on_policy['steps_per_segment']} batches"),
    ]


def _intent_table(job: DistillationJobSpec) -> Table:
    """Build the Rich table summarizing recipe intent."""
    table = Table(title="Distillation intent", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Area", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("recipe", job.name)
    table.add_row("mode", job.mode)
    table.add_row(
        "teacher",
        f"{job.teacher.model} ({job.teacher.model_id or job.teacher.checkpoint_path})",
    )
    student = job.student
    table.add_row("student tier", student.tier or "not specified")
    table.add_row(
        "student",
        (student.spec or {}).get("cls_path", "")
        if student.spec is not None
        else f"{student.source.model} ({student.source.checkpoint_path})",
    )
    table.add_row("teacher signals", ", ".join(_derived_teacher_signals(job)))
    table.add_row("dataset", f"{job.dataset.path} ({job.dataset.format})")
    table.add_row("validation", job.dataset.validation_path or "none")
    table.add_row("run dir", job.output.run_dir)
    table.add_row("num_steps", str(job.strategy.get("num_steps")))
    table.add_row("num_epochs", str(job.strategy.get("num_epochs")))
    table.add_row("devices", ", ".join(map(str, job.strategy.get("devices", []))))
    for label, value in _mixture_rows(job):
        table.add_row(label, value)
    return table


def _threshold_table(job: DistillationJobSpec) -> Table | None:
    """Build the acceptance-bar table, or ``None`` when the recipe sets none."""
    if job.evaluation is None:
        return None
    bars = job.evaluation.thresholds.model_dump(exclude_none=True)
    table = Table(title="Acceptance bars", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Bar", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("holdout", job.evaluation.holdout_path)
    table.add_row("targets", job.evaluation.targets)
    table.add_row("quantities", ", ".join(job.evaluation.quantities))
    for name, value in sorted(bars.items()):
        table.add_row(name, str(value))
    return table


def _warning_table(job: DistillationJobSpec) -> Table:
    """Build the table of pre-flight warnings a recipe earns."""
    table = Table(title="Pre-flight", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    missing = [
        (field, value) for field, value in _recipe_paths(job) if not _path_exists(value)
    ]
    for field, value in missing:
        table.add_row(field, f"[yellow]missing on disk:[/] {value}")
    if job.output.checkpoint_dir and not job.student.hooks:
        table.add_row(
            "output.checkpoint_dir",
            "[yellow]set without a CheckpointHook in student.hooks; "
            "nothing will be written[/]",
        )
    if job.mode == "on-policy" and float(job.on_policy["replay_ratio"]) == 1.0:
        table.add_row(
            "on_policy.replay_ratio",
            "[yellow]1.0 trains on generated frames only; the run has no "
            "anchor to stay near the reference distribution[/]",
        )
    if job.evaluation is None:
        table.add_row(
            "evaluation",
            "no acceptance bars recorded; `distill evaluate` "
            "will report numbers without a verdict",
        )
    if not table.rows:
        table.add_row("all", "[green]no issues found[/]")
    return table


def _render_report(job: DistillationJobSpec) -> None:
    """Render the Rich report card for a distillation recipe."""
    console.rule(f"[bold]Distillation report: {job.name}")
    console.print(_intent_table(job))
    console.print(_warning_table(job))
    thresholds = _threshold_table(job)
    if thresholds is not None:
        console.print(thresholds)
    if job.notes:
        console.print(Panel(Text(job.notes, overflow="fold"), title="Notes"))


def _build_role_model(
    source: SourceSpec, *, device: Any, role: str, map_location: str | None
) -> Any:
    """Build the model one role of the recipe names."""
    if source.model in {"mace", "aimnet2"}:
        return _build_supported_source_model(source, device=device)
    if source.model != "native-checkpoint":
        raise click.ClickException(
            f"{role} source model {source.model!r} cannot be built by the CLI; "
            "use a supported wrapper, a native checkpoint, or — for the "
            "student — a constructor spec."
        )
    if source.checkpoint_path is None:
        raise click.ClickException(f"{role} native-checkpoint needs checkpoint_path.")
    name = (source.model_extra or {}).get("model_name", role)
    try:
        loaded = load_checkpoint(
            source.checkpoint_path,
            checkpoint_index=source.checkpoint_index,
            map_location=map_location or str(device),
            model_names={name},
        )
    except KeyError as exc:
        raise click.ClickException(
            f"{role} checkpoint {source.checkpoint_path!r} does not hold a model "
            f"named {name!r}: {exc}. Set source.model_name to one it does hold."
        ) from exc
    models = loaded["models"] if isinstance(loaded, Mapping) else loaded.models
    entry = models[name]
    return entry["model"] if isinstance(entry, Mapping) else entry[0]


def _build_student(
    job: DistillationJobSpec, *, device: Any, map_location: str | None
) -> Any:
    """Build the student a recipe constructs or loads."""
    if job.student.source is not None:
        return _build_role_model(
            job.student.source, device=device, role="student", map_location=map_location
        )
    spec = job.student.spec
    try:
        student = _import_callable(spec["cls_path"])(**dict(spec.get("kwargs", {})))
    except Exception as exc:
        raise click.ClickException(
            f"student.spec did not build a model from {spec['cls_path']!r}: {exc}"
        ) from exc
    return student.to(device)


def _build_strategy(
    job: DistillationJobSpec, *, map_location: str | None
) -> DistillationStrategy:
    """Build the strategy a recipe declares, reporting its own errors cleanly."""
    device = _primary_strategy_device(job)
    teacher = _build_role_model(
        job.teacher, device=device, role="teacher", map_location=map_location
    )
    student = _build_student(job, device=device, map_location=map_location)
    hooks = _build_recipe_hooks(job)
    on_policy = None
    reference_dataset = None
    if job.on_policy is not None:
        on_policy = OnPolicyConfig.from_spec_dict(
            job.on_policy, student=student, teacher=teacher
        )
        reference_dataset = _dataset_from_spec_dict(
            {"path": job.dataset.path, "device": str(device)}
        )
    try:
        return DistillationStrategy.from_spec_dict(
            dict(job.strategy),
            models={"student": student, "teacher": teacher},
            hooks=hooks,
            on_policy=on_policy,
            reference_dataset=reference_dataset,
        )
    except (ValueError, TypeError) as exc:
        raise click.ClickException(f"strategy could not be built: {exc}") from exc


def _build_recipe_hooks(job: DistillationJobSpec) -> list[Any]:
    """Build the runtime hooks a recipe declares, one per requested stage."""
    hooks: list[Any] = []
    for hook_spec in job.student.hooks:
        stages = hook_spec.stage_values()
        if not stages:
            hooks.append(_build_checked_hook(hook_spec.spec))
            continue
        for stage in stages:
            hook = _build_checked_hook(hook_spec.spec)
            hook.stage = stage
            hooks.append(hook)
    return hooks


def _run_recipe(job: DistillationJobSpec, *, map_location: str | None) -> None:
    """Build the runtime components of a recipe and run it."""
    strategy = _build_strategy(job, map_location=map_location)
    with ExitStack() as stack:
        device = _primary_strategy_device(job)
        _attach_validation_config(
            strategy,
            job,
            stack,
            device=device,
            batch_size=job.dataset.batch_size,
            prefetch_factor=2,
            num_streams=4,
            use_streams=True,
            pin_memory=False,
            validation_path=None,
            validation_every_epochs=None,
            validation_every_steps=None,
        )
        if job.mode == "on-policy":
            strategy.run()
            return
        dataloader = _build_dataloader(
            job,
            stack,
            device=device,
            batch_size=job.dataset.batch_size,
            shuffle=True,
            drop_last=False,
            prefetch_factor=2,
            num_streams=4,
            use_streams=True,
            pin_memory=False,
        )
        strategy.run(dataloader)


@click.group(name="distill", epilog=_DISTILL_EPILOG)
def distill() -> None:
    """Author, review, run, and gate distillation recipes."""


@distill.group(name="spec")
def distill_spec() -> None:
    """Validate, report on, and execute saved distillation recipes."""


@distill.command("init")
@click.option(
    "--mode",
    type=click.Choice(["offline", "on-policy"]),
    default="offline",
    show_default=True,
    help="Which distillation loop the recipe describes.",
)
@click.option(
    "--tier",
    type=click.Choice(_TIERS),
    default="small",
    show_default=True,
    help="Student size template: width and depth only, never an architecture.",
)
@click.option("--dataset", required=True, help="Teacher-labeled store, or the anchor.")
@click.option("--output-dir", required=True, help="Run output directory.")
@click.option(
    "--teacher-model",
    default="mace",
    show_default=True,
    help="Teacher source family.",
)
@click.option("--teacher-id", default=None, help="Teacher model id.")
@click.option("--teacher-checkpoint", default=None, help="Teacher checkpoint path.")
@click.option(
    "--student-cls-path",
    default="my_package.my_module.MyStudentModel",
    show_default=True,
    help="Dotted path of the student constructor the tier sizes.",
)
@click.option("--lr", type=float, default=1e-4, show_default=True, help="Student LR.")
@click.option(
    "--num-steps", type=int, default=1000, show_default=True, help="Optimizer steps."
)
@click.option("--device", default="cuda", show_default=True, help="Strategy device.")
@click.option("--seed-dataset", default=None, help="Store the segment loop seeds from.")
@click.option(
    "--validation-dataset", "validation_path", default=None, help="Validation store."
)
@click.option(
    "--holdout-dataset", "holdout_path", default=None, help="Acceptance holdout store."
)
@click.option(
    "--out",
    "output",
    type=click.Path(path_type=Path),
    help="Write the recipe JSON to this file.",
)
def init_recipe(
    mode: DistillationMode,
    tier: StudentTier,
    dataset: str,
    output_dir: str,
    teacher_model: str,
    teacher_id: str | None,
    teacher_checkpoint: str | None,
    student_cls_path: str,
    lr: float,
    num_steps: int,
    device: str,
    seed_dataset: str | None,
    validation_path: str | None,
    holdout_path: str | None,
    output: Path | None,
) -> None:
    """Create a distillation recipe scaffold at the requested student tier."""
    try:
        payload = DistillationJobSpec.template(
            mode=mode,
            tier=tier,
            dataset=dataset,
            output_dir=output_dir,
            teacher_model=teacher_model,
            teacher_id=teacher_id,
            teacher_checkpoint=teacher_checkpoint,
            student_cls_path=student_cls_path,
            lr=lr,
            num_steps=num_steps,
            device=device,
            seed_dataset=seed_dataset,
            validation_path=validation_path,
            holdout_path=holdout_path,
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    _write_or_print(payload, output)
    if output is not None:
        console.print(f"[green]Created {mode} distillation recipe[/] {output}")


@distill.command("schema")
@click.option(
    "--out",
    "output",
    type=click.Path(path_type=Path),
    help="Write the schema JSON to this file.",
)
def dump_schema(output: Path | None) -> None:
    """Dump the distillation recipe JSON schema."""
    _write_or_print(DistillationJobSpec.model_json_schema(), output)


@distill_spec.command("report")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "show_json", is_flag=True, help="Print the normalized recipe.")
def report_recipe(path: Path, show_json: bool) -> None:
    """Validate a recipe and render what it intends to do."""
    job = _load_recipe(path)
    _render_report(job)
    if show_json:
        _write_or_print(job, None)


@distill_spec.command("run")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--map-location", default=None, help="Checkpoint map_location.")
@click.option(
    "--report/--no-report",
    "show_report",
    default=True,
    show_default=True,
    help="Render the report before execution.",
)
def run_recipe(path: Path, map_location: str | None, show_report: bool) -> None:
    """Build the models, data, and strategy of a recipe, then run it."""
    job = _load_recipe(path)
    if show_report:
        _render_report(job)
    _run_recipe(job, map_location=map_location)


@distill.command("evaluate")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--student-checkpoint",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Native checkpoint directory holding the trained student.",
)
@click.option("--checkpoint-index", type=int, default=-1, show_default=True)
@click.option(
    "--holdout", "holdout_path", default=None, help="Override the holdout store."
)
@click.option("--batch-size", type=int, default=None, help="Holdout loader batch size.")
@click.option("--map-location", default=None, help="Checkpoint map_location.")
@click.option(
    "--json-out",
    "json_out",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the acceptance report as JSON to this file.",
)
def evaluate_student(
    path: Path,
    student_checkpoint: Path,
    checkpoint_index: int,
    holdout_path: str | None,
    batch_size: int | None,
    map_location: str | None,
    json_out: Path | None,
) -> None:
    """Score a trained student against the recipe's holdout and acceptance bars.

    Exits non-zero when a bar is not cleared, so a sweep can gate on the
    command rather than on reading its output.
    """
    job = _load_recipe(path)
    if job.evaluation is None and holdout_path is None:
        raise click.ClickException(
            "the recipe records no evaluation section, so there is no holdout "
            "to score against; add one, or pass --holdout."
        )
    device = _primary_strategy_device(job)
    student = _build_role_model(
        SourceSpec(
            model="native-checkpoint",
            checkpoint_path=str(student_checkpoint),
            checkpoint_index=checkpoint_index,
        ),
        device=device,
        role="student",
        map_location=map_location,
    )
    evaluation = job.evaluation
    targets = "teacher" if evaluation is None else evaluation.targets
    scorer = None
    if targets == "teacher":
        scorer = _build_role_model(
            job.teacher, device=device, role="teacher", map_location=map_location
        )
    resolved_holdout = holdout_path or evaluation.holdout_path
    with ExitStack() as stack:
        holdout = _build_dataloader(
            job,
            stack,
            device=device,
            batch_size=batch_size
            or (None if evaluation is None else evaluation.batch_size),
            shuffle=False,
            drop_last=False,
            prefetch_factor=2,
            num_streams=4,
            use_streams=True,
            pin_memory=False,
            paths=[resolved_holdout],
        )
        metrics = evaluate_accuracy(
            student,
            holdout,
            targets=targets,
            quantities=None if evaluation is None else list(evaluation.quantities),
            scorer=scorer,
            device=device,
            name=job.name,
        )
    report = build_acceptance_report(
        [
            StudentEvaluation(
                name=job.student.tier or job.name,
                accuracy=metrics,
                num_parameters=sum(
                    parameter.numel() for parameter in student.parameters()
                ),
            )
        ],
        None if evaluation is None else evaluation.thresholds,
    )
    console.print(report)
    if json_out is not None:
        _write_or_print(report.to_dict(), json_out)
    if not report.accepted:
        raise click.exceptions.Exit(1)
