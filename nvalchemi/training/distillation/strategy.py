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
"""Knowledge-distillation strategy built on :class:`TrainingStrategy`."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import Field, PrivateAttr, model_validator

from nvalchemi._typing import ModelOutputs
from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks._context import DynamicsContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distillation._labels import _attach_teacher_labels
from nvalchemi.training.distillation.config import OnPolicyConfig
from nvalchemi.training.distillation.hooks import TeacherLabelHook
from nvalchemi.training.distillation.replay import ReplayBuffer, build_mixed_loader
from nvalchemi.training.distillation.scoring import (
    _SIGNAL_SPECS,
    InProcessTeacherScorer,
)
from nvalchemi.training.losses.composition import loss_target_keys
from nvalchemi.training.runtime import (
    freeze_unconfigured_models,
    move_to_devices,
    train_configured_models,
)
from nvalchemi.training.strategy import TrainingStrategy

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

    from nvalchemi.data.batch import Batch
    from nvalchemi.hooks._context import TrainContext
    from nvalchemi.training.losses.composition import ComposedLossFunction

__all__ = ["DistillationStrategy", "default_distillation_fn"]

_REQUIRED_MODELS = frozenset({"student", "teacher"})
"""Model names every distillation strategy must be given."""

_TEACHER_FIELD_PREFIX = "teacher_"
"""Prefix marking a loss target that a teacher signal has to supply."""

_SIGNALS_BY_FIELD = {spec.field: name for name, spec in _SIGNAL_SPECS.items()}
"""Teacher signal name, keyed by the batch field the signal populates."""


def default_distillation_fn(
    models: Mapping[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Run the student forward pass and prefix output keys with ``predicted_``.

    The teacher is never called here: teacher knowledge reaches the loss as
    ``teacher_*`` batch fields, either written offline by
    :func:`~nvalchemi.training.distillation.label_dataset` or attached to the
    batch by :meth:`DistillationStrategy.attach_teacher_labels`.

    Parameters
    ----------
    models : Mapping[str, BaseModelMixin]
        Named models of the strategy; only ``"student"`` is read.
    batch : Batch
        Input batch of atomic graphs.

    Returns
    -------
    dict[str, torch.Tensor]
        Predictions keyed by ``predicted_<output_name>`` with ``None`` outputs
        omitted.
    """
    outputs: ModelOutputs = models["student"](batch)
    return {
        f"predicted_{key}": value for key, value in outputs.items() if value is not None
    }


def _derived_teacher_signals(loss_fn: ComposedLossFunction) -> frozenset[str]:
    """Return the teacher signals the loss composition's targets require."""
    signals: set[str] = set()
    for key in loss_target_keys(loss_fn):
        if not key.startswith(_TEACHER_FIELD_PREFIX):
            continue
        signal = _SIGNALS_BY_FIELD.get(key)
        if signal is None:
            raise ValueError(
                "Loss targets must name a supported teacher target from "
                f"{sorted(_SIGNALS_BY_FIELD)!r}; got {key!r}."
            )
        signals.add(signal)
    return frozenset(signals)


def _student_label_dtype(student: BaseModelMixin) -> torch.dtype | None:
    """Return the dtype teacher labels are cast to for *student*.

    The first floating-point parameter decides. A student that exposes no
    parameters at all gets ``None``, which leaves labels in the teacher's own
    dtype.
    """
    parameters = getattr(student, "parameters", None)
    if not callable(parameters):
        return None
    for parameter in parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return None


class _TeacherLabelHook:
    """Label the batch a forward pass is about to consume, training or validation."""

    frequency = 1
    stage = TrainingStage.BEFORE_FORWARD

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Attach the teacher fields the upcoming batch is missing."""
        strategy: DistillationStrategy = ctx.workflow
        if ctx.batch is not None and strategy.label_missing:
            strategy.attach_teacher_labels(ctx.batch)


class DistillationStrategy(TrainingStrategy):
    """Train a student against a frozen teacher's signals.

    ``DistillationStrategy`` is a :class:`~nvalchemi.training.TrainingStrategy`
    whose named models are ``"student"`` and ``"teacher"``. The teacher is
    frozen by omission — it must not appear in ``optimizer_configs``, which is
    what puts it in eval mode with gradients disabled for the duration of
    :meth:`run` — while the student, and any auxiliary model such as a
    projection head, must be configured with an optimizer.

    Teacher knowledge reaches the loss as ordinary batch fields. Every signal
    an :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`
    produces populates one ``teacher_*`` field, and a loss term consumes it by
    pointing its ``target_key`` there: ``EnergyMSELoss(target_key="teacher_energy")``,
    ``ForceMSELoss(target_key="teacher_forces")``,
    :class:`~nvalchemi.training.distillation.PerAtomEnergyMatchingLoss` for
    ``teacher_node_energies``. Mixing teacher targets with reference targets in
    one objective is therefore ordinary loss composition, and so is annealing
    between them with a
    :class:`~nvalchemi.training.losses.base.LossWeightSchedule`.

    Those targets also decide what the teacher is asked for.
    ``teacher_signals=None`` (the default) derives the signal set from the
    ``teacher_*`` targets the loss reads, so the two cannot drift apart; an
    explicit set must cover the derived one and may add more. The resolved set
    is checked against the teacher's declared outputs at construction, as is
    the model/optimizer contract above and — for the stock ``training_fn`` —
    every loss component's prediction key against the student's declared
    outputs, so a misconfigured run fails before it starts rather than on its
    first batch.

    In offline distillation the labels travel with the sample. The intended
    path is :func:`~nvalchemi.training.distillation.label_dataset`: score the
    dataset once, persist the teacher fields into a Zarr store, and train from
    that store with no teacher forward pass at all. A batch that arrives
    without the required fields is labeled on the fly instead — training and
    validation alike — which keeps short runs and interactive sessions working
    without a labeling pass. ``label_missing=False`` turns that off and leaves
    an unlabeled batch to surface as a missing loss target. Either way
    ``training_fn`` stays a plain student forward —
    :func:`default_distillation_fn` unless the caller supplies one — so the
    recipe survives :meth:`to_spec_dict` and the teacher never enters the
    student's autograd graph.

    Setting ``on_policy`` switches :meth:`run` to the segment loop instead:
    the student's own propagator generates frames, the teacher labels them,
    they accumulate in a replay buffer, and each segment trains on a mixture of
    that buffer and ``reference_dataset`` at the configured ``replay_ratio``.
    Because the propagator holds the very module the optimizer updates, every
    segment generates from a fresher policy than the last — which is what makes
    the data on-policy, and why the propagator's model is checked for object
    identity with ``models["student"]`` at construction.

    Raises
    ------
    ValueError
        If ``models`` is not a named mapping containing ``"student"`` and
        ``"teacher"``, if the teacher is given an optimizer config, if the
        student or an auxiliary model is not, if a loss component reads a
        prediction the student does not declare, if the loss reads a
        ``teacher_*`` target that maps to no known signal, if an explicit
        ``teacher_signals`` omits a signal the loss needs, if no teacher signal
        is requested at all, or if the teacher cannot produce a requested
        signal. In on-policy mode, additionally if the run is sized in epochs
        rather than steps, if the propagator holds a different module than the
        student, if ``replay_ratio`` is ``0``, or if a ratio below ``1`` is
        paired with no ``reference_dataset``.

    Examples
    --------
    Distill energies, forces, and the teacher's per-atom energy decomposition
    from a store written by :func:`label_dataset`:

    >>> import torch
    >>> from nvalchemi.training import EnergyMSELoss, ForceMSELoss, OptimizerConfig
    >>> from nvalchemi.training.distillation import (
    ...     DistillationStrategy,
    ...     PerAtomEnergyMatchingLoss,
    ... )
    >>> loss_fn = (
    ...     EnergyMSELoss(target_key="teacher_energy")
    ...     + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True)
    ...     + 0.1 * PerAtomEnergyMatchingLoss()
    ... )
    >>> strategy = DistillationStrategy(  # doctest: +SKIP
    ...     models={"student": student, "teacher": teacher},
    ...     optimizer_configs={
    ...         "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
    ...     },
    ...     loss_fn=loss_fn,
    ...     num_steps=1_000,
    ... )
    >>> strategy.run(labeled_loader)  # doctest: +SKIP

    Notes
    -----
    Teacher conservativeness is deliberately not validated. A teacher that
    predicts forces with its own head rather than as the negative gradient of
    its energy is a first-class teacher here: the scorer detaches every signal
    it returns, so how the teacher produced a force never reaches the student.

    One seam does the labeling: an internal hook the strategy registers ahead
    of the caller's own, on ``BEFORE_FORWARD``, a stage both the training loop
    and the validation loop dispatch on the device-placed batch before its
    forward pass. Unlabeled validation data therefore needs no preparation, and
    a caller-supplied ``training_fn`` is covered too. Hooks are never
    serialized, so the seam is simply re-registered when :meth:`from_spec_dict`
    or :meth:`load_checkpoint` rebuilds the strategy.

    On-the-fly labels are attached to the device-placed batch the strategy
    trains on, which is a copy of the one the caller handed over, so they do not
    persist on the caller's object. A loader that replays the same systems every
    epoch therefore costs one teacher pass per epoch, which is the other reason
    a long run should label its dataset offline first.

    Labels are cast to the student's first floating-point parameter dtype, so a
    float64 teacher feeds a float32 student without a dtype error at the loss.
    The cast is resolved at construction; a student whose dtype changes
    afterwards needs a ``dtype_policy`` on the loss terms instead.

    :class:`~nvalchemi.training.ComposedLossFunction` renormalizes weights by
    default, so the ``0.1`` above is a ratio rather than a coefficient: the
    three terms run at ``1/2.1``, ``1/2.1``, and
    ``0.1/2.1``. Pass ``normalize_weights=False`` for literal weights, which
    also stops a :class:`~nvalchemi.training.losses.base.LossWeightSchedule` on
    one term from rescaling the others as it ramps.

    Checkpoints serialize every entry of ``models``, teacher included, so each
    :class:`~nvalchemi.training.hooks.CheckpointHook` write duplicates the
    frozen teacher's weights on disk. Size the checkpoint interval accordingly
    with a large teacher; storing the teacher by reference is planned.

    ``on_policy`` and ``reference_dataset`` hold live runtime objects — a
    propagator, a scorer, and datasets — that no spec can describe, so
    :meth:`to_spec_dict` omits them and warns. A strategy rebuilt from the spec
    of an on-policy run is therefore offline-shaped, and the on-policy pieces
    have to be re-supplied at construction until full recipe serialization
    lands.
    """

    teacher_signals: Annotated[
        frozenset[str] | None,
        Field(
            description=(
                "Teacher signals produced for every scored batch. ``None`` "
                "derives them from the ``teacher_*`` targets the loss reads; an "
                "explicit set must cover those and may request more."
            )
        ),
    ] = None
    label_missing: Annotated[
        bool,
        Field(
            description=(
                "Whether a batch lacking the required ``teacher_*`` fields is "
                "labeled on the fly by a teacher forward pass, in training and "
                "validation alike. ``False`` skips the teacher, so an unlabeled "
                "batch surfaces as a missing loss target."
            )
        ),
    ] = True
    on_policy: Annotated[
        OnPolicyConfig | None,
        Field(
            default=None,
            exclude=True,
            description=(
                "Segment-loop configuration turning ``run`` into on-policy "
                "distillation. ``None`` keeps the offline loop over the "
                "dataloader the caller passes to ``run``."
            ),
        ),
    ] = None
    reference_dataset: Annotated[
        BatchDatasetProtocol | None,
        Field(
            default=None,
            exclude=True,
            description=(
                "Teacher-labeled anchor dataset the on-policy mixture draws "
                "its ``1 - replay_ratio`` share from. Required whenever the "
                "ratio is below 1, and read only in on-policy mode."
            ),
        ),
    ] = None

    _scorer: InProcessTeacherScorer | None = PrivateAttr(default=None)
    _teacher_fields: tuple[str, ...] = PrivateAttr(default=())
    _replay_buffer: ReplayBuffer | None = PrivateAttr(default=None)

    @property
    def replay_buffer(self) -> ReplayBuffer | None:
        """Frames generated so far, or ``None`` before an on-policy run starts."""
        return self._replay_buffer

    @property
    def teacher_scorer(self) -> InProcessTeacherScorer:
        """Scorer producing the resolved teacher signals for one batch."""
        if self._scorer is None:
            raise RuntimeError(
                "DistillationStrategy has no teacher scorer; it is built during "
                "validation and must not be cleared."
            )
        return self._scorer

    @model_validator(mode="before")
    @classmethod
    def _default_distillation_training_fn(cls, data: Any) -> Any:
        """Fall back to the stock student-forward training function."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("training_fn") is None:
            normalized["training_fn"] = default_distillation_fn
        return normalized

    @model_validator(mode="before")
    @classmethod
    def _prepend_labeling_hook(cls, data: Any) -> Any:
        """Put the internal teacher-labeling hook ahead of the caller's hooks."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["hooks"] = [
            _TeacherLabelHook(),
            *list(normalized.get("hooks") or []),
        ]
        return normalized

    @model_validator(mode="after")
    def _validate_distillation(self) -> DistillationStrategy:
        """Enforce the student/teacher contract and resolve the teacher signals."""
        missing_models = _REQUIRED_MODELS - set(self.models)
        if self.single_model_input or missing_models:
            raise ValueError(
                "DistillationStrategy needs a named-model mapping holding "
                f"'student' and 'teacher'; got models={sorted(self.models)!r}."
            )
        if "teacher" in self.optimizer_configs:
            raise ValueError(
                "The teacher is frozen by omission, so it must not appear in "
                f"optimizer_configs; got {sorted(self.optimizer_configs)!r}."
            )
        unconfigured = set(self.models) - set(self.optimizer_configs) - {"teacher"}
        if unconfigured:
            raise ValueError(
                "Every model but the teacher must be given an optimizer config; "
                f"got unconfigured {sorted(unconfigured)!r}."
            )
        self._validate_student_outputs()
        signals = self._resolve_teacher_signals()
        self._scorer = InProcessTeacherScorer(
            self.models["teacher"],
            signals,
            cast_to=_student_label_dtype(self.models["student"]),
        )
        self._teacher_fields = tuple(
            sorted(_SIGNAL_SPECS[name].field for name in signals)
        )
        return self

    def _validate_student_outputs(self) -> None:
        """Check the loss's prediction keys against the student's declared outputs."""
        if self.training_fn is not default_distillation_fn:
            return
        declared = self.models["student"].model_config.outputs
        for component in self.loss_fn.components:
            key = getattr(component, "prediction_key", None)
            if key is None:
                continue
            output = key.removeprefix("predicted_")
            if output not in declared:
                raise ValueError(
                    "Student cannot produce the output required by loss component "
                    f"{type(component).__name__!r} reading prediction_key={key!r}; "
                    f"got outputs={sorted(declared)!r}, missing {output!r}."
                )

    def _resolve_teacher_signals(self) -> frozenset[str]:
        """Return the signal set the loss needs, widened by an explicit request."""
        derived = _derived_teacher_signals(self.loss_fn)
        resolved = derived if self.teacher_signals is None else self.teacher_signals
        uncovered = derived - resolved
        if uncovered:
            raise ValueError(
                "teacher_signals must cover every teacher target the loss reads; "
                f"got {sorted(resolved)!r}, missing {sorted(uncovered)!r}."
            )
        if not resolved:
            raise ValueError(
                "DistillationStrategy needs at least one teacher signal; got a "
                "loss reading no teacher target and "
                f"teacher_signals={self.teacher_signals!r}."
            )
        return resolved

    @model_validator(mode="after")
    def _validate_on_policy(self) -> DistillationStrategy:
        """Enforce the segment loop's duration, ownership, and mixture contract."""
        if self.on_policy is None:
            if self.reference_dataset is not None:
                raise ValueError(
                    "reference_dataset anchors the on-policy mixture and is "
                    "read only by the segment loop; got it set alongside "
                    "on_policy=None. Offline distillation trains on the "
                    "dataloader passed to run()."
                )
            return self
        if self.num_steps is None:
            raise ValueError(
                "On-policy distillation is sized in optimizer steps: every "
                "segment builds its own loader, so there is no fixed epoch to "
                f"convert. Got num_epochs={self.num_epochs!r}; set num_steps "
                "instead."
            )
        propagator_model = getattr(self.on_policy.dynamics, "model", None)
        if propagator_model is not self.models["student"]:
            raise ValueError(
                "OnPolicyConfig.dynamics must propagate the very module "
                "registered as models['student']: the data is on-policy only "
                "because each optimizer step is immediately visible to the "
                "propagator. Got a propagator holding a separate "
                f"{type(propagator_model).__name__} instance; build the "
                "dynamics around the student object itself."
            )
        if self.on_policy.replay_ratio == 0.0:
            raise ValueError(
                "replay_ratio=0 trains on reference data only, which is "
                "offline distillation paying for generation it never uses; "
                "drop on_policy and call run() with a loader over the labeled "
                "dataset instead."
            )
        if self.on_policy.replay_ratio < 1.0 and self.reference_dataset is None:
            raise ValueError(
                "A replay_ratio below 1 mixes reference data into every batch, "
                f"so reference_dataset is required; got replay_ratio="
                f"{self.on_policy.replay_ratio!r} and reference_dataset=None."
            )
        self._warn_on_partial_generation_signals()
        return self

    def _warn_on_partial_generation_signals(self) -> None:
        """Warn when generated frames will be relabeled on their way into training."""
        missing = frozenset(self.teacher_scorer.signals) - frozenset(
            self.on_policy.teacher_scorer.signals
        )
        if missing:
            warnings.warn(
                "The propagator's scorer does not produce every teacher signal "
                "the loss reads, so each generated frame is scored twice: once "
                "during generation and again on its way into a training step; "
                f"missing {sorted(missing)!r}. Request those signals on "
                "OnPolicyConfig.teacher_scorer to pay the teacher once.",
                UserWarning,
                stacklevel=2,
            )

    def attach_teacher_labels(self, batch: Batch) -> bool:
        """Attach the teacher fields *batch* is missing, and report whether it did.

        Labeling is idempotent: a batch that already carries every required
        ``teacher_*`` field is returned untouched, so re-training on a batch, or
        pre-labeling one that later reaches :meth:`run`, costs at most one
        teacher forward pass.

        Parameters
        ----------
        batch : Batch
            Batch to label in place. It must already sit on the teacher's
            device, which is the case for batches the strategy itself moves.

        Returns
        -------
        bool
            ``True`` when the teacher ran and fields were attached, ``False``
            when *batch* already carried them all.
        """
        if all(field in batch for field in self._teacher_fields):
            return False
        _attach_teacher_labels(batch, self.teacher_scorer.label(batch))
        return True

    def run(self, dataloader: Iterable[Batch] | None = None) -> None:
        """Execute the offline training loop or the on-policy segment loop.

        Without ``on_policy`` this is
        :meth:`~nvalchemi.training.TrainingStrategy.run` over *dataloader*,
        unchanged. With it, the strategy owns the loop and repeats three phases
        until ``num_steps`` optimizer steps have run:

        *Generate* — the propagator advances the live state batch by
        ``segment_steps``, seeded on the first segment from ``seed_dataset``.
        *Label and capture* — a
        :class:`~nvalchemi.training.distillation.TeacherLabelHook` registered on
        the propagator scores every ``label_frequency`` steps and mirrors each
        labeled frame into a host-memory sink; the segment's final frame is
        labeled too, then the sink is drained into the replay buffer.
        *Train* — a freshly built mixed loader draws ``steps_per_segment``
        batches at the configured ``replay_ratio``, each of which goes through
        the ordinary per-batch stages.

        Parameters
        ----------
        dataloader : Iterable[Batch] | None, optional
            Batches to train on in offline mode; any iterable, not necessarily
            a :class:`~nvalchemi.data.datapipes.dataloader.DataLoader`. Default
            ``None``, which is required in on-policy mode and rejected
            otherwise.

        Raises
        ------
        ValueError
            If *dataloader* is ``None`` in offline mode or supplied in
            on-policy mode, or if a segment's loader produces no batches.

        Notes
        -----
        One segment is one epoch: ``AFTER_EPOCH`` fires at each segment
        boundary and an epoch-cadence ``validation_config`` follows the
        segments, while a step-cadence one fires inside them, exactly as in the
        offline loop. Validation data is labeled on the fly by the same
        ``BEFORE_FORWARD`` seam that labels training batches, and generated
        frames arrive pre-labeled, so that seam skips them. The buffer the
        segments fill stays reachable as :attr:`replay_buffer` afterwards.

        The student generates in evaluation mode and enters training mode only
        for the training phase, so a conservative student's forces cost no
        second-order graph during generation. The teacher stays frozen and in
        evaluation mode across both phases.

        The loop deliberately leaves out three pieces of the offline loop's
        bookkeeping. It never seeks a dataloader to a restored intra-epoch
        position, because each segment's loader is built from scratch; it does
        not call ``set_epoch`` on a sampler, for the same reason; and it passes
        no dataloader to the ``SETUP`` stage, so a hook that rewraps the
        caller's loader has nothing to rewrap. Restarting an on-policy run
        mid-segment is not modeled — the propagator state is not checkpointed —
        so a resumed run continues from a freshly seeded trajectory.

        Chunking a propagator across segments is exact for the built-in
        propagators: :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` never
        resets ``step_count`` or the integrator state, and the Langevin
        thermostat draws from a counter-based generator keyed on the cumulative
        step count, so ``2 x K`` steps in one call and two ``K``-step calls
        produce identical trajectories. Two consequences are the loop's to
        own. Each chunk re-enters the propagator's hook context, which
        truncates the output of an open/close-sensitive hook such as
        :class:`~nvalchemi.dynamics.hooks.LoggingHook` once per segment — the
        loop registers no such hook itself, and a caller who does should expect
        per-segment files. And a chunk stops early once every graph has
        converged, so progress is read from ``dynamics.step_count`` rather than
        assumed to be ``segment_steps``; graduating converged structures and
        backfilling fresh seeds is a relaxation concern handled separately.
        """
        if self.on_policy is None:
            if dataloader is None:
                raise ValueError(
                    "Offline distillation trains on the caller's batches; got "
                    "run(dataloader=None) with on_policy=None. Pass a "
                    "dataloader, or configure on_policy to generate one."
                )
            super().run(dataloader)
            return
        if dataloader is not None:
            raise ValueError(
                "On-policy distillation builds its own loader every segment "
                "from reference_dataset and the replay buffer; got a "
                f"{type(dataloader).__name__} passed to run(). Set it as "
                "reference_dataset instead."
            )
        self._run_on_policy(self.on_policy)

    def _run_on_policy(self, config: OnPolicyConfig) -> None:
        """Drive generate-label-train segments until ``num_steps`` is reached."""
        training_started = False
        strategy_context = nullcontext(self) if self._context_depth > 0 else self
        with strategy_context:
            self._prepare_setup_hooks()
            self._validate_runtime_devices()
            self.models = move_to_devices(self.models, self.devices)
            self._run_setup_hooks()
            target_step_count = self._resolve_target_step_count(None)
            if self.step_count >= target_step_count:
                return
            self._apply_requires_grad_filter()
            try:
                primary_device = self.devices[0]
                flat_opts, flat_scheds = self._setup_runtime_optimizers(
                    rebuild=not self._resume_optimizer_state
                )
                state = self._seed_state(config).to(primary_device, non_blocking=True)
                buffer = ReplayBuffer(
                    capacity=config.replay_capacity, eviction=config.replay_eviction
                )
                self._replay_buffer = buffer
                sink = HostMemory(
                    capacity=(config.segment_steps + 1) * state.num_graphs
                )
                label_hook = TeacherLabelHook(
                    config.teacher_scorer, sink=sink, frequency=config.label_frequency
                )
                config.dynamics.register_hook(label_hook)
                try:
                    # The teacher is frozen across both phases; the student only
                    # enters training mode to be trained, so it generates in
                    # eval mode and its forces cost no second-order graph.
                    with freeze_unconfigured_models(
                        self.models, self.optimizer_configs
                    ):
                        while self.step_count < target_step_count:
                            state = config.dynamics.run(
                                state, n_steps=config.segment_steps
                            )
                            self._capture_segment(config, state, label_hook, buffer)
                            segment_steps = min(
                                config.steps_per_segment,
                                target_step_count - self.step_count,
                            )
                            with train_configured_models(
                                self.models, self.optimizer_configs
                            ):
                                training_started = self._train_segment(
                                    config,
                                    buffer,
                                    segment_steps=segment_steps,
                                    target_step_count=target_step_count,
                                    training_started=training_started,
                                    flat_opts=flat_opts,
                                    flat_scheds=flat_scheds,
                                )
                finally:
                    config.dynamics.hooks.remove(label_hook)

                if self._last_batch is not None:
                    self._update_hook_snapshot(loss_out=None)
                    self._run_hooks(TrainingStage.AFTER_TRAINING, self._last_batch)
                    if self.validation_config is not None:
                        self.validate()
                        self._step_metric_schedulers()
            finally:
                self._restore_requires_grad_filter()

    def _train_segment(
        self,
        config: OnPolicyConfig,
        buffer: ReplayBuffer,
        *,
        segment_steps: int,
        target_step_count: int,
        training_started: bool,
        flat_opts: list[torch.optim.Optimizer],
        flat_scheds: list[LRScheduler | None],
    ) -> bool:
        """Train one segment's mixture and close it as an epoch.

        Returns
        -------
        bool
            Whether the ``BEFORE_TRAINING`` stage has fired by now.
        """
        loader = build_mixed_loader(
            self.reference_dataset,
            buffer,
            replay_ratio=config.replay_ratio,
            batch_size=config.batch_size,
            num_batches=segment_steps,
        )
        primary_device = self.devices[0]
        consumed = 0
        for batch in loader:
            if consumed >= segment_steps or self.step_count >= target_step_count:
                break
            batch = batch.to(primary_device, non_blocking=True)
            self._update_hook_snapshot(batch=batch, loss_out=None)
            if not training_started:
                self._run_hooks(TrainingStage.BEFORE_TRAINING, batch)
                training_started = True
            if self.epoch_step_count == 0:
                self._run_hooks(TrainingStage.BEFORE_EPOCH, batch)
            self._train_batch_with_optimizers(batch, flat_opts, flat_scheds)
            self._validation_checkpoint(TrainingStage.AFTER_OPTIMIZER_STEP)
            consumed += 1
        if consumed == 0:
            raise ValueError(
                "the segment's mixed loader produced no batches before reaching "
                "the target step count; ensure reference_dataset and the replay "
                "buffer together hold at least one batch of "
                f"{config.batch_size} samples."
            )
        self.epoch_count += 1
        self.epoch_step_count = 0
        self._refresh_hook_counters()
        self._run_hooks(TrainingStage.AFTER_EPOCH, self._last_batch)
        self._validation_checkpoint(TrainingStage.AFTER_EPOCH)
        return training_started

    def _seed_state(self, config: OnPolicyConfig) -> Batch:
        """Return the batch the first segment propagates from.

        A ``sampler`` bin-packs the initial batch under its own size budget.
        Without one the whole seed dataset is propagated as a single batch,
        which keeps the trajectory count explicit: the seed dataset *is* the
        set of systems the run generates from, so size it to the device.
        """
        if config.sampler is not None:
            return config.sampler.build_initial_batch()
        seeds = config.seed_dataset
        return seeds.load_batches([list(range(len(seeds)))])[0]

    def _capture_segment(
        self,
        config: OnPolicyConfig,
        state: Batch,
        label_hook: TeacherLabelHook,
        buffer: ReplayBuffer,
    ) -> None:
        """Label the frame the segment ended on and drain the sink into *buffer*.

        The propagator's cadence rarely lands on a segment's last step, and that
        frame is the most on-policy one the segment produced, so the hook is
        dispatched once more against the step it just finished. Labeling is
        idempotent per step, so a cadence that did land there costs nothing and
        stores nothing twice.
        """
        label_hook(
            DynamicsContext(
                batch=state,
                step_count=max(config.dynamics.step_count - 1, 0),
                model=config.dynamics.model,
                workflow=config.dynamics,
            ),
            DynamicsStage.AFTER_STEP,
        )
        if label_hook.sink is not None and len(label_hook.sink) > 0:
            buffer.extend(label_hook.sink.drain())

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize declarative distillation knobs to a JSON-ready dict.

        ``on_policy`` and ``reference_dataset`` are omitted: they hold a live
        propagator, scorer, and datasets, none of which a spec can describe
        yet. Rebuilding an on-policy strategy from its spec therefore yields an
        offline-shaped one, and re-supplying the on-policy objects at
        construction is the only way back until recipe serialization lands.

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.

        Warns
        -----
        UserWarning
            If ``on_policy`` is set, because the spec cannot carry it.
        """
        spec = super().to_spec_dict()
        spec["teacher_signals"] = (
            None if self.teacher_signals is None else sorted(self.teacher_signals)
        )
        spec["label_missing"] = self.label_missing
        if self.on_policy is not None:
            warnings.warn(
                "on_policy and reference_dataset hold live runtime objects and "
                "are omitted from the spec, so a strategy rebuilt from it runs "
                "offline over the dataloader passed to run(). Re-supply them at "
                "construction to keep generating on-policy.",
                UserWarning,
                stacklevel=2,
            )
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: Sequence[Any] | None = None,
        training_fn: Any = None,
    ) -> DistillationStrategy:
        """Rebuild a :class:`DistillationStrategy` from ``to_spec_dict`` output.

        Parameters
        ----------
        spec : Mapping[str, Any]
            A dict produced by :meth:`to_spec_dict`, optionally after a JSON
            round-trip.
        models : BaseModelMixin | dict[str, BaseModelMixin] | None, optional
            Runtime model override(s). Distillation models are not serialized
            in full, so the student and teacher are normally re-supplied here.
        hooks : Sequence[Any] | None, optional
            Runtime hooks; defaults to an empty list.
        training_fn : Any, optional
            Runtime callable or dotted-path override.

        Returns
        -------
        DistillationStrategy
            A freshly validated distillation strategy ready to :meth:`run`.
        """
        required = ("optimizer_configs", "devices", "loss_fn_spec")
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(
                f"from_spec_dict: spec is missing required key(s) {missing}. "
                f"Expected keys: {list(required)}."
            )
        model_input = strategy_spec._models_from_spec_and_overrides(
            spec.get("model_specs", {}),
            models,
            single_model_input=strategy_spec._single_model_input_from_spec(
                spec.get("single_model_input")
            ),
        )
        return cls(
            models=model_input,
            optimizer_configs=strategy_spec._optimizer_configs_from_spec(
                spec["optimizer_configs"]
            ),
            num_epochs=spec.get("num_epochs"),
            num_steps=spec.get("num_steps"),
            epoch_step_modifier=spec.get("epoch_step_modifier", 1.0),
            hooks=list(hooks) if hooks is not None else [],
            training_fn=strategy_spec._training_fn_from_spec(spec, training_fn),
            loss_fn=strategy_spec._loss_fn_from_spec(spec["loss_fn_spec"]),
            devices=strategy_spec._devices_from_spec(spec["devices"]),
            teacher_signals=spec.get("teacher_signals"),
            label_missing=spec.get("label_missing", True),
        )
