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
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import Field, PrivateAttr, model_validator

from nvalchemi._serialization import _import_cls
from nvalchemi._typing import ModelOutputs
from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distillation._labels import (
    _TEACHER_FIELD_PREFIX,
    _attach_teacher_labels,
    _reject_foreign_fields,
)
from nvalchemi.training.distillation._restart import (
    _batch_from_state,
    _OnPolicyRestartHook,
)
from nvalchemi.training.distillation.config import (
    OnPolicyConfig,
    _dataset_from_spec_dict,
    _dataset_spec_dict,
)
from nvalchemi.training.distillation.hooks import TeacherLabelHook, _run_local_keys
from nvalchemi.training.distillation.replay import (
    _SCHEMA_REMEDY,
    ReplayBuffer,
    _batch_allocation,
    _batch_size_remedy,
    _emitted_device,
    _frame_schema,
    _same_device,
    build_mixed_loader,
)
from nvalchemi.training.distillation.scoring import (
    _EMBEDDING_KEYS,
    SUPPORTED_SIGNALS,
    InProcessTeacherScorer,
    scorer_fields,
    signal_fields,
    signal_for_field,
)
from nvalchemi.training.distributed import get_world_size
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
    from nvalchemi.training.losses.composition import (
        BaseLossFunction,
        ComposedLossFunction,
    )

__all__ = ["DistillationStrategy", "default_distillation_fn"]

_REQUIRED_MODELS = frozenset({"student", "teacher"})
"""Model names every distillation strategy must be given."""

_PREDICTION_KEY_PREFIX = "predicted_"
"""Prefix the stock training function publishes every student output under."""


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
        f"{_PREDICTION_KEY_PREFIX}{key}": value
        for key, value in outputs.items()
        if value is not None
    }


def _derived_teacher_signals(
    loss_fn: ComposedLossFunction, *, supplied_fields: Collection[str] = ()
) -> frozenset[str]:
    """Return the teacher signals the loss composition's targets require.

    A ``teacher_*`` target listed in *supplied_fields* is skipped rather than
    resolved or refused: it is a generation-supplied target, written onto every
    captured frame by the on-policy propagator's own scorer, so no signal of
    the strategy's scorer stands behind it.

    Parameters
    ----------
    loss_fn : ComposedLossFunction
        Loss composition whose target keys are read.
    supplied_fields : Collection[str], optional
        Batch fields the on-policy propagator's scorer declares it writes.
        Default ``()``, which is offline distillation, where nothing but a
        built-in signal populates the namespace.

    Returns
    -------
    frozenset[str]
        Signal names the strategy's own scorer has to produce.

    Raises
    ------
    ValueError
        If a ``teacher_*`` target maps to no built-in signal and no scorer
        declares it.
    """
    signals: set[str] = set()
    for key in loss_target_keys(loss_fn):
        if not key.startswith(_TEACHER_FIELD_PREFIX):
            continue
        signal = signal_for_field(key)
        if signal is None:
            if key in supplied_fields:
                continue
            raise ValueError(
                "Loss targets must name a supported teacher target from "
                f"{list(signal_fields(SUPPORTED_SIGNALS))!r}; got {key!r}. The "
                f"{_TEACHER_FIELD_PREFIX!r} prefix is reserved for those signals, so "
                "a field a custom scorer writes must be named outside it to reach "
                "the loss as an ordinary batch field — unless an on-policy "
                "propagator's scorer declares it in label_fields, which makes it a "
                "generation-supplied target every captured frame carries."
            )
        signals.add(signal)
    return frozenset(signals)


@contextmanager
def _eval_configured_models(
    models: Mapping[str, torch.nn.Module], optimizer_configs: Mapping[str, object]
) -> Iterator[None]:
    """Temporarily put the optimizer-configured models in evaluation mode.

    The mirror of
    :func:`~nvalchemi.training.runtime.train_configured_models`, which only
    ever sets training mode and restores the mode it found. A model that is
    never told otherwise therefore runs in training mode outside a training
    phase — with dropout live, batch-norm statistics moving, and a
    conservative model's forces building a second-order graph — which is what
    the on-policy loop's generation phase has to avoid.

    Parameters
    ----------
    models : Mapping[str, torch.nn.Module]
        Named models participating in the run.
    optimizer_configs : Mapping[str, object]
        Optimizer configuration keyed by model name. Models present in it are
        switched to evaluation mode while the context is active.

    Yields
    ------
    None
        Control while the configured models are in evaluation mode.
    """
    state = {
        name: model.training
        for name, model in models.items()
        if name in optimizer_configs
    }
    for name in state:
        models[name].eval()
    try:
        yield
    finally:
        for name, training in state.items():
            models[name].train(training)


@contextmanager
def _eval_propagator_model(
    propagator_model: object, student: BaseModelMixin
) -> Iterator[None]:
    """Temporarily put a propagator model that only *composes* the student in eval mode.

    :func:`_eval_configured_models` reaches the named models an optimizer
    updates, which a composition holding the student is not: it is no entry of
    ``models``, so nothing else ever takes it out of training mode. Left there,
    a shared-autograd composition differentiates its summed energy with
    ``create_graph=True`` — the second-order graph the generation phase exists
    to avoid — and every submodule the student does not own keeps moving its
    batch-norm statistics on generated frames. Enter this context *inside*
    :func:`_eval_configured_models`: restoring a composition's mode sets the
    mode of every module it holds, the student included, so it has to happen
    before the student's own mode is put back.

    Every submodule's own mode is snapshotted, not just the composition root's.
    :meth:`~torch.nn.Module.train` stamps one flag recursively, so restoring
    the root alone would hand back a frozen correction head — one the caller
    had put in evaluation mode individually, which a non-teacher entry of
    ``models`` cannot be because every one of those needs an optimizer config —
    in training mode, silently running its dropout afterwards.

    Parameters
    ----------
    propagator_model : object
        Model the propagator holds. A propagator holding *student* itself, or
        anything that is not a :class:`torch.nn.Module`, is left alone.
    student : BaseModelMixin
        Student the strategy trains, whose own mode
        :func:`_eval_configured_models` owns.

    Yields
    ------
    None
        Control while the composing model is in evaluation mode.
    """
    if propagator_model is student or not isinstance(propagator_model, torch.nn.Module):
        yield
        return
    modes = {module: module.training for module in propagator_model.modules()}
    propagator_model.eval()
    try:
        yield
    finally:
        for module, training in modes.items():
            module.training = training


def _propagates_student(propagator_model: object, student: BaseModelMixin) -> bool:
    """Return whether *propagator_model* is *student* or a model composing it."""
    if propagator_model is student:
        return True
    modules = getattr(propagator_model, "modules", None)
    return callable(modules) and any(module is student for module in modules())


def _student_label_dtype(student: BaseModelMixin) -> torch.dtype | None:
    """Return the dtype teacher labels are cast to for *student*.

    The first floating-point parameter decides, but never below single
    precision: a ``bfloat16``, ``float16``, or narrower student gets
    ``float32`` labels, while ``float32`` and ``float64`` are kept as they are.
    Two things make reduced precision the wrong label dtype. A store round-trips
    every floating field to the dtype of the dataset's ``positions``, which is
    float32 for essentially every dataset, so a label below it would disagree
    with what :func:`~nvalchemi.training.distillation.label_dataset` persisted;
    and the graph-balanced reductions the loss terms use accumulate in the
    residual's dtype, where a ``bfloat16`` sum saturates at 256. A student that
    exposes no parameters at all gets ``None``, which leaves labels in the
    teacher's own dtype.
    """
    parameters = getattr(student, "parameters", None)
    if not callable(parameters):
        return None
    for parameter in parameters():
        if parameter.is_floating_point():
            if parameter.dtype.itemsize < torch.float32.itemsize:
                return torch.float32
            return parameter.dtype
    return None


def _to_device(batch: Batch, device: torch.device) -> Batch:
    """Return *batch* on *device*, overlapping the copy only into device memory.

    A copy into device memory is queued asynchronously so it overlaps the work
    already on the stream, and stream ordering keeps every consumer behind it.
    A copy into host memory has no such ordering: ATen issues the transfer and
    returns without synchronizing, so a read that follows the call can observe
    a destination the transfer has not filled. That race is not tolerable here
    because the moved batch's index tensors are read on the host immediately —
    ``segment_lengths`` feeds the ``repeat_interleave`` behind ``batch_idx``,
    and ``batch_ptr`` slices the per-graph rows — where a half-written buffer
    surfaces as negative repeats, out-of-range indices, or a hang rather than
    as a wrong number.
    """
    return batch.to(device, non_blocking=device.type != "cpu")


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
    :class:`~nvalchemi.training.losses.base.LossWeightSchedule` — offline,
    where every sample carries its own reference labels. An on-policy run
    cannot: generated frames have no reference labels, and both mixture sources
    are required to carry the same fields, so its anchor has to be
    teacher-labeled rather than reference-labeled.

    Those targets also decide what the teacher is asked for.
    ``teacher_signals=None`` (the default) derives the signal set from the
    ``teacher_*`` targets the loss reads — the validation loss's included,
    whenever ``validation_config`` carries its own ``loss_fn`` — so objective
    and teacher cannot drift apart; an explicit set must cover the derived one
    and may add more. The resolved set is checked against the teacher's declared
    outputs at construction, as is the model/optimizer contract above and — for
    the stock ``training_fn`` — every loss component's prediction key against
    the outputs the student actually computes, which is its ``active_outputs``
    intersected with its declared ``outputs``, so a misconfigured run fails
    before it starts rather than on its first batch. The validation loss's
    prediction keys go through the same check whenever the effective validation
    function, ``validation_config.validation_fn`` falling back to
    ``training_fn``, is the stock one. None of this re-runs on assignment, so a
    ``validation_config`` attached after construction keeps the signals already
    resolved: pass it to the constructor, or name the wider set in
    ``teacher_signals``.

    Every resolved signal is a request for its fields on every batch rather
    than a permission to carry them, whether it was derived or named in
    ``teacher_signals``. A batch counts as labeled only when it holds every
    resolved field, so adding a validation loss with a new ``teacher_*`` target
    puts a training store written before it back on the teacher batch after
    batch — the same values, at the price of a forward pass each time. A store
    meant to train with no teacher pass at all has to be labeled with the same
    signal set the strategy resolves.

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
        prediction the student does not compute or names one outside the
        ``predicted_`` namespace under the stock ``training_fn``, if a loss reads
        a ``teacher_*`` target that maps to no known signal and that no
        on-policy propagator's scorer declares, if an explicit
        ``teacher_signals`` omits a signal a loss needs, if no teacher signal
        is requested at all, if the teacher cannot produce a requested signal,
        or if the teacher is a composition that plans more than one
        neighbor-list source. In on-policy mode, additionally if the run is
        sized in epochs rather than steps, if the propagator holds neither the
        student nor a model composing it, if ``replay_ratio`` is ``0``, if a
        ratio below ``1`` is paired with no ``reference_dataset``, if a ratio
        of ``1`` is paired with one, if the ratio and ``batch_size`` together
        allocate no samples to one mixture source, if ``replay_device`` names
        a device the ``reference_dataset`` does not emit on, if the
        ``reference_dataset`` carries fields the labeling hook strips from
        every generated frame, if the propagator's scorer declares a field
        outside the ``teacher_*`` namespace, or if that scorer's known fields
        and ``reference_dataset`` do not carry the same teacher fields.

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

    A composed teacher whose stages plan more than one neighbor-list source is
    refused at construction, since the scorer builds exactly one list per
    batch. Compose it to plan a single list instead —
    ``neighbor_adaptation="always"``, or a ``max_cutoff_ratio`` of at least
    the ratio of its largest to its smallest cutoff — and the scorer adapts
    that one list per batch.

    One seam does the labeling: an internal hook the strategy registers ahead
    of the caller's own, on ``BEFORE_FORWARD``, a stage both the training loop
    and the validation loop dispatch on the device-placed batch before its
    forward pass. Unlabeled validation data therefore needs no preparation — a
    ``validation_config`` with its own ``loss_fn`` has its ``teacher_*`` targets
    derived and its prediction keys checked alongside the training loss's — and
    a caller-supplied ``training_fn`` is covered too. Hooks are never
    serialized, so the seam is simply re-registered when :meth:`from_spec_dict`
    or :meth:`load_checkpoint` rebuilds the strategy, and a seam carried in the
    ``hooks`` such a rebuild is handed is replaced rather than kept, so chained
    rebuilds never accumulate one.

    Validating an EMA-averaged student against the live teacher is what
    ``ValidationConfig(use_ema="auto")`` does: the student's averaged weights
    replace its live ones, the teacher stays live, and the pass reports
    ``model_source="mixed"``. ``use_ema="always"`` currently also demands an
    inference-slot entry for the frozen teacher and fails at the first
    validation pass without one.

    On-the-fly labels are attached to the device-placed batch the strategy
    trains on, which is a copy of the one the caller handed over, so they do not
    persist on the caller's object. A loader that replays the same systems every
    epoch therefore costs one teacher pass per epoch, which is the other reason
    a long run should label its dataset offline first.

    The ``teacher_`` prefix is reserved for the built-in signals, so a loss
    target under it that names none of them is refused rather than left to fail
    as a missing batch field. A custom scorer's own field — anything
    :func:`~nvalchemi.training.distillation.label_dataset` persisted outside
    that signal set — reaches the loss as an ordinary batch field by being
    named outside the prefix, and is then invisible to signal derivation, which
    is what an explicit ``teacher_signals`` is for.

    On-policy runs relax that rule in exactly one way, the generation-supplied
    target: a ``teacher_*`` target naming no built-in signal is accepted when
    the propagator's scorer declares it in ``label_fields``, because that
    scorer writes it onto every frame the labeling hook captures and the frame
    carries it into the replay buffer. Such a field derives no signal — this
    strategy's own scorer produces built-in signals only — so at least one
    built-in ``teacher_*`` target, or an explicit ``teacher_signals``, is still
    required alongside it, ``reference_dataset`` has to carry it too (which the
    generation/anchor parity check enforces), and any validation data has to
    arrive already carrying it, since nothing labels it on the fly: a
    validation batch without it surfaces as the loss's missing-target
    ``KeyError``. A scorer declaring no ``label_fields`` supplies nothing, its
    fields being unknowable until it has scored a batch, so a custom target
    read against it is refused exactly as offline.

    Labeling runs with autocast disabled, so the teacher computes at its own
    precision no matter what precision context the surrounding training or
    validation step establishes, and an on-the-fly label matches the offline one
    bit for bit wherever the store returns the label dtype: a store round-trips
    every floating field to the dtype of the dataset's ``positions``, so over
    the usual float32 dataset every student but a float64 one sees identical
    labels on both paths, while a float64 student reads float32 back from the
    store and needs a ``dtype_policy`` to train from it.

    Labels are cast to the student's first floating-point parameter dtype, but
    never below single precision: a ``bfloat16`` or ``float16`` student gets
    float32 labels, because a store round-trips every floating field to the
    dtype of the dataset's ``positions`` and graph-balanced reductions
    accumulate in the residual's dtype. Such a student therefore needs
    ``dtype_policy="prediction_to_target"`` on its loss terms, which computes
    the loss in float32; a float64 teacher feeds a float32 student with no
    dtype policy at all. The cast is resolved at construction, so a student
    whose dtype changes afterwards needs a ``dtype_policy`` too.

    :class:`~nvalchemi.training.ComposedLossFunction` renormalizes weights by
    default, so the ``0.1`` above is a ratio rather than a coefficient: the
    three terms run at ``1/2.1``, ``1/2.1``, and
    ``0.1/2.1``. Pass ``normalize_weights=False`` for literal weights, which
    also stops a :class:`~nvalchemi.training.losses.base.LossWeightSchedule` on
    one term from rescaling the others as it ramps.

    The teacher is stored once per checkpoint root rather than at every index,
    so a periodic write costs the student's weights rather than the student's
    plus a frozen foundation teacher's; see :meth:`checkpoint_model_references`
    for what that means for a restart.

    ``on_policy`` and ``reference_dataset`` serialize as references too — the
    propagator's spec, the scorer's signal set over the strategy's own teacher,
    and the stores the datasets read — so a whole on-policy recipe survives
    :meth:`to_spec_dict` and rebuilds around re-supplied models. A run whose
    datasets live in memory, or whose propagator hides its constructor
    arguments, leaves the recipe out of the spec with a warning naming the
    piece, and re-supplying the on-policy objects at construction is then the
    way back. An interrupted on-policy run additionally carries its trajectory,
    the propagator's step count, and its replay frames through the checkpoint,
    so a resumed run continues the same trajectory instead of seeding a fresh
    one.
    """

    teacher_signals: Annotated[
        frozenset[str] | None,
        Field(
            description=(
                "Teacher signals produced for every scored batch. ``None`` "
                "derives them from the ``teacher_*`` targets the training and "
                "validation losses read; an explicit set must cover those and "
                "may request more, at the cost of re-scoring every batch a "
                "store labeled without the extra fields delivers."
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
    _on_policy_state: Any = PrivateAttr(default=None)
    _validated_step: int | None = PrivateAttr(default=None)

    @property
    def replay_buffer(self) -> ReplayBuffer | None:
        """Frames generated so far, or ``None`` before an on-policy run starts.

        One buffer serves every :meth:`run` call on a strategy, so a run
        continued with a raised ``num_steps`` keeps training on everything
        generated so far instead of throwing it away and regenerating it. The
        trajectory is still reseeded per call.
        """
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
        """Put the internal labeling and restart hooks ahead of the caller's hooks.

        A seam carried in the incoming hooks is replaced rather than kept, so
        rebuilding a strategy from a live one's ``hooks`` leaves exactly one of
        each, still ahead of every caller hook.
        """
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        internal: list[Any] = [_TeacherLabelHook()]
        if normalized.get("on_policy") is not None:
            internal.append(_OnPolicyRestartHook())
        normalized["hooks"] = [
            *internal,
            *(
                hook
                for hook in (normalized.get("hooks") or [])
                if not isinstance(hook, (_TeacherLabelHook, _OnPolicyRestartHook))
            ),
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
        self._teacher_fields = signal_fields(signals)
        return self

    def _validate_student_outputs(self) -> None:
        """Check both losses' prediction keys against the student's effective outputs.

        The stock ``training_fn`` returns exactly what the student's forward
        emits, which is ``active_outputs`` intersected with ``outputs`` rather
        than the declared set, so a student whose active set is narrowed — the
        common default for a pretrained wrapper — is caught here instead of on
        its first batch. A ``validation_config`` carrying its own ``loss_fn``
        goes through the same check whenever its effective validation function —
        ``validation_fn`` falling back to ``training_fn`` — is the stock one,
        since the validation loop reads the same predictions.
        """
        if self.training_fn is default_distillation_fn:
            self._validate_prediction_keys(self.loss_fn.components, "training")
        validation = self.validation_config
        if validation is None or validation.loss_fn is None:
            return
        if (validation.validation_fn or self.training_fn) is default_distillation_fn:
            self._validate_prediction_keys(validation.loss_fn.components, "validation")

    def _validate_prediction_keys(
        self, components: Sequence[BaseLossFunction], side: str
    ) -> None:
        """Check one composition's prediction keys, naming *side* in every error."""
        student = self.models["student"]
        declared = student.model_config.outputs
        active = student.output_data()
        for component in components:
            key = getattr(component, "prediction_key", None)
            if key is None:
                continue
            label = f"{side} loss component {type(component).__name__!r}"
            if not key.startswith(_PREDICTION_KEY_PREFIX):
                raise ValueError(
                    f"The {label} reads prediction_key={key!r}, which "
                    "default_distillation_fn never emits: it publishes every "
                    f"student output under {_PREDICTION_KEY_PREFIX}<output>. "
                    "Rename the key into that namespace, or pass a training_fn "
                    "that owns its own convention."
                )
            output = key.removeprefix(_PREDICTION_KEY_PREFIX)
            if output in active:
                continue
            if output in _EMBEDDING_KEYS:
                raise ValueError(
                    f"The {label} reads prediction_key={key!r}, which the stock "
                    "training_fn cannot produce: embeddings come from the "
                    "student's compute_embeddings(), not from its forward pass. "
                    "Pass a training_fn that calls compute_embeddings and returns "
                    f"the embedding under {key!r}."
                )
            if output in declared:
                raise ValueError(
                    "Student declares but does not compute the output required by "
                    f"the {label} reading prediction_key={key!r}; got "
                    f"active_outputs={sorted(active)!r}, missing {output!r}. Add "
                    "it to the student's model_config.active_outputs."
                )
            raise ValueError(
                f"Student cannot produce the output required by the {label} "
                f"reading prediction_key={key!r}; got outputs={sorted(declared)!r}, "
                f"missing {output!r}."
            )

    def _resolve_teacher_signals(self) -> frozenset[str]:
        """Return the signals both losses need, widened by an explicit request."""
        # Pydantic populates every field before the first mode="after"
        # validator, so the propagator's scorer is already readable here.
        supplied = (
            ()
            if self.on_policy is None
            else scorer_fields(self.on_policy.teacher_scorer) or ()
        )
        derived = {
            "training": _derived_teacher_signals(self.loss_fn, supplied_fields=supplied)
        }
        validation = self.validation_config
        if validation is not None and validation.loss_fn is not None:
            derived["validation"] = _derived_teacher_signals(
                validation.loss_fn, supplied_fields=supplied
            )
        required: frozenset[str] = frozenset().union(*derived.values())
        resolved = required if self.teacher_signals is None else self.teacher_signals
        uncovered = {
            side: sorted(signals - resolved)
            for side, signals in derived.items()
            if signals - resolved
        }
        if uncovered:
            raise ValueError(
                "teacher_signals must cover every teacher target the training and "
                f"validation losses read; got {sorted(resolved)!r}, missing "
                f"{uncovered!r}."
            )
        if not resolved:
            raise ValueError(
                "DistillationStrategy needs at least one teacher signal; got no "
                "teacher_* target in the training or validation loss and "
                f"teacher_signals={self.teacher_signals!r}. A generation-supplied "
                "target resolves no signal here, because this strategy's own "
                "scorer produces built-in signals only; pair it with a built-in "
                "teacher_* target, or request teacher_signals explicitly."
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
        if not _propagates_student(propagator_model, self.models["student"]):
            held = (
                "no model at all"
                if propagator_model is None
                else f"a separate {type(propagator_model).__name__} instance"
            )
            raise ValueError(
                "OnPolicyConfig.dynamics must propagate the very module "
                "registered as models['student'], on its own or composed into "
                "a larger model: the data is on-policy only because each "
                "optimizer step is immediately visible to the propagator. Got "
                f"a propagator holding {held}; build the dynamics around the "
                "student object itself."
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
        if self.on_policy.replay_ratio == 1.0 and self.reference_dataset is not None:
            raise ValueError(
                "replay_ratio=1 draws every sample of every batch from the "
                "replay buffer, so the anchor is policed for schema and device "
                "and then never sampled; got replay_ratio=1.0 alongside a "
                f"{type(self.reference_dataset).__name__} reference_dataset. "
                "Drop the anchor, or lower replay_ratio to mix it in."
            )
        self._validate_batch_allocation()
        # One probe answers both the device and the schema question.
        probe = (
            None
            if self.reference_dataset is None
            else self.reference_dataset.load_batches([[0]])[0]
        )
        self._validate_mixture_device(probe)
        self._validate_anchor_schema(probe)
        self._validate_generation_signals()
        return self

    def _validate_batch_allocation(self) -> None:
        """Reject a ratio that rounds one mixture source out of every batch."""
        ratio = self.on_policy.replay_ratio
        batch_size = self.on_policy.batch_size
        reference_samples, replay_samples = _batch_allocation(ratio, batch_size)
        if ratio >= 1.0 or min(reference_samples, replay_samples) > 0:
            return
        raise ValueError(
            "The mixture is drawn as whole samples of a batch, so replay_ratio "
            "and batch_size only mean something together; got replay_ratio="
            f"{ratio!r} with batch_size={batch_size!r}, which puts "
            f"{reference_samples} reference and {replay_samples} generated "
            "samples in every batch and leaves one source out of training "
            f"entirely; {_batch_size_remedy(ratio)}."
        )

    def _validate_mixture_device(self, probe: Batch | None) -> None:
        """Reject a staging device the reference dataset cannot be collated with.

        Parameters
        ----------
        probe : Batch | None
            One batch already drawn from ``reference_dataset``, whose device is
            what a composition or a device-less store is measured by. ``None``
            when there is no anchor to measure.
        """
        if self.reference_dataset is None or self.on_policy.replay_device is None:
            return
        reference_device = _emitted_device(self.reference_dataset, probe)
        replay_device = torch.device(self.on_policy.replay_device)
        if _same_device(reference_device, replay_device):
            return
        raise ValueError(
            "A mixed batch is collated before the strategy moves it, so the "
            "replay buffer and reference_dataset have to live on one device; "
            f"got replay_device={replay_device!s} and a reference dataset "
            f"emitting on {reference_device!s}. Leave replay_device unset to "
            "stage generated frames wherever the reference dataset lives, or "
            f"load the reference dataset on {replay_device!s}."
        )

    def _validate_anchor_schema(self, probe: Batch | None) -> None:
        """Reject an anchor holding fields no generated frame can ever carry.

        The full schema comparison needs frames to compare against and so runs
        inside the first segment's
        :func:`~nvalchemi.training.distillation.build_mixed_loader`, once a
        whole generation phase — propagator steps plus a teacher pass per
        labeled frame — has already been paid for. The part that depends on
        nothing the run produces is checked here instead: the labeling hook
        strips the propagator's own predictions, the ephemeral neighbor
        tensors, and the dynamics bookkeeping from every frame it stores, so an
        anchor carrying any of them can never be mixed. A store
        :func:`~nvalchemi.training.distillation.label_dataset` wrote over an
        existing reference set — the anchor a run graduating from offline
        distillation reaches for — keeps that set's own ``energy`` and
        ``forces``, which is the part rejected here; its neighbor tensors are
        dropped by default, and the sparse list ``keep_neighbors=True`` writes
        back is rejected here too.

        Parameters
        ----------
        probe : Batch | None
            One batch already drawn from ``reference_dataset``, read here for
            the schema its levels and fields report. ``None`` when there is no
            anchor to check.
        """
        if probe is None:
            return
        dropped = _run_local_keys()
        unmixable = sorted(
            name for name in _frame_schema(probe) if name.partition(".")[2] in dropped
        )
        if not unmixable:
            return
        raise ValueError(
            "reference_dataset carries fields no generated frame can, so the "
            "mixture would be rejected on the first segment's loader; got "
            f"{unmixable!r} on the anchor, which the labeling hook strips from "
            f"every frame it stores. {_SCHEMA_REMEDY}"
        )

    def _validate_generation_signals(self) -> None:
        """Check the propagator's teacher fields against the anchor and the loss.

        A scorer that declares neither ``label_fields`` nor a set of built-in
        signals writes fields nothing can know before it has scored a batch, so
        both checks below are skipped with a warning rather than run against an
        empty set — which would reject a custom scorer that in fact produces
        exactly what the anchor carries.
        """
        generated = scorer_fields(self.on_policy.teacher_scorer)
        if generated is None:
            warnings.warn(
                "The propagator's scorer declares neither label_fields nor "
                "built-in signals, so the teacher fields it writes are unknown "
                "until the first segment has generated them: neither their "
                "parity with reference_dataset nor whether every generated "
                "frame is scored twice can be checked at construction, and a "
                "mismatch surfaces as a rejected mixture once a whole "
                "generation phase has been paid for. Got signals="
                f"{sorted(self.on_policy.teacher_scorer.signals)!r}; declare "
                "label_fields on the scorer to restore both checks.",
                UserWarning,
                stacklevel=2,
            )
            return
        _reject_foreign_fields(generated, "A scorer's label_fields")
        if self.reference_dataset is not None:
            stored = frozenset(
                field
                for field in self.reference_dataset.field_names
                if field.startswith(_TEACHER_FIELD_PREFIX)
            )
            if frozenset(generated) != stored:
                raise ValueError(
                    "Generated frames and reference_dataset must carry the same "
                    "teacher fields, because mixing them into one batch keeps "
                    f"only the fields both hold; got generation "
                    f"{sorted(generated)!r} and reference {sorted(stored)!r}. "
                    "Request the same signals on OnPolicyConfig.teacher_scorer, "
                    "or relabel the reference dataset with label_dataset."
                )
        self._warn_on_partial_generation_signals(generated)

    def _warn_on_partial_generation_signals(self, generated: tuple[str, ...]) -> None:
        """Warn when generated frames will be relabeled on their way into training.

        Compared as fields rather than as signal names, so a custom scorer
        declaring the fields the loss reads under signal names of its own is
        not accused of leaving them out.
        """
        missing = frozenset(self._teacher_fields) - frozenset(generated)
        if missing:
            warnings.warn(
                "The propagator's scorer does not produce every teacher field "
                "the loss reads, so each generated frame is scored twice: once "
                "during generation and again on its way into a training step; "
                f"missing {sorted(missing)!r}. Request the signals populating "
                "those fields on OnPolicyConfig.teacher_scorer to pay the "
                "teacher once.",
                UserWarning,
                stacklevel=2,
            )

    def attach_teacher_labels(self, batch: Batch) -> bool:
        """Attach the teacher fields *batch* is missing, and report whether it did.

        Labeling is idempotent: a batch that already carries every required
        ``teacher_*`` field is returned untouched, so re-training on a batch, or
        pre-labeling one that later reaches :meth:`run`, costs at most one
        teacher forward pass. A batch carrying only some of them is re-scored in
        full and its existing teacher fields are overwritten, since a partial
        set means the batch was labeled for a different signal set than this
        objective reads.

        The teacher runs with autocast disabled whatever the caller's precision
        context, so labels never depend on how the surrounding training step is
        configured and on-the-fly labels match what
        :func:`~nvalchemi.training.distillation.label_dataset` persisted exactly
        wherever the store returns the label dtype, which over the usual float32
        dataset is every student but a float64 one; a float64 student reads
        float32 back and needs a ``dtype_policy`` on its loss terms.

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
        with torch.autocast(device_type=batch.device.type, enabled=False):
            labels = self.teacher_scorer.label(batch)
        _attach_teacher_labels(batch, labels)
        return True

    def checkpoint_model_references(self) -> dict[str, dict[str, Any]]:
        """Return the models a checkpoint stores once per root, not at every index.

        The teacher is frozen for the whole run, so writing its weights into
        every periodic checkpoint duplicates a model that never changed — the
        dominant cost of checkpointing a foundation teacher. Declaring it here
        stores it exactly once instead: the first checkpoint written under a
        root holds the teacher's weights, and every later one records the index
        they sit at. A run's hundredth checkpoint therefore costs the student's
        weights alone, and the tree stays self-contained, so a restart reads
        back the teacher the run actually trained against.

        Storing rather than referencing an external source is what makes that
        last part true. A teacher's ``checkpoint_spec()`` names the factory call
        that built it, which is the right thing to rebuild its *architecture*
        from but not its weights: a teacher loaded from a fine-tune checkpoint,
        or given a state dict after construction, carries weights that call
        does not reproduce. The checkpoint holds those weights itself and
        fingerprints them on the way back in. The digest samples each tensor
        rather than reading it whole, so it identifies the stored copy without
        validating it: a wrong file, a re-trained teacher, or one written at
        another precision is caught before a student trains against it, while
        an edit confined to values between two samples is not.

        Returns
        -------
        dict[str, dict[str, Any]]
            ``{"teacher": {"rebuild": "stored"}}``; the checkpoint layer adds
            the index and the fingerprint.
        """
        return {"teacher": {"rebuild": "stored"}}

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
            on-policy mode, if the on-policy loop is entered on more than one
            rank, or if a segment's loader produces no batches.

        Notes
        -----
        One segment is one epoch: ``AFTER_EPOCH`` fires at each segment
        boundary and an epoch-cadence ``validation_config`` follows the
        segments, while a step-cadence one fires inside them, exactly as in the
        offline loop. The run then closes with one terminal validation, skipped
        when a cadence already validated at the final step, so a metric-driven
        scheduler is never stepped twice on one set of metrics. Validation data
        is labeled on the fly by the same ``BEFORE_FORWARD`` seam that labels
        training batches, and generated frames arrive pre-labeled, so that seam
        skips them. The buffer the
        segments fill stays reachable as :attr:`replay_buffer` afterwards.

        The student is held in evaluation mode for the whole loop and flipped
        to training mode for each training phase only, so its dropout and
        batch-norm statistics never see a generated frame and a conservative
        student's forces cost no second-order graph during generation. A
        propagator model that merely composes the student is held in evaluation
        mode for the whole loop instead, because the training phase forwards
        ``models["student"]`` rather than the composition. The teacher stays
        frozen and in evaluation mode across both phases. Every mode is
        restored on the way out.

        Generated frames reach the buffer as training samples rather than
        propagator states: the labeling hook strips the ``energy``, ``forces``,
        and ``stress`` the student wrote during
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.compute` along with the
        neighbor tensors and dynamics bookkeeping, so a replay frame carries no
        self-label under a reference target's name. That shape is what
        ``reference_dataset`` has to match: each segment's loader compares the
        anchor's own batch schema against the buffer's and rejects any
        difference, because collation drops a field only one side holds and
        zero-fills a whole level only one side holds. An anchor carrying plain
        ``energy`` or ``forces`` is therefore an error rather than a batch that
        silently loses or fabricates them — label it with
        :func:`~nvalchemi.training.distillation.label_dataset` first. On-policy
        losses read ``teacher_*``, built-in fields and any generation-supplied
        field the propagator's scorer declares alike, and the teacher fields
        the two sources carry are checked against each other at construction
        whenever the scorer declares enough for them to be known.

        Both mixture sources are collated before the strategy moves the batch,
        so generated frames are staged on the reference dataset's device unless
        ``OnPolicyConfig.replay_device`` names another one; a run with no anchor
        keeps them in host memory, where the segment's sink drained them.

        The loop leaves out two pieces of the offline loop's bookkeeping. It
        never seeks a dataloader to a restored intra-epoch position, because
        each segment's loader is built from scratch, and it passes no
        dataloader to the ``SETUP`` stage, so a hook that rewraps the caller's
        loader has nothing to rewrap. It does call ``set_epoch`` on each
        segment's sampler: a freshly built mixed sampler owns a generator keyed
        on ``OnPolicyConfig.seed`` that would otherwise restart at the same
        seed every segment and redraw the identical reference samples for the
        whole run. That knob, not the global ``torch`` seed, is what makes
        replicate runs draw independently.

        The segment is the restart granularity. A run restored from a
        checkpoint picks its trajectory, the propagator's step count, and its
        replay frames back up instead of seeding afresh, and a segment a
        checkpoint interrupted part-way is counted as finished on the way in:
        its ``AFTER_EPOCH`` hooks never fire, the batches it had left are not
        replayed, and the run opens a fresh segment at the next epoch index
        rather than redrawing the reference samples the interrupted one
        already trained on. The trajectory is continuous either way, while a
        checkpoint taken part-way through a training phase costs the resumed
        run one extra generation phase for the segment it re-enters. An
        offline run graduating to the segment loop from a partial epoch is
        closed the same way. The replay buffer is kept across calls: a second
        :meth:`run` on one strategy — continuing a finished run with a raised
        ``num_steps`` — appends to the frames the first filled instead of
        regenerating them, while still reseeding its own trajectory, so a
        ``sampler`` seed source that the first call exhausted raises on the
        second.

        Because that loader is the loop's own, it is not rank-sharded, and
        neither is the seed state: the loop refuses to start in a distributed
        world of more than one rank rather than have every rank generate,
        label, and train on the same frames. Distributing the offline path is
        unaffected, and rank-sharded generation is planned. The restart bundle
        is rank-local for the same reason it is written at all — it rides in a
        strategy checkpoint, which ``CheckpointHook`` writes on rank zero alone
        — so a bundle whose world size differs from the resuming one at either
        end is dropped with a warning and the rank seeds afresh.

        Chunking a propagator across segments is exact for the built-in
        propagators: :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` never
        resets ``step_count`` or the integrator state, and the Langevin
        thermostat draws from a counter-based generator keyed on the cumulative
        step count, so ``2 x K`` steps in one call and two ``K``-step calls
        produce identical trajectories. Three consequences are the loop's to
        own. Each chunk re-enters the propagator's hook context, which
        truncates the output of an open/close-sensitive hook such as
        :class:`~nvalchemi.dynamics.hooks.LoggingHook` once per segment — the
        loop registers no such hook itself, and a caller who does should expect
        per-segment files. And a chunk stops early once every graph has
        converged, so progress is read from ``dynamics.step_count`` rather than
        assumed to be ``segment_steps``; graduating converged structures and
        backfilling fresh seeds is a relaxation concern handled separately, and
        an ``OnPolicyConfig.sampler`` only bin-packs the initial batch rather
        than refilling it. Prefer a bare propagator to a
        :class:`~nvalchemi.dynamics.FusedStage` here for the same reason:
        a fused stage fires a priming forward pass on every ``run``, so
        chunking one into segments pays that pass once per segment.
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
            self._validate_single_process()
            self.models = move_to_devices(self.models, self.devices)
            self._run_setup_hooks()
            target_step_count = self._resolve_target_step_count(None)
            if self.step_count >= target_step_count:
                return
            self._close_interrupted_segment()
            self._apply_requires_grad_filter()
            try:
                primary_device = self.devices[0]
                flat_opts, flat_scheds = self._setup_runtime_optimizers(
                    rebuild=not self._resume_optimizer_state
                )
                if self._replay_buffer is None:
                    self._replay_buffer = ReplayBuffer(
                        capacity=config.replay_capacity,
                        eviction=config.replay_eviction,
                        device=self._resolve_replay_device(config),
                    )
                buffer = self._replay_buffer
                state = _to_device(self._resume_or_seed(config, buffer), primary_device)
                self._on_policy_state = state
                sink = HostMemory(
                    capacity=(config.segment_steps + 1) * state.num_graphs
                )
                label_hook = TeacherLabelHook(
                    config.teacher_scorer, sink=sink, frequency=config.label_frequency
                )
                config.dynamics.register_hook(label_hook)
                try:
                    # The teacher is frozen across both phases; the student sits
                    # in eval mode and is flipped to training mode by the inner
                    # context for the training phase only.
                    with (
                        freeze_unconfigured_models(self.models, self.optimizer_configs),
                        _eval_configured_models(self.models, self.optimizer_configs),
                        _eval_propagator_model(
                            config.dynamics.model, self.models["student"]
                        ),
                    ):
                        while self.step_count < target_step_count:
                            state = config.dynamics.run(
                                state, n_steps=config.segment_steps
                            )
                            self._on_policy_state = state
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
                    if (
                        self.validation_config is not None
                        and self._validated_step != self.step_count
                    ):
                        self.validate()
                        self._step_metric_schedulers()
            finally:
                self._restore_requires_grad_filter()

    def _validate_single_process(self) -> None:
        """Reject a multi-rank launch the segment loop does not shard.

        The world size is read at run time rather than at construction because
        that is when a launcher has initialized the process group, and because
        an offline strategy the same script builds is free to be distributed.
        """
        world_size = get_world_size(self.distributed_manager)
        if world_size == 1:
            return
        raise ValueError(
            "On-policy distillation is single-process for now: each segment "
            "builds its own loader from a rank-local replay buffer and the "
            "seed state is not sharded, so every rank would propagate the same "
            "trajectories, pay the same teacher bill, and train on the same "
            f"frames. Got world_size={world_size!r}. Run the segment loop on "
            "one process, or distill offline — label the dataset with "
            "label_dataset and train the store with a DDPHook, which shards it "
            "as usual. Rank-sharded generation is planned."
        )

    def _close_interrupted_segment(self) -> None:
        """Count a segment a restored run stopped part-way through as finished.

        A checkpoint taken mid-segment — and an offline run graduating to the
        segment loop from a partial epoch — restores a nonzero
        ``epoch_step_count``, which the loop has no way to honor: each segment
        builds its own loader, the batches the interrupted segment had already
        drawn are gone with it, and the trajectory that produced them is
        reseeded anyway. Closing it here is what keeps the rest of the loop
        coherent: ``BEFORE_EPOCH`` fires for the resumed segment,
        ``epoch_step_count`` stays inside ``steps_per_segment``, and the
        mixture sampler advances past the epoch index the interrupted segment
        already drew with instead of redrawing its reference samples.

        The parent's :meth:`_prepare_epoch_step_count` is deliberately not used
        for this: it reconciles the restored counters against a fixed number of
        batches per epoch, which the graduation path — where the offline
        epochs were a different size — does not have.
        """
        if self.epoch_step_count == 0:
            return
        self.epoch_count += 1
        self.epoch_step_count = 0
        self._refresh_hook_counters()

    def _validation_checkpoint(self, stage: TrainingStage) -> bool:
        """Run a scheduled validation and remember the step it fired at.

        The segment loop closes with a terminal validation, which would
        otherwise repeat the pass an epoch cadence has just run at the same
        ``step_count`` and step every metric-driven scheduler a second time on
        identical metrics. Recording the step is what lets the closing block
        tell a cadence that already landed there from one that did not.

        Parameters
        ----------
        stage : TrainingStage
            Lifecycle stage that triggered this checkpoint.

        Returns
        -------
        bool
            Whether a validation pass ran at this checkpoint.
        """
        fired = super()._validation_checkpoint(stage)
        if fired:
            self._validated_step = self.step_count
        return fired

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
            seed=config.seed,
        )
        self._set_sampler_epoch(loader)
        primary_device = self.devices[0]
        consumed = 0
        for batch in loader:
            if consumed >= segment_steps or self.step_count >= target_step_count:
                break
            batch = _to_device(batch, primary_device)
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
                "The segment's mixed loader produced no batches before the "
                "target step count was reached; ensure reference_dataset and "
                "the replay buffer together hold at least one batch of "
                f"batch_size={config.batch_size!r} samples."
            )
        self.epoch_count += 1
        self.epoch_step_count = 0
        self._refresh_hook_counters()
        self._run_hooks(TrainingStage.AFTER_EPOCH, self._last_batch)
        self._validation_checkpoint(TrainingStage.AFTER_EPOCH)
        return training_started

    def _resolve_replay_device(
        self, config: OnPolicyConfig
    ) -> torch.device | str | None:
        """Return the device the segment loop stages generated frames on.

        Frames reach the buffer from a host-memory sink rather than from the
        propagator, so an unset ``replay_device`` means the reference dataset's
        device: the two mixture sources are collated into one batch before the
        strategy moves it, and only the anchor decides where that happens. A
        run with no anchor leaves them in host memory.

        The anchor's device is the one it actually emits on, measured from a
        batch when no declaration settles it — a composition declares no device
        at all, and a store opened without one declares an index-less ``cuda``
        that names whichever device is current. Reading the declaration alone
        would stage the buffer in host memory beside a CUDA-resident anchor and
        fail only once the first segment's loader collated them.
        """
        if config.replay_device is not None:
            return config.replay_device
        if self.reference_dataset is None:
            return None
        return _emitted_device(self.reference_dataset)

    def _resume_or_seed(self, config: OnPolicyConfig, buffer: ReplayBuffer) -> Batch:
        """Return the batch to propagate, resuming a checkpointed run when there is one.

        A restored checkpoint carries the trajectory the interrupted run had
        reached, the propagator's cumulative step count, and the frames already
        in its replay buffer, so the resumed run continues the same trajectory
        rather than starting a fresh one from the seeds. That is what makes the
        continuation exact for a propagator whose whole state is the batch and
        that counter — the built-in integrators, whose Langevin noise is drawn
        from a counter-based generator keyed on the step count. A propagator
        carrying internal state of its own is not continued that far: a
        relaxation optimizer's adaptive history lives outside the batch, so
        :class:`~nvalchemi.dynamics.optimizers.FIRE` re-initializes its
        timestep, its mixing coefficient, and its uphill counter from the
        constructor arguments and only the positions continue.

        The bundle's frames *are* the replay buffer as of the checkpoint, so
        they replace what the buffer holds rather than being appended to it.
        The buffer outlives a :meth:`run` call, and a strategy restored while
        still holding the frames it generated would otherwise carry the
        pre-checkpoint half of them twice: not a diversity loss, since the
        mixed loader draws with replacement, but a weighting skew toward the
        stale states — exactly backwards for an on-policy loop — on top of
        double the buffer memory and an eviction horizon reached a restart
        early.

        The bundle describes one rank's run, because
        :class:`~nvalchemi.training.hooks.CheckpointHook` writes the strategy
        checkpoint it rides in on rank zero alone. It is consumed only when
        that rank is the whole world at both ends of the restart, and dropped
        with a warning otherwise — see :meth:`_rank_local_restart_reason`.
        """
        restored = self._take_restart_state()
        if restored is None:
            return self._seed_state(config)
        reason = self._rank_local_restart_reason()
        if reason is not None:
            warnings.warn(
                f"The on-policy restart bundle is dropped: {reason} It holds "
                "one rank's trajectory and one rank's replay frames, and "
                "replaying those onto every rank would have every rank "
                "propagate rank zero's structures and train on rank zero's "
                "frames. This rank seeds afresh from its own share of the seed "
                "source with a cold replay buffer instead, so budget the first "
                "segments after the restart for refilling it: until they do, "
                "the mixture is drawn from the reference dataset alone.",
                UserWarning,
                stacklevel=2,
            )
            return self._seed_state(config)
        config.dynamics.step_count = int(restored["dynamics_step_count"])
        frames = restored.get("replay_frames")
        if frames is not None:
            buffer.clear()
            buffer.extend(_batch_from_state(frames))
        return _batch_from_state(restored["md_state"])

    def _rank_local_restart_reason(self) -> str | None:
        """Return why a rank-zero-only restart bundle cannot be consumed, or ``None``.

        Two worlds have to agree for the bundle to describe the whole run: the
        one it was written in and the one it is restored into. ``step_count``
        counts a rank's own optimizer steps while ``global_step_count``
        advances by the world size, so their ratio recovers the world the
        checkpoint was saved at without the bundle carrying a schema for it.

        Returns
        -------
        str | None
            A sentence naming the mismatch, or ``None`` when a single rank
            wrote the bundle and a single rank is restoring it.
        """
        world_size = get_world_size(self.distributed_manager)
        saved_world_size = (
            self.global_step_count // self.step_count if self.step_count > 0 else 1
        )
        if world_size > 1:
            return (
                f"the segment loop is resuming on world_size={world_size!r}, "
                "and the checkpoint it rides in is written by rank zero alone."
            )
        if saved_world_size > 1:
            return (
                f"it was written on world_size={saved_world_size!r} and is "
                "being restored on one rank, which would silently continue "
                "rank zero's trajectories and discard the rest."
            )
        return None

    def _take_restart_state(self) -> dict[str, Any] | None:
        """Return the on-policy bundle a restored checkpoint carried, once."""
        for hook in self.hooks:
            if isinstance(hook, _OnPolicyRestartHook):
                return hook.take()
        return None

    def _seed_state(self, config: OnPolicyConfig) -> Batch:
        """Return the batch the first segment propagates from.

        A ``sampler`` bin-packs the initial batch under its own size budget,
        from its own dataset — which is why the config takes it *instead of* a
        ``seed_dataset``. A ``seed_dataset`` is propagated whole as a single
        batch, which keeps the trajectory count explicit: it *is* the set of
        systems the run generates from, so size it to the device.

        Either way the batch enters the run carrying none of the propagator's
        bookkeeping, so the run installs its own. ``status`` and ``system_id``
        describe the run that wrote them, and a seed loaded from a store a
        dynamics sink filled — the obvious provenance for "relax these
        structures, then generate from the minima" — arrives holding whatever
        it graduated with.
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.step` freezes every graph
        whose ``status`` has reached ``exit_status``, so a stale one would run
        a segment that moves nothing and fills the buffer with copies of the
        seeds, reported as a normal run.
        """
        if config.sampler is not None:
            state = config.sampler.build_initial_batch()
        else:
            seeds = config.seed_dataset
            state = seeds.load_batches([list(range(len(seeds)))])[0]
        for key in BaseDynamics._bookkeeping_keys:
            if key in state:
                del state[key]
        return state

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
        asked once more for the step it just finished. Labeling is idempotent
        per step, so a cadence that did land there costs nothing and stores
        nothing twice.

        The hook's private entry point is called rather than the hook itself,
        because this is a *forced* label rather than a cadence dispatch, and the
        two are treated differently: a cadence firing on the step right after a
        forced label is passed over, so a ``segment_steps`` that is a multiple
        of ``label_frequency`` pays for one teacher pass per segment boundary
        instead of two on adjacent frames. Going through ``__call__`` would
        build a :class:`~nvalchemi.hooks._context.DynamicsContext` the hook
        reads two fields of and lose that distinction.
        """
        label_hook._label_frame(
            state, max(config.dynamics.step_count - 1, 0), forced=True
        )
        if label_hook.sink is not None and len(label_hook.sink) > 0:
            buffer.extend(label_hook.sink.drain())

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize declarative distillation knobs to a JSON-ready dict.

        The bundle names its own strategy class under ``strategy_cls``, the key
        :meth:`to_checkpoint_dict` writes with the same value, so a spec that
        travels alone still says which strategy rebuilds it.

        An on-policy run serializes too: ``on_policy`` becomes the recipe
        :meth:`~nvalchemi.training.distillation.OnPolicyConfig.to_spec_dict`
        produces — the propagator's spec, the scorer's signals, the seed
        store's path, and every scalar knob — and ``reference_dataset`` becomes
        the store it reads. Both are references rather than objects: the
        rebuilt strategy needs its models supplied, and a dataset that holds
        its samples in memory cannot be named at all.

        A piece the recipe cannot describe leaves the whole ``on_policy``
        entry out and says why, rather than writing a recipe that would rebuild
        into a different run. A strategy rebuilt from such a spec is
        offline-shaped and needs the on-policy objects supplied at
        construction.

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.

        Warns
        -----
        UserWarning
            If ``on_policy`` or ``reference_dataset`` holds something no recipe
            can describe, naming what it was.
        """
        spec = super().to_spec_dict()
        spec["strategy_cls"] = f"{type(self).__module__}.{type(self).__qualname__}"
        spec["teacher_signals"] = (
            None if self.teacher_signals is None else sorted(self.teacher_signals)
        )
        spec["label_missing"] = self.label_missing
        if self.on_policy is None:
            return spec
        try:
            spec["on_policy"] = self.on_policy.to_spec_dict(
                teacher=self.models["teacher"]
            )
            spec["reference_dataset"] = (
                None
                if self.reference_dataset is None
                else _dataset_spec_dict(
                    self.reference_dataset, "DistillationStrategy.reference_dataset"
                )
            )
        except ValueError as exc:
            warnings.warn(
                f"The on-policy recipe is omitted from the spec: {exc} A "
                "strategy rebuilt from this spec runs offline over the "
                "dataloader passed to run().",
                UserWarning,
                stacklevel=2,
            )
            spec.pop("on_policy", None)
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: Sequence[Any] | None = None,
        training_fn: Any = None,
        on_policy: OnPolicyConfig | None = None,
        reference_dataset: BatchDatasetProtocol | None = None,
        sampler: Any = None,
    ) -> DistillationStrategy:
        """Rebuild a :class:`DistillationStrategy` from ``to_spec_dict`` output.

        A spec carrying an ``on_policy`` recipe rebuilds the segment loop too:
        the propagator around the supplied student, the scorer around the
        supplied teacher, and the seed and reference datasets from the stores
        they name. Passing *on_policy* or *reference_dataset* overrides the
        recipe's own, which is how a run whose datasets live in memory — or
        whose propagator carries hooks — is restored.

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
        on_policy : OnPolicyConfig | None, optional
            Segment loop to use instead of the spec's recipe. Default ``None``
            (rebuild the recipe, when the spec carries one).
        reference_dataset : BatchDatasetProtocol | None, optional
            Anchor dataset to use instead of the one the recipe names. Default
            ``None``.
        sampler : SizeAwareSampler | None, optional
            Runtime seed sampler for a recipe that was serialized with one,
            which no spec can carry. Default ``None``.

        Returns
        -------
        DistillationStrategy
            A freshly validated distillation strategy ready to :meth:`run`.

        Raises
        ------
        ValueError
            If *spec* is missing a required key, if its ``strategy_cls`` entry
            is not a dotted class path string, or if that path resolves to a
            class that is not a :class:`DistillationStrategy` subclass.

        Notes
        -----
        The segment loop is resolved by a fixed precedence: an explicitly
        supplied *on_policy* wins, then a loop the caller registered for the
        restore, then the spec's own recipe. A recipe is the weakest source
        because it is the only one that cannot be complete — it names its seed
        store by path and its propagator by constructor arguments, so a loop
        the caller is already holding is the more faithful description of the
        run. A loop handed to this method or to
        :func:`~nvalchemi.training.load_checkpoint` must therefore be the one
        that runs, never quietly replaced by a describable recipe the
        checkpoint happens to carry.

        The corollary is that a spec resume cannot re-supply an in-memory seed
        set. A recipe refuses to describe an
        :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset` by
        design — there is no path to name it by — so a run seeded from one
        serializes without its ``on_policy`` block at all, and restoring it
        means passing *on_policy* here. A propagator carrying hooks takes the
        same route.
        """
        required = ("optimizer_configs", "devices", "loss_fn_spec")
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(
                f"from_spec_dict: spec is missing required key(s) {missing}. "
                f"Expected keys: {list(required)}."
            )
        raw_strategy_cls = spec.get("strategy_cls")
        if raw_strategy_cls is not None:
            if not isinstance(raw_strategy_cls, str):
                raise ValueError(
                    "from_spec_dict: 'strategy_cls' must be a dotted class path "
                    f"string; got {type(raw_strategy_cls).__name__}."
                )
            if not issubclass(_import_cls(raw_strategy_cls), cls):
                raise ValueError(
                    f"from_spec_dict: {raw_strategy_cls!r} must resolve to a "
                    f"{cls.__name__} subclass."
                )
        model_input = strategy_spec._models_from_spec_and_overrides(
            spec.get("model_specs", {}),
            models,
            single_model_input=strategy_spec._single_model_input_from_spec(
                spec.get("single_model_input")
            ),
        )
        recipe = spec.get("on_policy")
        rebuildable = isinstance(model_input, Mapping) and _REQUIRED_MODELS <= set(
            model_input
        )
        if on_policy is None and recipe is not None and rebuildable:
            on_policy = OnPolicyConfig.from_spec_dict(
                recipe,
                student=model_input["student"],
                teacher=model_input["teacher"],
                sampler=sampler,
            )
        anchor_spec = spec.get("reference_dataset")
        if reference_dataset is None and anchor_spec is not None:
            reference_dataset = _dataset_from_spec_dict(anchor_spec)
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
            on_policy=on_policy,
            reference_dataset=reference_dataset,
        )
