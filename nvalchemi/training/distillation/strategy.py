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

import dataclasses
import warnings
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import Field, PrivateAttr, model_validator

from nvalchemi._typing import ModelOutputs
from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks._context import DynamicsContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distillation._labels import (
    _TEACHER_FIELD_PREFIX,
    _attach_teacher_labels,
    _reject_foreign_fields,
)
from nvalchemi.training.distillation._seeding import (
    _check_seed_fields,
    _check_seed_status,
    _SeedSampler,
    _stamp_bookkeeping,
)
from nvalchemi.training.distillation.config import OnPolicyConfig
from nvalchemi.training.distillation.hooks import (
    TeacherLabelHook,
    _ConvergedFrameHook,
    _run_local_keys,
    _strip_replay_frame,
)
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
    from nvalchemi.training.losses.composition import ComposedLossFunction

__all__ = ["DistillationStrategy", "default_distillation_fn"]

_REQUIRED_MODELS = frozenset({"student", "teacher"})
"""Model names every distillation strategy must be given."""


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


@dataclasses.dataclass(frozen=True)
class _RelaxationLifecycle:
    """Machinery a relaxation segment loop drives between its segments."""

    capture: _ConvergedFrameHook
    sampler: SizeAwareSampler | _SeedSampler


def _refill_sampler(
    config: OnPolicyConfig, state: Batch
) -> SizeAwareSampler | _SeedSampler:
    """Return the source structures are backfilled from as others graduate.

    A configured :class:`~nvalchemi.dynamics.sampler.SizeAwareSampler` already
    holds the run's dataset, its own size budget, and the record of what the
    initial batch consumed, so it serves the backfill directly. A seed dataset
    is adapted instead, under the envelope of the batch it seeded.
    """
    if config.sampler is not None:
        return config.sampler
    return _SeedSampler(
        config.seed_dataset,
        consumed=state.num_graphs,
        recycle=config.recycle_seeds,
        max_atoms=int(state.num_nodes),
        max_batch_size=state.num_graphs,
    )


def _nested_propagators(dynamics: BaseDynamics) -> Iterator[BaseDynamics]:
    """Yield *dynamics* and every propagator composed inside it.

    A :class:`~nvalchemi.dynamics.FusedStage` holds its sub-stages as
    ``(code, dynamics)`` pairs and dispatches their hooks as well as its own,
    so anything reading the hooks of a propagator has to read theirs too.
    """
    yield dynamics
    for _, sub_stage in getattr(dynamics, "sub_stages", ()):
        yield from _nested_propagators(sub_stage)


def _competing_migrators(
    dynamics: BaseDynamics, criterion: ConvergenceHook
) -> list[ConvergenceHook]:
    """Return the status migrators already on *dynamics* that are not *criterion*.

    Both places a propagator can hold one are searched: its registered hooks,
    where a migrating :class:`~nvalchemi.dynamics.base.ConvergenceHook` fires
    every step, and its ``convergence_hook``, which the lifecycle is about to
    replace and whose migration would otherwise be dropped without a word.

    A :class:`~nvalchemi.dynamics.FusedStage` is searched sub-stage by
    sub-stage as well, because that is where its own migrators live:
    constructing one registers a migrating hook on every non-last sub-stage
    unconditionally, and on the last one whenever it declares a
    ``convergence_hook`` of its own. Those hooks fire ahead of the fused-level
    ones, so a scan of the fused propagator alone reports a clean propagator
    while the sub-stage migrator graduates the batch first. A sub-stage
    ``convergence_hook`` that only detects convergence is not a competitor
    itself — the migrator ``FusedStage`` derives from it is, and it is found
    among that sub-stage's hooks. The hooks registered at the fused level
    through ``register_fused_hook`` fire on the whole batch right behind the
    fused propagator's own, so they are read alongside them.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator the lifecycle is being installed on.
    criterion : ConvergenceHook
        The lifecycle's own criterion, which is not a competitor.

    Returns
    -------
    list[ConvergenceHook]
        The competing criteria, in the order they were found.
    """
    return [
        hook
        for propagator in _nested_propagators(dynamics)
        for hook in (
            *propagator.hooks,
            *getattr(propagator, "fused_hooks", ()),
            propagator.convergence_hook,
        )
        if isinstance(hook, ConvergenceHook)
        and hook is not criterion
        and hook.source_status is not None
        and hook.target_status is not None
    ]


@contextmanager
def _relaxation_lifecycle(
    config: OnPolicyConfig, state: Batch
) -> Iterator[_RelaxationLifecycle | None]:
    """Install the convergence machinery of a relaxation run on the propagator.

    The config's :attr:`~OnPolicyConfig.convergence_criterion` is put on the
    propagator twice, deliberately. As a registered ``AFTER_STEP`` hook it
    migrates the status of converged graphs, which is what freezes them in the
    propagator's step, what the capture hook behind it stores them on, and what
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` graduates them
    on; as the propagator's ``convergence_hook`` it is the detector ending a
    chunk early once every graph has converged. One criterion drives both,
    rather than a run whose graduation and detection disagree — a criterion the
    propagator was built with is restored on the way out, and so is ``done``,
    which :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` raises off
    the temporary refill sampler this context owns and would otherwise leave on
    a propagator the caller means to reuse.

    That is only true while it is the *sole* migrator, so a propagator already
    carrying one is refused rather than run: a looser criterion of its own
    graduates a structure before the configured one accepts it, which freezes
    it out of the path capture and leaves the converged route nothing to store,
    so the trajectory ends in neither. The criterion also has to migrate off
    the status the seeds are stamped with here, or nothing ever freezes and
    nothing ever graduates while the run reports itself configured.

    The lifecycle is likewise the run's sole refill source, so a propagator
    carrying a sampler of its own is refused too. That sampler makes
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` refill on its own
    cadence, mid segment, and the compaction that follows a graduation moves
    the survivors under the capture hook's positional bookkeeping, which then
    reads the wrong rows and stores neither the minima it is holding nor the
    ones still to come.

    Parameters
    ----------
    config : OnPolicyConfig
        Segment-loop configuration, holding the criterion.
    state : Batch
        Seed batch, stamped here with the fields the refill cycle maintains.

    Yields
    ------
    _RelaxationLifecycle | None
        The machinery the segment loop drives, or ``None`` for a config that
        manages no lifecycle.

    Raises
    ------
    ValueError
        If the propagator already carries a status-migrating criterion, if it
        carries a sampler of its own, or if the configured criterion migrates
        off a status no seed carries.
    """
    criterion = config.convergence_criterion
    if criterion is None:
        yield None
        return
    dynamics = config.dynamics
    competing = _competing_migrators(dynamics, criterion)
    if competing:
        migrations = [(hook.source_status, hook.target_status) for hook in competing]
        raise ValueError(
            "The relaxation lifecycle owns graduation for this run, so the "
            "propagator must carry no other status-migrating ConvergenceHook; "
            f"got {migrations!r} beside the configured "
            f"({criterion.source_status!r}, "
            f"{criterion.target_status!r}). A second migrator graduates "
            "structures at its own threshold, and one that graduates them "
            "before the configured criterion accepts them stores them by "
            "neither capture route. Remove it, or drop convergence and let the "
            "propagator manage its own lifecycle. On a FusedStage the migrator "
            "is one the stage built for a sub-stage rather than one the caller "
            "registered: every non-last sub-stage carries one, and the last "
            "one does whenever it was given a convergence_hook, so only a "
            "single sub-stage without its own criterion is free of them."
        )
    if dynamics.sampler is not None:
        raise ValueError(
            "The relaxation lifecycle owns the refill as well as graduation, "
            "so the propagator must carry no sampler of its own; got "
            f"{type(dynamics.sampler).__name__!r}. A propagator that refills "
            "inside run compacts the survivors to the front of the batch mid "
            "segment, which leaves the capture hook's positional bookkeeping "
            "pointing at the wrong structures and drops the minima it was "
            "meant to store. Pass it as OnPolicyConfig.sampler, which "
            "backfills from the same dataset at the segment boundary, and "
            "leave the propagator's own unset."
        )
    _stamp_bookkeeping(state)
    _check_seed_status(state, criterion)
    capture = _ConvergedFrameHook(sink=HostMemory(capacity=state.num_graphs))
    detector = dynamics.convergence_hook
    was_done = dynamics.done
    # Registered ahead of the capture and labeling hooks, so a graph that
    # converges on this step is graduated before either of them reads its
    # status and the two capture routes never store it twice.
    dynamics.register_hook(criterion)
    dynamics.register_hook(capture)
    dynamics.convergence_hook = criterion
    try:
        yield _RelaxationLifecycle(
            capture=capture, sampler=_refill_sampler(config, state)
        )
    finally:
        dynamics.convergence_hook = detector
        dynamics.done = was_done
        dynamics.hooks.remove(criterion)
        dynamics.hooks.remove(capture)


def _propagates_student(propagator_model: object, student: BaseModelMixin) -> bool:
    """Return whether *propagator_model* is *student* or a model composing it."""
    if propagator_model is student:
        return True
    modules = getattr(propagator_model, "modules", None)
    return callable(modules) and any(module is student for module in modules())


def _student_label_dtype(student: BaseModelMixin) -> torch.dtype | None:
    """Return the dtype teacher labels are cast to for *student*.

    The first floating-point parameter decides, whatever its precision: a
    ``bfloat16`` student gets ``bfloat16`` labels, since which dtypes a store
    can hold is the store's rule and
    :func:`~nvalchemi.training.distillation.label_dataset` checks it for the
    scorer it is handed. A student that exposes no parameters at all gets
    ``None``, which leaves labels in the teacher's own dtype.
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
    :class:`~nvalchemi.training.losses.base.LossWeightSchedule` — offline,
    where every sample carries its own reference labels. An on-policy run
    cannot: generated frames have no reference labels, and both mixture sources
    are required to carry the same fields, so its anchor has to be
    teacher-labeled rather than reference-labeled.

    Those targets also decide what the teacher is asked for.
    ``teacher_signals=None`` (the default) derives the signal set from the
    ``teacher_*`` targets the loss reads, so the two cannot drift apart; an
    explicit set must cover the derived one and may add more. The resolved set
    is checked against the teacher's declared outputs at construction, as is
    the model/optimizer contract above and — for the stock ``training_fn`` —
    every loss component's prediction key against the outputs the student
    actually computes, which is its ``active_outputs`` intersected with its
    declared ``outputs``, so a misconfigured run fails before it starts rather
    than on its first batch.

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
    identity with ``models["student"]`` at construction. A relaxation
    propagator adds ``OnPolicyConfig.convergence``: converged structures are
    stored once, graduate out of the batch at the segment boundary, and are
    replaced by fresh seeds, so the buffer keeps filling with structures that
    are still moving.

    Raises
    ------
    ValueError
        If ``models`` is not a named mapping containing ``"student"`` and
        ``"teacher"``, if the teacher is given an optimizer config, if the
        student or an auxiliary model is not, if a loss component reads a
        prediction the student does not compute, if the loss reads a
        ``teacher_*`` target that maps to no known signal and that no
        on-policy propagator's scorer declares, if an explicit
        ``teacher_signals`` omits a signal the loss needs, if no teacher signal
        is requested at all, if the teacher cannot produce a requested signal,
        or if the teacher is a composition that plans more than one
        neighbor-list source. In on-policy mode, additionally if the run is
        sized in epochs rather than steps, if the propagator holds neither the
        student nor a model composing it, if ``replay_ratio`` is ``0``, if a
        ratio below ``1`` is paired with no ``reference_dataset``, if a ratio
        of ``1`` is paired with one, if the ratio and ``batch_size`` together
        allocate no samples to one mixture source, if ``reference_dataset``
        emits on an accelerator the run does not train on, if
        ``replay_device`` names a device the ``reference_dataset`` does not
        emit on, if the ``reference_dataset`` carries fields the labeling hook
        strips from every generated frame, if the propagator's scorer declares
        a field outside the ``teacher_*`` namespace, or if that scorer's known
        fields and ``reference_dataset`` do not carry the same teacher fields.

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
    forward pass. Unlabeled validation data therefore needs no preparation, and
    a caller-supplied ``training_fn`` is covered too. Hooks are never
    serialized, so the seam is simply re-registered when :meth:`from_spec_dict`
    or :meth:`load_checkpoint` rebuilds the strategy.

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
    validation step establishes, and an on-the-fly label matches the offline
    one bit for bit.

    Labels are cast to the student's first floating-point parameter dtype,
    whatever its precision, so a float64 teacher feeds a float32 or
    ``bfloat16`` student without a dtype error at the loss. The cast is
    resolved at construction; a student whose dtype changes afterwards needs a
    ``dtype_policy`` on the loss terms.

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
        self._teacher_fields = signal_fields(signals)
        return self

    def _validate_student_outputs(self) -> None:
        """Check the loss's prediction keys against the student's effective outputs.

        The stock ``training_fn`` returns exactly what the student's forward
        emits, which is ``active_outputs`` intersected with ``outputs`` rather
        than the declared set, so a student whose active set is narrowed — the
        common default for a pretrained wrapper — is caught here instead of on
        its first batch.
        """
        if self.training_fn is not default_distillation_fn:
            return
        student = self.models["student"]
        declared = student.model_config.outputs
        active = student.output_data()
        for component in self.loss_fn.components:
            key = getattr(component, "prediction_key", None)
            if key is None:
                continue
            output = key.removeprefix("predicted_")
            if output in active:
                continue
            component_name = type(component).__name__
            if output in _EMBEDDING_KEYS:
                raise ValueError(
                    f"Loss component {component_name!r} reads prediction_key={key!r}, "
                    "which the stock training_fn cannot produce: embeddings come "
                    "from the student's compute_embeddings(), not from its forward "
                    "pass. Pass a training_fn that calls compute_embeddings and "
                    f"returns the embedding under {key!r}."
                )
            if output in declared:
                raise ValueError(
                    "Student declares but does not compute the output required by "
                    f"loss component {component_name!r} reading prediction_key="
                    f"{key!r}; got active_outputs={sorted(active)!r}, missing "
                    f"{output!r}. Add it to the student's "
                    "model_config.active_outputs."
                )
            raise ValueError(
                "Student cannot produce the output required by loss component "
                f"{component_name!r} reading prediction_key={key!r}; "
                f"got outputs={sorted(declared)!r}, missing {output!r}."
            )

    def _resolve_teacher_signals(self) -> frozenset[str]:
        """Return the signal set the loss needs, widened by an explicit request."""
        # Pydantic populates every field before the first mode="after"
        # validator, so the propagator's scorer is already readable here.
        supplied = (
            ()
            if self.on_policy is None
            else scorer_fields(self.on_policy.teacher_scorer) or ()
        )
        derived = _derived_teacher_signals(self.loss_fn, supplied_fields=supplied)
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
        self._validate_anchor_device()
        self._validate_mixture_device()
        self._validate_anchor_schema()
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

    def _validate_anchor_device(self) -> None:
        """Reject an anchor emitting on an accelerator the run does not train on."""
        if self.reference_dataset is None:
            return
        reference_device = _emitted_device(self.reference_dataset)
        primary = self.devices[0]
        if (
            reference_device is None
            or reference_device.type == "cpu"
            or _same_device(reference_device, primary)
        ):
            return
        raise ValueError(
            "A segment's mixture is collated on the reference dataset's own "
            "device before the strategy moves it, so an anchor that emits on "
            "an accelerator has to emit on the device the run trains on; got a "
            f"reference dataset emitting on {reference_device!s} and "
            f"devices[0]={primary!s}. A Zarr-backed Dataset resolves an unset "
            "device to CUDA whenever one is visible — open it as "
            f"Dataset(..., device={str(primary)!r}) to follow the run, or leave "
            "it in host memory."
        )

    def _validate_mixture_device(self) -> None:
        """Reject a staging device the reference dataset cannot be collated with."""
        if self.reference_dataset is None or self.on_policy.replay_device is None:
            return
        reference_device = _emitted_device(self.reference_dataset)
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

    def _validate_anchor_schema(self) -> None:
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
        """
        if self.reference_dataset is None:
            return
        dropped = _run_local_keys()
        probe = self.reference_dataset.load_batches([[0]])[0]
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
        configured and on-the-fly labels match
        :func:`~nvalchemi.training.distillation.label_dataset` exactly.

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

        An ``OnPolicyConfig.convergence`` criterion adds a fourth phase between
        generation and training, for the relaxation propagators whose
        trajectories end: *graduate and backfill* — converged structures are
        stored once as the minimum they reached, then leave the batch through
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` and are
        replaced by fresh seeds wherever the seed source still holds any.
        Generation stops when it runs dry and the last trajectory finishes, and
        the remaining steps train on the buffer already filled.

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
            rank, if a segment's loader produces no batches, if the seed
            structures lack a field the propagator opens its step with, if the
            propagator already carries a status-migrating criterion or a
            sampler of its own, or if the configured criterion migrates off a
            status no seed carries.

        Warns
        -----
        UserWarning
            If a lifecycle-managed run runs out of trajectories and seeds before
            reaching ``num_steps``, because the remaining steps then train on
            the frames already generated.

        Notes
        -----
        One segment is one epoch: ``AFTER_EPOCH`` fires at each segment
        boundary and an epoch-cadence ``validation_config`` follows the
        segments, while a step-cadence one fires inside them, exactly as in the
        offline loop. Validation data is labeled on the fly by the same
        ``BEFORE_FORWARD`` seam that labels training batches, and generated
        frames arrive pre-labeled, so that seam skips them. The buffer the
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
        replicate runs draw independently. Restarting an on-policy run
        mid-segment is not modeled — the propagator state is not checkpointed —
        so a resumed run continues from a freshly seeded trajectory.

        Because that loader is the loop's own, it is not rank-sharded, and
        neither is the seed state: the loop refuses to start in a distributed
        world of more than one rank rather than have every rank generate,
        label, and train on the same frames. Distributing the offline path is
        unaffected, and rank-sharded generation is planned.

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
        assumed to be ``segment_steps``. Prefer a bare propagator to a
        :class:`~nvalchemi.dynamics.FusedStage` here for the same reason:
        a fused stage fires a priming forward pass on every ``run``, so
        chunking one into segments pays that pass once per segment.

        A relaxation run is what that early exit exists for, and
        ``OnPolicyConfig.convergence`` is what turns it into a lifecycle. The
        criterion is registered on the propagator ahead of the labeling hook and
        installed as its detector for the duration of the loop, so a converged
        structure freezes in the propagator's step, is captured once at
        ``AFTER_STEP`` on the step its ``status`` reaches the propagator's
        ``exit_status`` — a transition rather than an ``ON_CONVERGE`` dispatch,
        which a :class:`~nvalchemi.dynamics.FusedStage` never makes on itself —
        and is left out of every later path capture of the segment instead of
        being stored again on each one. At the segment
        boundary those structures graduate through
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` and fresh
        seeds are appended in their place, where the seed source has any left. A
        ``sampler`` backfills from its own dataset under its own size budget,
        while a ``seed_dataset`` is propagated whole and therefore leaves its
        cursor past the last structure: a graduation narrows the batch instead,
        until ``recycle_seeds`` restarts the dataset from the beginning. The
        refill sampler is attached for that call alone, because the
        propagator's ``run`` only exits a chunk early while it holds none. Once
        no trajectory is left and no seed remains to start one, the loop warns
        and keeps training on the buffer it has until ``num_steps``. The two
        capture routes therefore partition a segment's frames rather than
        overlapping on any of them, and the converged ones are labeled in a
        single teacher pass as their sink is drained rather than one pass per
        convergence step.

        Note that generation and graduation move together only for a propagator
        whose trajectories end. A thermostat run never converges, which is
        exactly why ``convergence`` defaults to ``None`` and no lifecycle is
        managed unless it is set.
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
            self._apply_requires_grad_filter()
            try:
                primary_device = self.devices[0]
                flat_opts, flat_scheds = self._setup_runtime_optimizers(
                    rebuild=not self._resume_optimizer_state
                )
                state = self._seed_state(config).to(primary_device, non_blocking=True)
                buffer = ReplayBuffer(
                    capacity=config.replay_capacity,
                    eviction=config.replay_eviction,
                    device=self._resolve_replay_device(config),
                )
                self._replay_buffer = buffer
                label_hook = TeacherLabelHook(
                    config.teacher_scorer, frequency=config.label_frequency
                )
                with _relaxation_lifecycle(config, state) as lifecycle:
                    config.dynamics.register_hook(label_hook)
                    try:
                        # The teacher is frozen across both phases; the student
                        # sits in eval mode and is flipped to training mode by
                        # the inner context for the training phase only.
                        with (
                            freeze_unconfigured_models(
                                self.models, self.optimizer_configs
                            ),
                            _eval_configured_models(
                                self.models, self.optimizer_configs
                            ),
                            _eval_propagator_model(
                                config.dynamics.model, self.models["student"]
                            ),
                        ):
                            while self.step_count < target_step_count:
                                if state is not None:
                                    state = self._generate_segment(
                                        config, state, label_hook, lifecycle, buffer
                                    )
                                    if state is None:
                                        self._warn_generation_exhausted(
                                            config, target_step_count
                                        )
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

    def _generate_segment(
        self,
        config: OnPolicyConfig,
        state: Batch,
        label_hook: TeacherLabelHook,
        lifecycle: _RelaxationLifecycle | None,
        buffer: ReplayBuffer,
    ) -> Batch | None:
        """Propagate one segment, store what it produced, and refill the batch.

        Parameters
        ----------
        config : OnPolicyConfig
            Segment-loop configuration.
        state : Batch
            Batch this segment propagates from.
        label_hook : TeacherLabelHook
            Hook labeling and capturing the frames along the path.
        lifecycle : _RelaxationLifecycle | None
            Convergence machinery, or ``None`` when none is managed.
        buffer : ReplayBuffer
            Buffer the segment's frames are stored in.

        Returns
        -------
        Batch | None
            The batch the next segment propagates from, or ``None`` once every
            trajectory has finished and the seed source has nothing left to
            start a fresh one from.
        """
        # Sized per segment because a refill changes the trajectory count.
        label_hook.sink = HostMemory(
            capacity=(config.segment_steps + 1) * state.num_graphs
        )
        if lifecycle is not None:
            lifecycle.capture.sink = HostMemory(capacity=state.num_graphs)
        state = config.dynamics.run(state, n_steps=config.segment_steps)
        if lifecycle is not None:
            self._capture_budget_graduates(config, state, label_hook, lifecycle)
        self._capture_segment(config, state, label_hook, buffer)
        if lifecycle is None:
            return state
        self._capture_converged(config, lifecycle, buffer)
        return self._refill_segment(config, lifecycle, state)

    def _capture_budget_graduates(
        self,
        config: OnPolicyConfig,
        state: Batch,
        label_hook: TeacherLabelHook,
        lifecycle: _RelaxationLifecycle,
    ) -> None:
        """Store the structures a step budget graduated as the chunk ended.

        A :class:`~nvalchemi.dynamics.FusedStage` sub-stage that graduates on
        an ``n_steps`` budget rather than on a criterion migrates status after
        the fused ``AFTER_STEP`` dispatch, so the capture hook reads ``0`` on
        the step the budget runs out and the status is consistent only once
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` has returned. A
        budget that graduates every remaining graph ends the chunk on that step
        as well, so the segment behind it opens on a batch with nothing left
        moving and neither capture route ever reaches the frame.

        Which is why this runs before the segment's closing dispatch rather
        than after it: that dispatch labels a subset still moving and marks the
        step as covered, and this one has to read the marker as the propagator
        left it. The marker is the idempotence guard, because the path route
        stores a whole frame only while nothing has graduated yet — exactly the
        status the budget migration hid behind — so a step it already stored
        needs no second capture and a re-dispatch would only duplicate it. The
        capture hook's own record covers the other direction, keeping a
        criterion's graduates from being written twice.
        """
        last_step = max(config.dynamics.step_count - 1, 0)
        if label_hook.labeled_step == last_step:
            return
        lifecycle.capture(
            DynamicsContext(
                batch=state, step_count=last_step, workflow=config.dynamics
            ),
            DynamicsStage.AFTER_STEP,
        )

    def _capture_converged(
        self,
        config: OnPolicyConfig,
        lifecycle: _RelaxationLifecycle,
        buffer: ReplayBuffer,
    ) -> None:
        """Label the structures that converged this segment and store them.

        This is the deferred half of on-policy labeling. Converged frames are
        captured raw, at the step each structure reached its minimum, and the
        teacher sees them here in one pass over the whole segment's graduates
        rather than one pass per convergence step — which is what decouples the
        teacher's batch size from the propagated one. They are stripped to the
        replay-frame contract afterwards, so they enter the buffer under the
        same schema the path frames froze it with, and staged back onto the
        buffer's own device, which the path route left in host memory when the
        run has no anchor to follow.
        """
        sink = lifecycle.capture.sink
        if len(sink) == 0:
            return
        frames = sink.drain().to(self.devices[0], non_blocking=True)
        _attach_teacher_labels(frames, config.teacher_scorer.label(frames))
        buffer.extend(_strip_replay_frame(frames).to(buffer.device or "cpu"))

    def _refill_segment(
        self,
        config: OnPolicyConfig,
        lifecycle: _RelaxationLifecycle,
        state: Batch,
    ) -> Batch | None:
        """Graduate the converged structures and backfill fresh seeds.

        The sampler is attached for this call alone.
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` cuts a chunk short
        once every graph has converged, but only while no sampler is
        configured, and that early exit is exactly the signal that a refill is
        due — leaving the sampler attached for the whole loop would trade it
        for segments spent propagating frozen structures.

        A replacement arrives holding whatever its source stored it with, and
        ``refill_check`` deliberately preserves that, so the run installs its
        own bookkeeping over the rows the backfill appended — the same
        invariant the seed batch enters under, completed here. The one field
        kept is the ``system_id`` the sampler handed out, which is the sampler's
        to number. Anything else a source carried is the record of the run that
        wrote it: a seed store filled by a relaxation holds ``status`` at the
        code its structures graduated on, and a replacement arriving frozen is
        never propagated, stored raw as a minimum it never reached, and
        graduated again at the next boundary.

        Returns
        -------
        Batch | None
            The refilled batch, or ``None`` once nothing is left to propagate,
            which is what ``refill_check`` itself returns in that case — the
            ``done`` flag it raises alongside outlives the sampler it was
            derived from and is not read here.
        """
        dynamics = config.dynamics
        survivors = int((state["status"].view(-1) < dynamics.exit_status).sum())
        previous = dynamics.sampler
        dynamics.sampler = lifecycle.sampler
        try:
            refilled = dynamics.refill_check(state, dynamics.exit_status)
        finally:
            dynamics.sampler = previous
        if refilled is state:
            return refilled
        lifecycle.capture.reset()
        if refilled is not None:
            fresh = refilled.num_graphs - survivors
            for key, default_fn in dynamics._bookkeeping_keys.items():
                if key != "system_id":
                    refilled[key][survivors:] = default_fn(fresh, refilled.device)
        return refilled

    def _warn_generation_exhausted(
        self, config: OnPolicyConfig, target_step_count: int
    ) -> None:
        """Announce that the run trains on what it has already generated."""
        warnings.warn(
            "Every generated trajectory has finished and the seed source has "
            "nothing left to start a fresh one from, so generation stopped "
            f"after {config.dynamics.step_count} propagator steps with "
            f"{len(self._replay_buffer)} frames in the replay buffer; the "
            f"remaining {target_step_count - self.step_count} training steps "
            "draw from that buffer. Set recycle_seeds=True to keep generating "
            "from the beginning of the seed dataset, or pass more seed "
            "structures — a seed_dataset is propagated whole, so more of them "
            "lengthen the run by widening the initial batch rather than by "
            "backfilling it.",
            UserWarning,
            stacklevel=2,
        )

    def _resolve_replay_device(
        self, config: OnPolicyConfig
    ) -> torch.device | str | None:
        """Return the device the segment loop stages generated frames on.

        Frames reach the buffer from a host-memory sink rather than from the
        propagator, so an unset ``replay_device`` means the reference dataset's
        device: the two mixture sources are collated into one batch before the
        strategy moves it, and only the anchor decides where that happens. A
        run with no anchor leaves them in host memory.
        """
        if config.replay_device is not None:
            return config.replay_device
        if self.reference_dataset is None:
            return None
        return _emitted_device(self.reference_dataset)

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

        The structures are then checked against what the propagator reads
        before its first force evaluation, because a missing ``velocities`` or
        ``forces`` field surfaces from inside a kernel otherwise.

        Raises
        ------
        ValueError
            If the seed structures lack a field the propagator opens its step
            with.
        """
        if config.sampler is not None:
            state = config.sampler.build_initial_batch()
        else:
            seeds = config.seed_dataset
            state = seeds.load_batches([list(range(len(seeds)))])[0]
        for key in BaseDynamics._bookkeeping_keys:
            if key in state:
                del state[key]
        _check_seed_fields(state, config.dynamics)
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
