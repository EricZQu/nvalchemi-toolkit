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
"""Accuracy, teacher-consistency, and non-conservative diagnostics for students.

Two evaluations live here. :func:`evaluate_accuracy` runs a student over a
held-out set and reports energy, force, and stress errors against either a
reference dataset's own labels or a teacher's, together with the
teacher-consistency diagnostics that only make sense against a teacher.
:func:`nonconservative_residual` measures the part of a teacher's force field
no conservative student can fit, which is the floor the first evaluation is
read against.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch

from nvalchemi.data import Batch
from nvalchemi.training._validation import ValidationConfig, ValidationLoop
from nvalchemi.training.distillation._labels import _attach_teacher_labels
from nvalchemi.training.distillation.scoring import (
    InProcessTeacherScorer,
    TeacherScorer,
)
from nvalchemi.training.distributed import all_reduce, is_distributed_initialized
from nvalchemi.training.losses.composition import (
    ComposedLossFunction,
    as_composed_loss,
)
from nvalchemi.training.losses.terms import (
    EnergyMSELoss,
    ForceMSELoss,
    StressMSELoss,
)
from nvalchemi.training.strategy import default_training_fn

if TYPE_CHECKING:
    from collections.abc import Callable

    from nvalchemi.models.base import BaseModelMixin

__all__ = [
    "AccuracyMetrics",
    "AccuracyQuantity",
    "NonConservativeResidual",
    "evaluate_accuracy",
    "nonconservative_residual",
]

AccuracyQuantity: TypeAlias = Literal["energy", "forces", "stress", "atomic_energies"]
"""Quantity an accuracy evaluation compares between a student and a target."""

_TEACHER_TARGET_KEYS: dict[str, str] = {
    "energy": "teacher_energy",
    "forces": "teacher_forces",
    "stress": "teacher_stress",
    "atomic_energies": "teacher_node_energies",
}
"""Batch field a teacher scorer writes, keyed by quantity."""

_REFERENCE_TARGET_KEYS: dict[str, str] = {
    "energy": "energy",
    "forces": "forces",
    "stress": "stress",
    "atomic_energies": "atomic_energies",
}
"""Batch field a reference dataset carries, keyed by quantity."""

_PREDICTION_KEYS: dict[str, str] = {
    "energy": "predicted_energy",
    "forces": "predicted_forces",
    "stress": "predicted_stress",
    "atomic_energies": "predicted_atomic_energies",
}
"""Prediction key :func:`default_training_fn` publishes, keyed by quantity."""

_SUPERVISED_QUANTITIES = ("energy", "forces", "stress")
"""Quantities a built-in loss term supervises, in report order."""

_DEFAULT_QUANTITIES: tuple[AccuracyQuantity, ...] = ("energy", "forces")
"""Quantities evaluated when a caller names none."""

_EPS = 1e-12
"""Denominator guard for direction normalization and cosine similarity."""


@dataclasses.dataclass(frozen=True)
class AccuracyMetrics:
    """Errors of one student against one set of targets over a held-out set.

    Every metric is an exact global reduction over the evaluated set — the sum
    of residuals divided by the total count, not a mean of per-batch means — so
    the value does not depend on how the data was batched. Errors are reported
    in the units the batch carries them in: eV for energies, eV/A for forces,
    and the stress units of the dataset.

    Fields are ``None`` for quantities the pass could not measure, either
    because they were not requested or because a batch carried no such
    prediction or target.

    Attributes
    ----------
    name : str
        Label carried into reports.
    num_graphs : int
        Number of graphs evaluated.
    num_atoms : int
        Number of atoms evaluated.
    energy_mae, energy_rmse : float | None
        Total-energy error per graph.
    energy_per_atom_mae, energy_per_atom_rmse : float | None
        Total-energy error divided by each graph's atom count.
    forces_mae, forces_rmse : float | None
        Force error per Cartesian component, averaged over every component of
        every atom.
    stress_mae, stress_rmse : float | None
        Stress error per component, averaged over all nine components.
    force_cosine_mean : float | None
        Mean over atoms of the cosine similarity between the predicted and
        target force vectors.
    force_cosine_aggregate : float | None
        Cosine similarity of the two force fields taken as single vectors over
        the whole evaluated set, which weights atoms by force magnitude
        instead of equally.
    atomic_energy_mae, atomic_energy_rmse : float | None
        Per-atom energy residual, populated only when both sides publish an
        atomic energy decomposition.
    """

    name: str
    num_graphs: int
    num_atoms: int
    energy_mae: float | None = None
    energy_rmse: float | None = None
    energy_per_atom_mae: float | None = None
    energy_per_atom_rmse: float | None = None
    forces_mae: float | None = None
    forces_rmse: float | None = None
    stress_mae: float | None = None
    stress_rmse: float | None = None
    force_cosine_mean: float | None = None
    force_cosine_aggregate: float | None = None
    atomic_energy_mae: float | None = None
    atomic_energy_rmse: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the populated fields as a plain dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if getattr(self, field.name) is not None
        }


@dataclasses.dataclass(frozen=True)
class NonConservativeResidual:
    r"""Non-conservative component of a teacher's force field.

    A force field decomposes into a conservative part and a remainder,
    :math:`F = -\nabla E + F_{\perp}`, and a student that predicts forces as
    the gradient of an energy can represent only the first term. The work
    integral of the first term around any closed path vanishes, so a closed-path
    integral of the teacher's field measures :math:`F_{\perp}` alone.

    Attributes
    ----------
    num_probes : int
        Number of closed loops integrated, counting each graph of each batch
        separately.
    amplitude : float
        Loop side length in A.
    segments : int
        Midpoint-rule samples per side.
    loop_work_mean_abs, loop_work_max_abs : float
        Mean and maximum over probes of the absolute work accumulated around
        one closed loop, in eV.
    force_floor, force_floor_max : float
        Mean and maximum over probes of ``|W| / L``, the loop work divided by
        the loop's path length, in eV/A.
    force_rms : float
        Root-mean-square teacher force magnitude at the loop centers, in eV/A.
    relative_floor : float
        ``force_floor`` divided by ``force_rms``.
    """

    num_probes: int
    amplitude: float
    segments: int
    loop_work_mean_abs: float
    loop_work_max_abs: float
    force_floor: float
    force_floor_max: float
    force_rms: float
    relative_floor: float

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dictionary."""
        return dataclasses.asdict(self)


class _ScoredBatches:
    """Re-iterable view that moves each batch to *device* and labels it.

    Labeling happens after the device move rather than inside
    :meth:`ValidationLoop.execute`, so a teacher evaluating a CPU-resident
    dataset on GPU runs where the student does.
    """

    def __init__(
        self, source: Iterable[Batch], scorer: TeacherScorer, device: torch.device
    ) -> None:
        self.source = source
        self.scorer = scorer
        self.device = device

    def __iter__(self) -> Iterator[Batch]:
        """Yield each source batch with the scorer's teacher fields attached."""
        for batch in self.source:
            placed = batch.to(self.device, non_blocking=True)
            _attach_teacher_labels(placed, self.scorer.label(placed))
            yield placed


class _MetricAccumulator:
    """Exact residual sums over a validation pass, as a per-batch callback.

    Implements :class:`~nvalchemi.training.BatchValidationCallback`, so the
    accumulator rides along on a :class:`ValidationLoop` pass and sees the
    predictions the loop already computed. Sums are kept as float64 device
    tensors and reduced once at the end, so no metric forces a
    host synchronization per batch.
    """

    def __init__(
        self,
        device: torch.device,
        quantities: Sequence[str],
        target_keys: Mapping[str, str],
    ) -> None:
        self.device = device
        self.quantities = tuple(quantities)
        self.target_keys = dict(target_keys)
        self._sums: dict[str, torch.Tensor] = {}

    def __call__(
        self,
        *,
        batch: Batch,
        predictions: Mapping[str, torch.Tensor],
        loss: Any,  # noqa: ARG002
        batch_count: int,  # noqa: ARG002
        step_count: int,  # noqa: ARG002
        epoch: int,  # noqa: ARG002
    ) -> None:
        """Accumulate one validation batch's residual sums."""
        self._add("num_graphs", batch.num_graphs)
        self._add("num_atoms", batch.num_nodes)
        for quantity in self.quantities:
            prediction = predictions.get(_PREDICTION_KEYS[quantity])
            target = getattr(batch, self.target_keys[quantity], None)
            if prediction is None or target is None:
                continue
            if quantity == "atomic_energies":
                self._accumulate(
                    quantity, prediction.reshape(-1), target.reshape(-1), target.numel()
                )
                continue
            self._accumulate(quantity, prediction, target, target.numel())
            if quantity == "energy":
                counts = batch.num_nodes_per_graph.reshape(
                    (-1,) + (1,) * (target.ndim - 1)
                )
                self._accumulate(
                    "energy_per_atom",
                    prediction / counts,
                    target / counts,
                    target.shape[0],
                )
            elif quantity == "forces":
                self._accumulate_cosine(prediction, target)

    def _add(self, key: str, value: torch.Tensor | float) -> None:
        """Add one contribution to the running float64 sum *key*."""
        tensor = value.detach() if isinstance(value, torch.Tensor) else value
        scalar = torch.as_tensor(tensor, device=self.device, dtype=torch.float64)
        previous = self._sums.get(key)
        self._sums[key] = scalar if previous is None else previous + scalar

    def _accumulate(
        self, name: str, prediction: torch.Tensor, target: torch.Tensor, count: int
    ) -> None:
        """Add the absolute and squared residual sums of one quantity.

        Shapes must match exactly; broadcasting a ``(B,)`` target against a
        ``(B, 1)`` prediction would silently measure every pairing.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target of {name!r} must have the same shape; got "
                f"{tuple(prediction.shape)!r} and {tuple(target.shape)!r}."
            )
        residual = prediction.detach().to(torch.float64) - target.detach().to(
            torch.float64
        )
        self._add(f"{name}_abs", residual.abs().sum())
        self._add(f"{name}_sq", residual.pow(2).sum())
        self._add(f"{name}_count", float(count))

    def _accumulate_cosine(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Add the per-atom and aggregate force-alignment sums."""
        predicted = prediction.detach().to(torch.float64)
        reference = target.detach().to(torch.float64)
        dot = (predicted * reference).sum(dim=-1)
        norms = predicted.norm(dim=-1) * reference.norm(dim=-1)
        self._add("force_cosine_sum", (dot / norms.clamp_min(_EPS)).sum())
        self._add("force_cosine_count", float(dot.numel()))
        self._add("force_dot", dot.sum())
        self._add("force_predicted_sq", predicted.pow(2).sum())
        self._add("force_target_sq", reference.pow(2).sum())

    def metrics(self, *, name: str, distributed_manager: Any | None) -> AccuracyMetrics:
        """Return the reduced metrics, all-reducing sums under distributed runs.

        Raises
        ------
        ValueError
            If no quantity was measured, which means every batch was missing
            either the prediction or the target of every requested quantity.
        """
        keys = tuple(sorted(self._sums))
        packed = torch.stack([self._sums[key] for key in keys])
        if is_distributed_initialized(distributed_manager):
            all_reduce(packed, distributed_manager)
        totals = {key: float(packed[index]) for index, key in enumerate(keys)}
        if not any(key.endswith("_count") for key in totals):
            raise ValueError(
                "No accuracy metric could be measured; every batch was missing "
                f"the prediction or the target of every requested quantity "
                f"{list(self.quantities)!r} (targets {self.target_keys!r})."
            )
        energy_mae, energy_rmse = _mae_rmse(totals, "energy")
        per_atom_mae, per_atom_rmse = _mae_rmse(totals, "energy_per_atom")
        forces_mae, forces_rmse = _mae_rmse(totals, "forces")
        stress_mae, stress_rmse = _mae_rmse(totals, "stress")
        atomic_mae, atomic_rmse = _mae_rmse(totals, "atomic_energies")
        return AccuracyMetrics(
            name=name,
            num_graphs=int(totals.get("num_graphs", 0.0)),
            num_atoms=int(totals.get("num_atoms", 0.0)),
            energy_mae=energy_mae,
            energy_rmse=energy_rmse,
            energy_per_atom_mae=per_atom_mae,
            energy_per_atom_rmse=per_atom_rmse,
            forces_mae=forces_mae,
            forces_rmse=forces_rmse,
            stress_mae=stress_mae,
            stress_rmse=stress_rmse,
            force_cosine_mean=_ratio(
                totals.get("force_cosine_sum"), totals.get("force_cosine_count")
            ),
            force_cosine_aggregate=_aggregate_cosine(totals),
            atomic_energy_mae=atomic_mae,
            atomic_energy_rmse=atomic_rmse,
        )


def _mae_rmse(
    totals: Mapping[str, float], prefix: str
) -> tuple[float | None, float | None]:
    """Return the MAE and RMSE of one quantity, or two ``None`` when unmeasured."""
    count = totals.get(f"{prefix}_count", 0.0)
    if count <= 0.0:
        return None, None
    return totals[f"{prefix}_abs"] / count, math.sqrt(totals[f"{prefix}_sq"] / count)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return ``numerator / denominator``, or ``None`` when either is missing."""
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _aggregate_cosine(totals: Mapping[str, float]) -> float | None:
    """Return the cosine similarity of the two force fields taken as one vector."""
    dot = totals.get("force_dot")
    if dot is None:
        return None
    norm = math.sqrt(totals["force_predicted_sq"] * totals["force_target_sq"])
    return dot / norm if norm > 0.0 else None


def _resolve_device(model: Any, device: torch.device | str | None) -> torch.device:
    """Return the requested device, else the device the model's parameters sit on."""
    if device is not None:
        return torch.device(device)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            return parameter.device
    return torch.device("cpu")


def _as_scorer(model: TeacherScorer | BaseModelMixin, signals: Sequence[str]) -> Any:
    """Return *model* as a scorer, wrapping a bare model in an in-process one."""
    if isinstance(model, TeacherScorer):
        missing = sorted(set(signals) - set(model.signals))
        if missing:
            raise ValueError(
                f"Scorer must produce the signals {list(signals)!r} this evaluation "
                f"reads; got {sorted(model.signals)!r}, missing {missing!r}."
            )
        return model
    return InProcessTeacherScorer(model, signals)


def _metric_loss(
    quantities: Sequence[str], target_keys: Mapping[str, str]
) -> ComposedLossFunction:
    """Build the composed loss whose gradient requirement drives the pass."""
    terms = []
    if "energy" in quantities:
        terms.append(EnergyMSELoss(target_key=target_keys["energy"], per_atom=True))
    if "forces" in quantities:
        terms.append(ForceMSELoss(target_key=target_keys["forces"]))
    if "stress" in quantities:
        terms.append(StressMSELoss(target_key=target_keys["stress"]))
    if not terms:
        raise ValueError(
            "At least one of 'energy', 'forces', or 'stress' must be evaluated so "
            f"the validation pass has a loss to run; got {list(quantities)!r}."
        )
    return as_composed_loss(ComposedLossFunction(terms))


def evaluate_accuracy(
    model: BaseModelMixin,
    data: Iterable[Batch],
    *,
    targets: Literal["reference", "teacher"] = "reference",
    quantities: Sequence[AccuracyQuantity] | None = None,
    scorer: TeacherScorer | BaseModelMixin | None = None,
    target_keys: Mapping[str, str] | None = None,
    loss_fn: ComposedLossFunction | None = None,
    validation_fn: Callable[..., Any] = default_training_fn,
    grad_mode: Literal["auto", "enabled", "disabled"] = "auto",
    device: torch.device | str | None = None,
    distributed_manager: Any | None = None,
    name: str = "accuracy",
) -> AccuracyMetrics:
    """Measure a student's error over a held-out set.

    The pass itself runs through :class:`~nvalchemi.training.ValidationLoop`, so
    eval mode, the autograd policy an autograd-force student needs, autocast,
    and device placement behave exactly as they do during training validation.
    The metrics are accumulated separately, as exact global residual sums, and
    the loop's own loss value is discarded: a loss is a training objective with
    its own graph balancing, while an evaluation wants the plain per-atom and
    per-component errors a paper reports.

    Point *targets* at ``"reference"`` to compare against the dataset's own
    labels — the DFT holdout — and at ``"teacher"`` to compare against the
    ``teacher_*`` fields, whether they were written offline by
    :func:`~nvalchemi.training.distillation.label_dataset` or on the fly by a
    *scorer* passed here. Against a teacher the force-alignment and per-atom
    energy diagnostics fill in as well, since both sides then describe the same
    decomposition.

    Parameters
    ----------
    model : BaseModelMixin
        Student to evaluate. Left in the training mode it arrived in.
    data : Iterable[Batch]
        Re-iterable holdout set. One-shot iterators are rejected.
    targets : {"reference", "teacher"}, optional
        Which family of batch fields to compare against. Default
        ``"reference"``.
    quantities : Sequence[AccuracyQuantity] | None, optional
        Quantities to evaluate. ``"atomic_energies"`` is a diagnostic only and
        never enters the pass's loss. Default ``None`` (energy and forces).
    scorer : TeacherScorer | BaseModelMixin | None, optional
        Teacher used to label each batch before it is evaluated. A bare model
        is wrapped in an
        :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` for
        the requested quantities. Default ``None`` (the batches are used as
        they arrive).
    target_keys : Mapping[str, str] | None, optional
        Per-quantity overrides of the batch field to compare against, applied
        on top of the map *targets* selects. Default ``None``.
    loss_fn : ComposedLossFunction | None, optional
        Loss driving the pass. Default ``None`` (mean-squared terms over the
        requested supervised quantities).
    validation_fn : Callable, optional
        Forward callable invoked as ``validation_fn(model, batch)``. Default
        :func:`~nvalchemi.training.default_training_fn`.
    grad_mode : {"auto", "enabled", "disabled"}, optional
        Autograd policy. ``"auto"`` enables gradients when the loss needs them,
        which is what lets an autograd-force student be evaluated at all.
        Default ``"auto"``.
    device : torch.device | str | None, optional
        Device the pass runs on. Default ``None`` (the model's own device).
    distributed_manager : Any | None, optional
        Manager used to all-reduce the metric sums. Default ``None``.
    name : str, optional
        Label stored on the result. Default ``"accuracy"``.

    Returns
    -------
    AccuracyMetrics
        Errors and consistency diagnostics over the whole set.

    Raises
    ------
    ValueError
        If *quantities* names an unknown quantity, if no supervised quantity is
        requested, if a prediction and its target disagree on shape, or if no
        metric could be measured at all.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import evaluate_accuracy
    >>> metrics = evaluate_accuracy(student, holdout)  # doctest: +SKIP
    >>> metrics.forces_mae  # doctest: +SKIP
    0.031

    Against the teacher, labeling on the fly:

    >>> metrics = evaluate_accuracy(  # doctest: +SKIP
    ...     student,
    ...     holdout,
    ...     targets="teacher",
    ...     scorer=teacher,
    ...     quantities=("energy", "forces", "atomic_energies"),
    ... )

    Notes
    -----
    The student is called through *validation_fn* exactly as a training loop
    would call it, so a student that reads a neighbor list needs batches that
    carry one — built with
    :func:`~nvalchemi.neighbors.compute_neighbors`, produced by the loader, or
    assembled by a composed model pipeline. A *scorer* has no such requirement:
    it builds and rolls back the teacher's own list per batch.

    Under a distributed run every rank must evaluate the same quantities, since
    the metric sums are packed into one tensor in a shared key order before the
    all-reduce. Ranks that saw different quantities would pack different
    tensors and deadlock.
    """
    requested = tuple(quantities) if quantities is not None else _DEFAULT_QUANTITIES
    unknown = sorted(set(requested) - set(_PREDICTION_KEYS))
    if unknown:
        raise ValueError(
            f"Accuracy quantities must be names from {sorted(_PREDICTION_KEYS)!r}; "
            f"got unsupported {unknown!r}."
        )
    base = _TEACHER_TARGET_KEYS if targets == "teacher" else _REFERENCE_TARGET_KEYS
    resolved_keys = dict(base) | dict(target_keys or {})
    supervised = [
        quantity for quantity in _SUPERVISED_QUANTITIES if quantity in requested
    ]
    resolved_device = _resolve_device(model, device)

    evaluation_data: Iterable[Batch] = data
    if scorer is not None:
        signals = [
            "node_energies" if quantity == "atomic_energies" else quantity
            for quantity in requested
        ]
        evaluation_data = _ScoredBatches(
            data, _as_scorer(scorer, signals), resolved_device
        )

    accumulator = _MetricAccumulator(resolved_device, requested, resolved_keys)
    config = ValidationConfig(
        validation_data=evaluation_data,
        loss_fn=loss_fn or _metric_loss(supervised, resolved_keys),
        grad_mode=grad_mode,
        batch_callback=accumulator,
        name=name,
    )
    loop = ValidationLoop(
        validation_data=evaluation_data,
        config=config,
        device=resolved_device,
        model=model,
        validation_fn=validation_fn,
        distributed_manager=distributed_manager,
    )
    with loop:
        loop.execute()
    return accumulator.metrics(name=name, distributed_manager=distributed_manager)


@contextmanager
def _displaced(batch: Batch, positions: torch.Tensor) -> Iterator[None]:
    """Swap *positions* onto *batch* for the block, restoring the originals after."""
    original = batch.positions
    batch.positions = positions
    try:
        yield
    finally:
        batch.positions = original


def _per_graph_sum(values: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Sum a per-atom scalar into one value per graph."""
    totals = values.new_zeros(batch.num_graphs)
    return totals.index_add_(0, batch.batch_idx, values)


def _probe_directions(
    batch: Batch, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two per-graph orthonormal displacement directions for *batch*.

    Each direction has unit Frobenius norm within each graph, so a loop built
    from them has the same side length in every graph of a mixed-size batch.
    """
    positions = batch.positions
    device = positions.device if generator is None else generator.device
    shape = tuple(positions.shape)
    first = torch.randn(shape, generator=generator, device=device).to(positions)
    second = torch.randn(shape, generator=generator, device=device).to(positions)
    index = batch.batch_idx
    overlap = _per_graph_sum((first * second).sum(dim=-1), batch)
    norm = _per_graph_sum(first.pow(2).sum(dim=-1), batch)
    second = second - (overlap / norm.clamp_min(_EPS))[index].unsqueeze(-1) * first
    first = first / norm.sqrt().clamp_min(_EPS)[index].unsqueeze(-1)
    second_norm = _per_graph_sum(second.pow(2).sum(dim=-1), batch)
    second = second / second_norm.sqrt().clamp_min(_EPS)[index].unsqueeze(-1)
    return first, second


def nonconservative_residual(
    teacher: TeacherScorer | BaseModelMixin,
    data: Iterable[Batch] | Batch,
    *,
    num_loops: int = 4,
    amplitude: float = 0.05,
    segments: int = 4,
    generator: torch.Generator | None = None,
) -> NonConservativeResidual:
    r"""Estimate the part of a teacher's force field no student can fit.

    A student that differentiates an energy produces a curl-free field, so it
    can only ever fit the conservative part of a teacher trained to emit forces
    directly. This probe measures the rest, without assuming the teacher is
    differentiable and without training anything.

    **The estimator.** Around each graph, two per-graph orthonormal
    displacement directions :math:`u` and :math:`v` span a rectangular loop of
    side *amplitude* :math:`\varepsilon` through configuration space, from
    :math:`R` to :math:`R + \varepsilon u` to
    :math:`R + \varepsilon u + \varepsilon v` to :math:`R + \varepsilon v` and
    back. The teacher's work around that closed loop,
    :math:`W = \oint F \cdot \mathrm{d}R`, is integrated with the midpoint rule
    using *segments* samples per side. For a conservative field the integrand
    is :math:`-\nabla E \cdot \mathrm{d}R` and :math:`W` vanishes identically,
    so what the probe reports is the non-conservative component alone.

    **The floor.** A conservative student makes force error
    :math:`\Delta F = F - F_{\text{student}}` with
    :math:`\oint \Delta F \cdot \mathrm{d}R = W`, and Cauchy-Schwarz then gives
    :math:`\max \lVert \Delta F \rVert \ge |W| / L` along that loop, where
    :math:`L = 4\varepsilon` is its path length. That ratio is reported as
    ``force_floor``.

    **What it does not measure.** The floor is a lower bound on the *largest*
    force error along a probed loop at a probed displacement scale, not a bound
    on the error averaged over a dataset, and it shrinks linearly with
    *amplitude* — a loop of zero size proves nothing. Choose *amplitude* to
    match the displacements the student will see, of the order of a thermal
    vibration. A conservative teacher does not report exactly zero either; it
    reports the quadrature error of the midpoint rule, which falls as
    *segments* rises and is what a comparison against a direct-force teacher
    should be read against.

    Parameters
    ----------
    teacher : TeacherScorer | BaseModelMixin
        Teacher whose field is probed. A bare model is wrapped in an
        :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`, which
        also handles building and rolling back whatever neighbor list the
        teacher needs at each probe point.
    data : Iterable[Batch] | Batch
        Held-out structures to probe. Positions are displaced in place and
        restored before returning.
    num_loops : int, optional
        Loops integrated per graph. Default ``4``.
    amplitude : float, optional
        Loop side length in A. Default ``0.05``.
    segments : int, optional
        Midpoint-rule samples per side; each costs one teacher force
        evaluation, so one loop costs ``4 * segments``. Default ``4``.
    generator : torch.Generator | None, optional
        Generator drawing the loop directions. Default ``None`` (the global
        RNG).

    Returns
    -------
    NonConservativeResidual
        Loop work, the force floor it implies, and its size relative to the
        teacher's own force scale.

    Raises
    ------
    ValueError
        If *amplitude*, *num_loops*, or *segments* is not positive, if the
        scorer does not produce forces, or if *data* holds no graphs.

    Examples
    --------
    >>> from nvalchemi.training.distillation.evaluation import (
    ...     nonconservative_residual,
    ... )
    >>> residual = nonconservative_residual(teacher, holdout)  # doctest: +SKIP
    >>> residual.relative_floor  # doctest: +SKIP
    0.02
    """
    if amplitude <= 0.0 or num_loops <= 0 or segments <= 0:
        raise ValueError(
            "amplitude, num_loops, and segments must all be positive; got "
            f"amplitude={amplitude!r}, num_loops={num_loops!r}, "
            f"segments={segments!r}."
        )
    scorer = _as_scorer(teacher, ["forces"])
    works: list[torch.Tensor] = []
    force_squares: list[torch.Tensor] = []
    for batch in [data] if isinstance(data, Batch) else data:
        force_squares.append(
            scorer.label(batch)["teacher_forces"][0].pow(2).sum(dim=-1).flatten()
        )
        base = batch.positions
        for _ in range(num_loops):
            first, second = _probe_directions(batch, generator)
            works.append(
                _loop_work(scorer, batch, base, first, second, amplitude, segments)
            )
    if not works:
        raise ValueError("data must hold at least one graph to probe.")
    work = torch.cat(works).abs().to(torch.float64)
    floor = work / (4.0 * amplitude)
    magnitudes = torch.cat(force_squares).to(torch.float64)
    return NonConservativeResidual(
        num_probes=int(work.numel()),
        amplitude=amplitude,
        segments=segments,
        loop_work_mean_abs=float(work.mean()),
        loop_work_max_abs=float(work.max()),
        force_floor=float(floor.mean()),
        force_floor_max=float(floor.max()),
        force_rms=float(magnitudes.mean().sqrt()),
        relative_floor=float(floor.mean() / magnitudes.mean().sqrt().clamp_min(_EPS)),
    )


def _loop_work(
    scorer: Any,
    batch: Batch,
    base: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    amplitude: float,
    segments: int,
) -> torch.Tensor:
    """Integrate the teacher's work around one closed rectangular loop."""
    corners = (
        torch.zeros_like(first),
        amplitude * first,
        amplitude * (first + second),
        amplitude * second,
    )
    work = base.new_zeros(batch.num_graphs)
    for index in range(4):
        start = corners[index]
        step = (corners[(index + 1) % 4] - start) / segments
        for sample in range(segments):
            with _displaced(batch, base + start + step * (sample + 0.5)):
                forces = scorer.label(batch)["teacher_forces"][0]
            work = work + _per_graph_sum((forces * step).sum(dim=-1), batch)
    return work
