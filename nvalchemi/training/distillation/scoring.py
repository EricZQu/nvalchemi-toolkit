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
"""Teacher scoring interfaces for knowledge distillation."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

import torch

from nvalchemi.models.base import ModelConfig, NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.models.base import BaseModelMixin

__all__ = [
    "InProcessTeacherScorer",
    "SUPPORTED_SIGNALS",
    "SignalLevel",
    "TeacherLabels",
    "TeacherScorer",
    "scorer_fields",
    "signal_fields",
    "signal_for_field",
]

SignalLevel: TypeAlias = Literal["node", "system"]
"""Batch level a teacher signal is attached at."""

TeacherLabels: TypeAlias = dict[str, tuple[torch.Tensor, SignalLevel]]
"""Teacher signals for one batch, keyed by the batch field they populate."""


@dataclasses.dataclass(frozen=True)
class _SignalSpec:
    """Model output, batch field, and level backing one teacher signal."""

    model_output: str | None
    field: str
    level: SignalLevel
    extra_fields: tuple[str, ...] = ()


_SIGNAL_SPECS: dict[str, _SignalSpec] = {
    "energy": _SignalSpec("energy", "teacher_energy", "system"),
    "forces": _SignalSpec("forces", "teacher_forces", "node"),
    "stress": _SignalSpec("stress", "teacher_stress", "system"),
    "node_energies": _SignalSpec("atomic_energies", "teacher_node_energies", "node"),
    "embeddings": _SignalSpec(None, "teacher_node_embeddings", "node"),
}
"""Supported teacher signals, keyed by signal name."""

SUPPORTED_SIGNALS: frozenset[str] = frozenset(_SIGNAL_SPECS)
"""Teacher signal names :class:`InProcessTeacherScorer` can produce."""

_DENSE_NEIGHBOR_KEYS = frozenset(
    {"neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"}
)
"""Node-level neighbor tensors a ``MATRIX`` build writes."""

_SPARSE_NEIGHBOR_KEYS = frozenset({"neighbor_list", "neighbor_list_shifts"})
"""Edge-level neighbor tensors a ``COO`` build writes."""

_NEIGHBOR_KEYS = _DENSE_NEIGHBOR_KEYS | _SPARSE_NEIGHBOR_KEYS
"""Ephemeral neighbor keys; the distillation package's shared definition."""

_STORABLE_DTYPES = (torch.float16, torch.float32, torch.float64)
"""Floating-point dtypes an ALCHEMI Zarr store can hold."""

_EMBEDDING_KEYS = frozenset({"node_embeddings", "graph_embeddings"})
"""Batch keys that :meth:`compute_embeddings` implementations write in place."""

_CUTOFF_TOLERANCE = 1e-6
"""Absolute tolerance when matching a pre-built neighbor list to a cutoff."""

_CUTOFF_ATTR = "_neighbor_list_cutoff"
"""Batch attribute recording the cutoff a neighbor list was built at."""

_HALF_LIST_ATTR = "_neighbor_list_half"
"""Batch attribute recording whether a neighbor list holds each pair once."""

_PIPELINE_SOURCES_ATTR = "_pipeline_neighbor_sources"
"""Instance-dict attribute a composed pipeline's neighbor hook captures its per-source lists in."""

_SHADOWED_NEIGHBOR_ATTRS = _NEIGHBOR_KEYS | {
    "edge_ptr",
    _CUTOFF_ATTR,
    _HALF_LIST_ATTR,
}
"""Instance-dict neighbor attributes snapshotted and restored around a rebuild."""


def signal_fields(signals: Iterable[str]) -> tuple[str, ...]:
    """Return every batch field the named signals populate, sorted.

    A signal usually populates one field, but may populate companion fields
    alongside it; all of them are reported here, so a consumer can prepare for
    (or check for) the whole set a scorer will write.

    Parameters
    ----------
    signals : Iterable[str]
        Signal names, each of which must be in :data:`SUPPORTED_SIGNALS`.

    Returns
    -------
    tuple[str, ...]
        Batch field names, deduplicated and sorted.

    Raises
    ------
    KeyError
        If a name is not a supported signal.
    """
    fields: set[str] = set()
    for name in signals:
        spec = _SIGNAL_SPECS.get(name)
        if spec is None:
            raise KeyError(
                f"Unknown teacher signal {name!r}; supported signals are "
                f"{sorted(SUPPORTED_SIGNALS)!r}."
            )
        fields.add(spec.field)
        fields.update(spec.extra_fields)
    return tuple(sorted(fields))


def signal_for_field(field: str) -> str | None:
    """Return the signal that populates *field*, or ``None`` when no signal does.

    Parameters
    ----------
    field : str
        Batch field name to resolve back to the signal that writes it.

    Returns
    -------
    str | None
        Name of the signal populating *field*, or ``None`` when *field* is not
        one a supported signal writes.
    """
    for name, spec in _SIGNAL_SPECS.items():
        if field == spec.field or field in spec.extra_fields:
            return name
    return None


def _normalize_signal_shape(signal: str, value: torch.Tensor) -> torch.Tensor:
    """Reshape a raw teacher output to the canonical shape for *signal*."""
    match signal:
        case "energy":
            return value.unsqueeze(-1) if value.ndim == 1 else value
        case "node_energies":
            return value.reshape(-1)
        case "stress":
            return value.reshape(-1, 3, 3)
        case _:
            return value


def _node_embedding_shapes(teacher: BaseModelMixin) -> dict[str, tuple[int, ...]]:
    """Return the teacher's embedding shapes, empty when it publishes none."""
    try:
        return teacher.embedding_shapes or {}
    except NotImplementedError:
        return {}


def _planned_neighbor_sources(teacher: BaseModelMixin) -> int:
    """Return how many neighbor lists *teacher* consumes per batch."""
    factory = getattr(teacher, "make_neighbor_hooks", None)
    if not callable(factory):
        return 1
    hooks = factory()
    if not isinstance(hooks, list) or not hooks:
        return 1
    sources = getattr(hooks[0], "sources", None)
    return len(sources) if isinstance(sources, (list, tuple)) else 1


def _matches_neighbor_config(batch: Batch, config: NeighborConfig) -> bool:
    """Return whether *batch* already carries a list the teacher can consume."""
    if config.half_list or getattr(batch, _HALF_LIST_ATTR, None) is not False:
        return False
    cutoff = getattr(batch, _CUTOFF_ATTR, None)
    if cutoff is None or abs(float(cutoff) - config.cutoff) > _CUTOFF_TOLERANCE:
        return False
    required = (
        ("neighbor_list",)
        if config.format == NeighborListFormat.COO
        else ("neighbor_matrix", "num_neighbors")
    )
    # The provenance stamps live in the instance dict, so while a shadowed list
    # is present they describe that list rather than anything in storage.
    if _NEIGHBOR_KEYS & batch.__dict__.keys():
        return all(key in batch.__dict__ for key in required)
    return all(key in batch for key in required)


def _snapshot_grad_flags(batch: Batch, config: ModelConfig) -> dict[str, bool]:
    """Return the ``requires_grad`` flag of every input the teacher may enable."""
    keys = {"positions"} | set(config.gradient_keys) | set(config.autograd_inputs)
    flags: dict[str, bool] = {}
    for key in keys:
        value = getattr(batch, key, None)
        if isinstance(value, torch.Tensor):
            flags[key] = value.requires_grad
    return flags


def _restore_grad_flags(batch: Batch, flags: dict[str, bool]) -> None:
    """Restore the ``requires_grad`` flags captured by :func:`_snapshot_grad_flags`."""
    for key, flag in flags.items():
        value = getattr(batch, key, None)
        if isinstance(value, torch.Tensor) and value.requires_grad != flag:
            value.requires_grad_(flag)


@contextmanager
def _isolated_neighbors(batch: Batch, config: NeighborConfig | None) -> Iterator[None]:
    """Build the teacher's neighbor list on *batch*, restoring prior state on exit.

    A pre-built list is reused only when it is a *known* full list at the same
    cutoff and format the teacher asks for. Reading a full list as a half list
    (or the reverse) miscounts every pair, and the core records the cutoff a
    list was built at (``_neighbor_list_cutoff``) but not its half-list
    provenance, so a teacher configured with ``half_list=True`` always rebuilds
    and so does any batch whose provenance is unknown. Every list built here is
    stamped with ``_neighbor_list_half``, which a caller holding a full list can
    also set itself to opt into reuse, and which lets reuse widen on its own if
    the core starts recording the same thing.

    When the list is rebuilt, the node-level neighbor tensors, the edge group
    (which COO construction replaces wholesale), and every neighbor attribute
    the batch carries in its instance dictionary are snapshotted and restored
    afterwards. The instance dictionary matters because a composed pipeline
    leaves the neighbor list of its default source there — a shadow that wins
    over storage on attribute lookup — so a teacher scoring a live pipeline
    batch would otherwise read the student's list at the student's cutoff. The
    incoming list is hidden in both formats for the duration, so neither a
    direct attribute read nor
    :func:`~nvalchemi.models._ops.neighbor_filter.prepare_neighbors_for_model`
    can resolve anything but the list built here.

    A composed pipeline also captures its whole per-source table in the instance
    dictionary, under ``_pipeline_neighbor_sources``, and a composed teacher
    resolves its own list out of that table by source index before it consults
    anything canonical — so the table is hidden across the whole block, reuse
    included, and restored on exit. Hiding it only around a rebuild would leave
    a reused list resolvable through the table instead.

    Parameters
    ----------
    batch : Batch
        Batch to build neighbors on; mutated for the duration of the block.
    config : NeighborConfig | None
        Neighbor requirements of the teacher, or ``None`` for a model that
        needs no neighbor list.

    Yields
    ------
    None
    """
    saved_sources = (
        {_PIPELINE_SOURCES_ATTR: batch.__dict__.pop(_PIPELINE_SOURCES_ATTR)}
        if _PIPELINE_SOURCES_ATTR in batch.__dict__
        else {}
    )
    try:
        if config is None or _matches_neighbor_config(batch, config):
            yield
            return

        atoms = batch._atoms_group
        saved_nodes = (
            {key: atoms[key] for key in _NEIGHBOR_KEYS if key in atoms}
            if atoms is not None
            else {}
        )
        saved_edges = batch._storage.groups.pop("edges", None)
        saved_shadows = {
            name: batch.__dict__.pop(name)
            for name in _SHADOWED_NEIGHBOR_ATTRS
            if name in batch.__dict__
        }
        if atoms is not None:
            for key in saved_nodes:
                del atoms[key]
        try:
            compute_neighbors(batch, config=config)
            setattr(batch, _HALF_LIST_ATTR, config.half_list)
            yield
        finally:
            if atoms is not None:
                for key in _NEIGHBOR_KEYS:
                    if key in atoms:
                        del atoms[key]
                for key, value in saved_nodes.items():
                    atoms[key] = value
            if saved_edges is None:
                batch._storage.groups.pop("edges", None)
            else:
                batch._storage.groups["edges"] = saved_edges
            for name in _SHADOWED_NEIGHBOR_ATTRS:
                batch.__dict__.pop(name, None)
            batch.__dict__.update(saved_shadows)
    finally:
        batch.__dict__.update(saved_sources)


@runtime_checkable
class TeacherScorer(Protocol):
    """Structural interface for objects that produce teacher signals for a batch.

    Implementations declare which signals they emit and return, for one
    :class:`~nvalchemi.data.Batch`, a mapping from batch field name to a
    ``(tensor, level)`` pair. Levels are ``"node"`` or ``"system"``, matching
    :meth:`~nvalchemi.data.Batch.add_key`. Tensors must be detached so a
    consumer can store them without holding an autograd graph, and live on the
    device of the batch they were computed from.

    An implementation may also publish ``label_fields``, the batch fields its
    :meth:`label` populates, which lets a consumer learn the fields without
    scoring a batch first. Consumers read it through :func:`scorer_fields`
    rather than off the attribute, because a scorer that declares nothing but
    built-in signals still has knowable fields. The protocol will not grow
    required members, so ``isinstance`` keeps accepting a scorer declaring only
    ``signals`` and ``label``.

    See Also
    --------
    InProcessTeacherScorer : Scorer that evaluates a teacher in this process.
    nvalchemi.training.distillation.labeling.label_dataset : Offline consumer.
    """

    signals: frozenset[str]

    def label(self, batch: Batch) -> TeacherLabels:
        """Return ``{batch field: (detached tensor, level)}`` for *batch*."""
        ...


def scorer_fields(scorer: TeacherScorer) -> tuple[str, ...] | None:
    """Return the batch fields *scorer* populates, or ``None`` when they cannot be known.

    Resolved in three steps: a ``label_fields`` declaration on *scorer* is taken
    at its word; otherwise a scorer whose signals are all in
    :data:`SUPPORTED_SIGNALS` gets the fields those signals populate; otherwise
    the fields are unknown, because a custom scorer is free to map a signal name
    of its own onto whatever fields it likes.

    ``None`` is not the empty tuple: a scorer whose :meth:`TeacherScorer.label`
    returns nothing declares ``()``, while an undeclared scorer with a custom
    signal is ``None``. A consumer that needs the fields — an idempotency check
    that skips scoring a batch already carrying them, say — must treat ``None``
    as unknown rather than as nothing to check.

    Parameters
    ----------
    scorer : TeacherScorer
        Scorer to resolve the fields of.

    Returns
    -------
    tuple[str, ...] | None
        Batch field names the scorer writes, or ``None`` when they cannot be
        determined without scoring a batch.
    """
    declared = getattr(scorer, "label_fields", None)
    if declared is not None and not isinstance(declared, str):
        return tuple(declared)
    if frozenset(scorer.signals) <= SUPPORTED_SIGNALS:
        return signal_fields(scorer.signals)
    return None


class InProcessTeacherScorer:
    """Score a batch with a teacher model loaded in the current process.

    The scorer owns the teacher's evaluation contract so callers do not have
    to: it narrows ``active_outputs`` to exactly the outputs the requested
    signals need, builds (and afterwards restores) whatever neighbor list the
    teacher requires, chooses the grad mode the teacher's autograd outputs
    need, detaches everything it returns, and normalizes each signal to its
    canonical shape. The batch it is handed is left exactly as it was found, so
    a scorer can be called mid-training on a live student batch.

    Each signal maps to one batch field at one level: ``energy`` to
    ``teacher_energy`` ``(B, 1)`` and ``stress`` to ``teacher_stress``
    ``(B, 3, 3)`` at system level; ``forces`` to ``teacher_forces`` ``(V, 3)``,
    ``node_energies`` to ``teacher_node_energies`` ``(V,)``, and ``embeddings``
    to ``teacher_node_embeddings`` ``(V, D)`` at node level. Every signal but
    ``embeddings`` comes from the forward pass; ``embeddings`` comes from
    :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`. Those
    fields are published as ``label_fields``, so a consumer can learn what the
    scorer writes without scoring a batch.

    Requested *signals* are validated at construction: unknown names and
    signals whose model output the teacher does not declare both raise
    immediately, rather than warning during the first forward pass.

    Parameters
    ----------
    teacher : BaseModelMixin
        Model wrapper used to produce the signals. Placed in evaluation mode at
        construction. Its parameters are never modified — neither their values
        nor their ``requires_grad`` flags — because every returned tensor is
        detached.
    signals : Iterable[str]
        Signal names to produce. Supported: ``"energy"``, ``"forces"``,
        ``"stress"``, ``"node_energies"``, ``"embeddings"``.
    cast_to : torch.dtype | None, optional
        Cast floating-point outputs to this dtype, e.g. to score in the
        student's precision or to store labels at lower precision than the
        teacher computes them. Any floating-point dtype is accepted; whether a
        store can hold it is the store's own rule, checked by
        :func:`~nvalchemi.training.distillation.labeling.label_dataset` at the
        store boundary. Default ``None`` (keep the teacher's dtype).

    Raises
    ------
    ValueError
        If *signals* is empty, names an unsupported signal, requires a model
        output the teacher does not declare, requests ``"embeddings"`` from a
        teacher that publishes no node-embedding shape, *cast_to* is not a
        floating-point dtype, or *teacher* is a composition planning more than
        one neighbor-list source.

    Examples
    --------
    >>> from nvalchemi.training.distillation import InProcessTeacherScorer
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", "forces"])  # doctest: +SKIP
    >>> labels = scorer.label(batch)  # doctest: +SKIP
    >>> labels["teacher_forces"][1]  # doctest: +SKIP
    'node'

    Notes
    -----
    A pre-built neighbor list is reused only when it is a known full list at
    the teacher's own cutoff and format. A half-list teacher, and any batch
    whose half-list provenance is unknown, gets a list that is rebuilt for the
    forward pass and rolled back afterwards; a caller holding a full list can
    opt into reuse by setting ``batch._neighbor_list_half = False``. A list the
    caller keeps as an instance attribute rather than in batch storage — as a
    composed pipeline does for its default neighbor source — is hidden for the
    duration of the rebuild and restored verbatim, and the per-source table a
    composed pipeline captures alongside it (``_pipeline_neighbor_sources``) is
    hidden for the whole of scoring, reuse included, so the teacher never scores
    against the student's neighborhoods.

    A teacher whose composition plans more than one neighbor-list source is
    refused at construction, because the scorer builds exactly one list per
    batch while such a composition resolves each step's list out of a captured
    source table only its own hooks produce. Compose the teacher to plan a
    single list instead — ``neighbor_adaptation="always"``, or a
    ``max_cutoff_ratio`` of at least the ratio of its largest to its smallest
    cutoff — and it adapts that one list per step.

    ``requires_grad`` on ``positions`` and the teacher's declared autograd
    inputs is snapshotted before the forward pass and restored afterwards, so a
    flag the caller set stays set while a flag the teacher enabled is cleared
    again.
    """

    def __init__(
        self,
        teacher: BaseModelMixin,
        signals: Iterable[str],
        *,
        cast_to: torch.dtype | None = None,
    ) -> None:
        """Validate the requested signals against the teacher's declared outputs."""
        requested = frozenset(signals)
        if not requested:
            raise ValueError(
                f"At least one teacher signal must be requested; got {sorted(requested)!r}."
            )
        unsupported = requested - SUPPORTED_SIGNALS
        if unsupported:
            raise ValueError(
                f"Teacher signals must be names from {sorted(SUPPORTED_SIGNALS)!r}; "
                f"got unsupported {sorted(unsupported)!r}."
            )
        required = frozenset(
            spec.model_output
            for spec in (_SIGNAL_SPECS[name] for name in requested)
            if spec.model_output is not None
        )
        declared = teacher.model_config.outputs
        missing = required - declared
        if missing:
            raise ValueError(
                f"Teacher cannot produce the outputs required by signals "
                f"{sorted(requested)!r}; got outputs={sorted(declared)!r}, "
                f"missing {sorted(missing)!r}."
            )
        if (
            "embeddings" in requested
            and "node_embeddings" not in _node_embedding_shapes(teacher)
        ):
            raise ValueError(
                "Teacher must publish a ``node_embeddings`` shape to serve the "
                f"``embeddings`` signal; got {sorted(_node_embedding_shapes(teacher))!r}."
            )
        if cast_to is not None and not cast_to.is_floating_point:
            raise ValueError(
                f"cast_to must be a floating-point dtype; got {cast_to!r}."
            )
        planned = _planned_neighbor_sources(teacher)
        if planned > 1:
            raise ValueError(
                f"Teacher plans {planned!r} neighbor-list sources, but a scorer "
                "builds one list per batch; compose the teacher to plan a single "
                'list with neighbor_adaptation="always" or a max_cutoff_ratio of '
                "at least its largest-to-smallest cutoff ratio."
            )

        self.teacher = teacher
        self.signals = requested
        self.label_fields = signal_fields(requested)
        self.cast_to = cast_to
        self._required_outputs = required
        evaluate = getattr(teacher, "eval", None)
        if callable(evaluate):
            evaluate()

    def label(self, batch: Batch) -> TeacherLabels:
        """Return the requested teacher signals for *batch*.

        Parameters
        ----------
        batch : Batch
            Batch to score. Restored to its incoming state before returning,
            including neighbor tensors and any pre-existing embeddings.

        Returns
        -------
        TeacherLabels
            Mapping from batch field name to ``(detached tensor, level)``.

        Raises
        ------
        RuntimeError
            If the teacher omits an output or embedding a requested signal
            needs.
        """
        config = self.teacher.model_config
        previous_active = set(config.active_outputs)
        grad_flags = _snapshot_grad_flags(batch, config)
        try:
            self.teacher.set_config("active_outputs", set(self._required_outputs))
            with _isolated_neighbors(batch, config.neighbor_config):
                labels = self._forward_labels(batch) if self._required_outputs else {}
                if "embeddings" in self.signals:
                    labels.update(self._embedding_labels(batch))
        finally:
            self.teacher.set_config("active_outputs", previous_active)
            _restore_grad_flags(batch, grad_flags)
        return labels

    def _forward_labels(self, batch: Batch) -> TeacherLabels:
        """Run the teacher forward pass and collect its detached signals."""
        config = self.teacher.model_config
        grad_mode = (
            torch.enable_grad()
            if config.autograd_outputs & self._required_outputs
            else torch.no_grad()
        )
        with grad_mode:
            outputs = self.teacher(batch)
        labels: TeacherLabels = {}
        for name in sorted(self.signals):
            spec = _SIGNAL_SPECS[name]
            if spec.model_output is None:
                continue
            value = outputs.get(spec.model_output)
            if value is None:
                raise RuntimeError(
                    f"Teacher returned no {spec.model_output!r} output for the "
                    f"{name!r} signal."
                )
            labels[spec.field] = (self._finalize(name, value), spec.level)
        del outputs
        return labels

    def _embedding_labels(self, batch: Batch) -> TeacherLabels:
        """Compute node embeddings without leaving them attached to *batch*.

        Pre-existing embeddings are cleared before the call, because wrappers
        that attach embeddings through :meth:`Batch.add_key` reject a key that
        already exists, and are restored into the group they came from rather
        than through :meth:`Batch.__setitem__`, whose level routing follows the
        attribute registry rather than the incoming layout.
        """
        saved_groups = {}
        for key in _EMBEDDING_KEYS:
            group = batch._storage.group_from_attr(key)
            if group is not None:
                saved_groups[key] = (group, group[key])
                del batch[key]
        saved_tracked = {
            level: names & _EMBEDDING_KEYS
            for level, names in (batch.keys or {}).items()
        }
        for level in saved_tracked:
            batch.keys[level] -= _EMBEDDING_KEYS
        try:
            with torch.no_grad():
                self.teacher.compute_embeddings(batch)
            if "node_embeddings" not in batch:
                raise RuntimeError(
                    "Teacher compute_embeddings() must write ``node_embeddings`` onto "
                    f"the batch; got {sorted(key for key in _EMBEDDING_KEYS if key in batch)!r}."
                )
            spec = _SIGNAL_SPECS["embeddings"]
            value = self._finalize("embeddings", batch["node_embeddings"].clone())
            return {spec.field: (value, spec.level)}
        finally:
            for key in _EMBEDDING_KEYS:
                if key in batch:
                    del batch[key]
            for key, (group, value) in saved_groups.items():
                group[key] = value
            for level, names in saved_tracked.items():
                batch.keys[level] = (batch.keys[level] - _EMBEDDING_KEYS) | names

    def _finalize(self, signal: str, value: torch.Tensor) -> torch.Tensor:
        """Detach *value*, normalize it to the canonical shape, and cast it."""
        value = _normalize_signal_shape(signal, value.detach())
        if self.cast_to is not None and value.is_floating_point():
            value = value.to(self.cast_to)
        return value
