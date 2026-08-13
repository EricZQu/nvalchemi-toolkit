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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from nvalchemi.models.base import ModelConfig, NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["InProcessTeacherScorer", "TeacherScorer"]


@dataclasses.dataclass(frozen=True)
class _SignalSpec:
    """Model output, batch field, and level backing one teacher signal."""

    model_output: str | None
    field: str
    level: str


_SIGNAL_SPECS: dict[str, _SignalSpec] = {
    "energy": _SignalSpec("energy", "teacher_energy", "system"),
    "forces": _SignalSpec("forces", "teacher_forces", "node"),
    "stress": _SignalSpec("stress", "teacher_stress", "system"),
    "node_energies": _SignalSpec("atomic_energies", "teacher_node_energies", "node"),
    "embeddings": _SignalSpec(None, "teacher_node_embeddings", "node"),
}
"""Supported teacher signals, keyed by signal name.

``model_output`` is ``None`` for signals produced outside the forward pass
(``embeddings`` comes from :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`).
Canonical shapes are ``(B, 1)`` energy, ``(V, 3)`` forces, ``(B, 3, 3)`` stress,
``(V,)`` node energies, and ``(V, D)`` node embeddings.
"""

_NEIGHBOR_KEYS = frozenset(
    {
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_matrix_shifts",
        "neighbor_list",
        "neighbor_list_shifts",
    }
)
"""Ephemeral neighbor keys, mirroring ``ConvergedSnapshotHook._NEIGHBOR_KEYS``."""

_EMBEDDING_KEYS = frozenset({"node_embeddings", "graph_embeddings"})
"""Batch keys that :meth:`compute_embeddings` implementations write in place."""

_CUTOFF_TOLERANCE = 1e-6
"""Absolute tolerance when matching a pre-built neighbor list to a cutoff."""


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


def _matches_neighbor_config(batch: Batch, config: NeighborConfig) -> bool:
    """Return whether *batch* already carries a list the teacher can consume."""
    if config.half_list:
        return False
    cutoff = getattr(batch, "_neighbor_list_cutoff", None)
    if cutoff is None or abs(float(cutoff) - config.cutoff) > _CUTOFF_TOLERANCE:
        return False
    if config.format == NeighborListFormat.COO:
        return "neighbor_list" in batch
    return "neighbor_matrix" in batch and "num_neighbors" in batch


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

    A pre-built list is reused only when it is a full list at the same cutoff
    and format the teacher asks for.  A batch records the cutoff it was built
    at but not whether it is a half list, and consuming a full list as a half
    list (or the reverse) double-counts every pair, so a teacher configured
    with ``half_list=True`` always rebuilds.

    When the list is rebuilt, the node-level neighbor tensors, the edge group
    (which COO construction replaces wholesale), and the
    ``_neighbor_list_cutoff`` stamp are snapshotted and restored afterwards.

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
    if config is None or _matches_neighbor_config(batch, config):
        yield
        return

    atoms = batch._atoms_group
    saved_nodes = (
        {key: atoms[key] for key in _NEIGHBOR_KEYS if key in atoms}
        if atoms is not None
        else {}
    )
    saved_edges = batch._storage.groups.get("edges")
    saved_cutoff = getattr(batch, "_neighbor_list_cutoff", None)
    try:
        compute_neighbors(batch, config=config)
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
        if saved_cutoff is None:
            if hasattr(batch, "_neighbor_list_cutoff"):
                delattr(batch, "_neighbor_list_cutoff")
        else:
            batch._neighbor_list_cutoff = saved_cutoff


@runtime_checkable
class TeacherScorer(Protocol):
    """Structural interface for objects that produce teacher signals for a batch.

    Implementations declare which signals they emit and return, for one
    :class:`~nvalchemi.data.Batch`, a mapping from batch field name to a
    ``(tensor, level)`` pair.  Levels are ``"node"`` or ``"system"``, matching
    :meth:`~nvalchemi.data.Batch.add_key`.  Tensors must be detached so a
    consumer can store them without holding an autograd graph.

    See Also
    --------
    InProcessTeacherScorer : Scorer that evaluates a teacher in this process.
    nvalchemi.training.distillation.labeling.label_dataset : Offline consumer.
    """

    signals: frozenset[str]

    def label(self, batch: Batch) -> dict[str, tuple[torch.Tensor, str]]:
        """Return ``{batch field: (detached tensor, level)}`` for *batch*."""
        ...


class InProcessTeacherScorer:
    """Score a batch with a teacher model loaded in the current process.

    The scorer owns the teacher's evaluation contract so callers do not have
    to: it narrows ``active_outputs`` to exactly the outputs the requested
    signals need, builds (and afterwards restores) whatever neighbor list the
    teacher requires, chooses the grad mode the teacher's autograd outputs
    need, detaches everything it returns, and normalizes each signal to its
    canonical shape.  The batch it is handed is left exactly as it was found,
    so a scorer can be called mid-training on a live student batch.

    Requested *signals* are validated at construction: unknown names and
    signals whose model output the teacher does not declare both raise
    immediately, rather than warning during the first forward pass.

    Parameters
    ----------
    teacher : BaseModelMixin
        Model wrapper used to produce the signals.  Placed in evaluation mode
        at construction.  Its parameters are never modified — neither their
        values nor their ``requires_grad`` flags — because every returned
        tensor is detached.
    signals : Iterable[str]
        Signal names to produce.  Supported: ``"energy"``, ``"forces"``,
        ``"stress"``, ``"node_energies"``, ``"embeddings"``.
    cast_to : torch.dtype | None, optional
        Cast floating-point outputs to this dtype, e.g. to store labels at
        lower precision than the teacher computes them.  Default ``None``
        (keep the teacher's dtype).

    Raises
    ------
    ValueError
        If *signals* is empty, names an unsupported signal, requires a model
        output the teacher does not declare, or requests ``"embeddings"`` from
        a teacher that publishes no node-embedding shape.

    Examples
    --------
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", "forces"])  # doctest: +SKIP
    >>> labels = scorer.label(batch)  # doctest: +SKIP
    >>> labels["teacher_forces"][1]  # doctest: +SKIP
    'node'

    Notes
    -----
    - A pre-built neighbor list is reused only when it is a full list at the
      teacher's own cutoff and format; anything else is rebuilt and rolled
      back.
    - ``requires_grad`` on ``positions`` and the teacher's declared autograd
      inputs is snapshotted before the forward pass and restored afterwards, so
      a flag the caller set stays set and a flag the teacher enabled is
      cleared again.
    """

    def __init__(
        self,
        teacher: BaseModelMixin,
        signals: Iterable[str],
        *,
        cast_to: torch.dtype | None = None,
    ) -> None:
        requested = frozenset(signals)
        if not requested:
            raise ValueError(
                "At least one teacher signal must be requested; got an empty selection."
            )
        unsupported = requested - frozenset(_SIGNAL_SPECS)
        if unsupported:
            raise ValueError(
                f"Unsupported teacher signal(s); got {sorted(unsupported)!r}, "
                f"expected names from {sorted(_SIGNAL_SPECS)!r}."
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
                f"Teacher cannot produce the output(s) {sorted(missing)!r} required by "
                f"signals {sorted(requested)!r}; it declares outputs={sorted(declared)!r}."
            )
        if (
            "embeddings" in requested
            and "node_embeddings" not in _node_embedding_shapes(teacher)
        ):
            raise ValueError(
                "Teacher publishes no 'node_embeddings' entry in embedding_shapes, so "
                f"the 'embeddings' signal is unavailable; got {type(teacher).__name__!r}."
            )

        self.teacher = teacher
        self.signals = requested
        self.cast_to = cast_to
        self._required_outputs = required
        evaluate = getattr(teacher, "eval", None)
        if callable(evaluate):
            evaluate()

    def label(self, batch: Batch) -> dict[str, tuple[torch.Tensor, str]]:
        """Return the requested teacher signals for *batch*.

        Parameters
        ----------
        batch : Batch
            Batch to score.  Restored to its incoming state before returning,
            including neighbor tensors and any pre-existing embeddings.

        Returns
        -------
        dict[str, tuple[torch.Tensor, str]]
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

    def _forward_labels(self, batch: Batch) -> dict[str, tuple[torch.Tensor, str]]:
        """Run the teacher forward pass and collect its detached signals."""
        config = self.teacher.model_config
        grad_mode = (
            torch.enable_grad()
            if config.autograd_outputs & self._required_outputs
            else torch.no_grad()
        )
        with grad_mode:
            outputs = self.teacher(batch)
        labels: dict[str, tuple[torch.Tensor, str]] = {}
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

    def _embedding_labels(self, batch: Batch) -> dict[str, tuple[torch.Tensor, str]]:
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
                    "Teacher compute_embeddings() wrote no 'node_embeddings' onto the "
                    f"batch; got {type(self.teacher).__name__!r}."
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
