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
"""Offline labeling of a dataset with teacher signals."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch
from tensordict import TensorDict

from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    _get_cat_dim,
)
from nvalchemi.data.level_storage import UniformLevelStorage
from nvalchemi.training.distillation.scoring import (
    _DENSE_NEIGHBOR_KEYS,
    _NEIGHBOR_KEYS,
    _STORABLE_DTYPES,
    _reject_foreign_fields,
    scorer_fields,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from nvalchemi.data import Batch
    from nvalchemi.data.datapipes.backends.zarr import StoreLike
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
    from nvalchemi.training.distillation.scoring import SignalLevel, TeacherScorer

__all__ = ["label_dataset"]


_FieldSchema: TypeAlias = dict[str, tuple[str, torch.dtype, tuple[int, ...]]]
"""Store level, dtype, and row shape of every field a labeled chunk persists."""

_STORE_LEVELS = {"atoms": "atom", "edges": "edge", "system": "system"}
"""Store level names of the built-in batch levels; a custom level keeps its name."""

_REPORTED_MISMATCHES = 4
"""Number of disagreeing store arrays named before an integrity error truncates."""

_PREFETCH_METHODS = ("prefetch_fused_batches", "get_fused_batches", "cancel_prefetch")
"""Dataset methods the pipelined chunk loop reads ahead through."""

_AUTO_PROBE_CHUNKS = 2
"""Chunks ``prefetch="auto"`` reads sequentially before it tries reading ahead."""

_PIPELINE_LOAD_FRACTION = 0.5
"""Load-to-processing time ratio below which ``prefetch="auto"`` stays sequential."""


@dataclasses.dataclass(frozen=True)
class _StoreState:
    """Sample counts and field schema of an existing labeled store."""

    active: int
    total: int
    schema: _FieldSchema


def _torn_store_error(detail: str) -> ValueError:
    """Return the error raised for a store an interrupted run left inconsistent."""
    return ValueError(
        "Store is inconsistent, so a resumed run cannot line up with the dataset: "
        f"{detail}. This is what a labeling run interrupted mid-append leaves "
        "behind; truncate the store back to its committed samples or label into a "
        "fresh one."
    )


def _store_array(reader: AtomicDataZarrReader, field: str, level: str) -> Any | None:
    """Return the Zarr array backing *field*, or ``None`` when the store has none.

    A field at a built-in level lives under ``core/`` or ``custom/``; a field at
    a custom level lives under ``levels/<level>/``.
    """
    root = reader._root
    if level in _STORE_LEVELS.values():
        groups = [root[name] for name in ("core", "custom") if name in root]
    else:
        levels = root["levels"] if "levels" in root else {}
        groups = [levels[level]] if level in levels else []
    for group in groups:
        if field in group:
            return group[field]
    return None


def _level_totals(reader: AtomicDataZarrReader, num_samples: int) -> dict[str, int]:
    """Return the row count every level the store declares should hold.

    Built-in levels follow the atom and edge pointers; a segmented or product
    custom level follows its own pointer, and a uniform one has a row per sample.
    """
    totals = {
        "atom": int(reader._atoms_ptr[-1].item()),
        "edge": int(reader._edges_ptr[-1].item()),
        "system": num_samples,
    }
    for level in set(reader.field_levels.values()) - set(totals):
        pointer = reader._level_ptrs.get(level)
        totals[level] = num_samples if pointer is None else int(pointer[-1].item())
    return totals


def _check_store_integrity(reader: AtomicDataZarrReader) -> None:
    """Raise when a store's arrays disagree about how many samples it holds.

    An append interrupted between extending the pointers, masks, and field
    arrays and committing ``num_samples`` leaves them at different lengths;
    resuming from such a store would misplace every remaining sample. Only
    array metadata is inspected.
    """
    committed = reader._root.attrs.get("num_samples")
    if committed is None:
        raise _torn_store_error("the store records no committed sample count")
    num_samples = int(committed)
    meta = reader._root["meta"]
    pointers = {"atoms_ptr": reader._atoms_ptr, "edges_ptr": reader._edges_ptr}
    for name, pointer in pointers.items():
        if int(pointer[0].item()) != 0 or bool((pointer[1:] < pointer[:-1]).any()):
            raise _torn_store_error(
                f"meta/{name} is not a non-decreasing pointer array starting at zero; "
                f"got {pointer.tolist()!r}"
            )
    totals = _level_totals(reader, num_samples)
    lengths = {
        "meta/atoms_ptr": (int(reader._atoms_ptr.numel()), num_samples + 1),
        "meta/edges_ptr": (int(reader._edges_ptr.numel()), num_samples + 1),
        "meta/samples_mask": (int(reader._samples_mask.numel()), num_samples),
    }
    for name, expected in (("atoms_mask", "atom"), ("edges_mask", "edge")):
        if name in meta:
            lengths[f"meta/{name}"] = (int(meta[name].shape[0]), totals[expected])
    for field, level in reader.field_levels.items():
        array = _store_array(reader, field, level)
        if array is None:
            raise _torn_store_error(
                f"the store declares field {field!r} but holds no array for it"
            )
        cat_dim = _get_cat_dim(field) % len(array.shape)
        lengths[field] = (int(array.shape[cat_dim]), totals[level])
    mismatched = [
        f"{name} holds {found!r} rows where {expected!r} are committed"
        for name, (found, expected) in lengths.items()
        if found != expected
    ]
    if mismatched:
        reported = ", ".join(mismatched[:_REPORTED_MISMATCHES])
        remaining = len(mismatched) - _REPORTED_MISMATCHES
        raise _torn_store_error(
            f"{num_samples!r} samples are committed but {reported}"
            + (f", and {remaining!r} further arrays disagree" if remaining > 0 else "")
        )


def _row_shape(field: str, shape: Sequence[int]) -> tuple[int, ...]:
    """Return *shape* without the axis a store concatenates *field* along."""
    cat_dim = _get_cat_dim(field) % len(shape)
    return tuple(size for axis, size in enumerate(shape) if axis != cat_dim)


def _store_schema(reader: AtomicDataZarrReader) -> _FieldSchema:
    """Return the level, dtype, and row shape of every field an existing store holds.

    Runs after :func:`_check_store_integrity`, so every declared field is known
    to have an array. Dtypes come from an empty slice, which reads no chunk.
    """
    schema: _FieldSchema = {}
    for field, level in reader.field_levels.items():
        array = _store_array(reader, field, level)
        dtype = torch.from_numpy(array[:0]).dtype
        schema[field] = (level, dtype, _row_shape(field, array.shape))
    return schema


def _existing_store_state(store: StoreLike) -> _StoreState | None:
    """Return the state of *store*, or ``None`` when it cannot be read."""
    try:
        reader = AtomicDataZarrReader(store)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    try:
        _check_store_integrity(reader)
        return _StoreState(
            active=len(reader),
            total=int(reader._samples_mask.numel()),
            schema=_store_schema(reader),
        )
    finally:
        reader.close()


def _ensure_system_group(batch: Batch) -> None:
    """Give *batch* an empty, sized system group when it has none.

    A batch of bare positions and atomic numbers carries no system group, and
    :meth:`~nvalchemi.data.Batch.add_key` cannot create one for a built-in level.
    """
    if "system" in batch._storage.groups:
        return
    batch._storage.groups["system"] = UniformLevelStorage(
        data=TensorDict({}, batch_size=[batch.num_graphs], device=batch.device),
        device=batch.device,
        attr_map=batch._storage.attr_map,
        validate=False,
    )


def _batch_schema(batch: Batch) -> _FieldSchema:
    """Return the level, dtype, and row shape a writer would persist for each field.

    Mirrors the writer's layout: a system-level tensor has its unit axes after
    the sample axis squeezed away before it is stored. Levels come from the
    batch's storage rather than ``batch.keys``, which carries only the built-in
    levels, so a custom-level field is held to the schema like any other.
    """
    schema: _FieldSchema = {}
    for level, names in batch.level_keys.items():
        for name in names:
            value = batch[name]
            shape = tuple(value.shape)
            if level == "system":
                while len(shape) > 2 and shape[1] == 1:
                    shape = shape[:1] + shape[2:]
            store_level = _STORE_LEVELS.get(level, level)
            schema[name] = (store_level, value.dtype, _row_shape(name, shape))
    return schema


def _check_chunk_schema(
    reference: _FieldSchema, outgoing: _FieldSchema, indices: Sequence[int]
) -> None:
    """Raise when a chunk would write a different schema than the store holds.

    The writer's append extends only the arrays a store already holds, so
    drifting fields would misalign arrays, drifting dtypes would cast labels,
    and drifting row shapes would truncate them, all without an error.
    """
    chunk = f"the chunk covering samples {indices[0]!r}-{indices[-1]!r}"
    extra = sorted(set(outgoing) - set(reference))
    missing = sorted(set(reference) - set(outgoing))
    if extra or missing:
        raise ValueError(
            "Every labeled chunk must write the fields the store holds; "
            f"{chunk} writes extra {extra!r} and is missing {missing!r}."
        )
    drifted = ", ".join(
        f"{name} is stored as {reference[name]!r} but arrives as {outgoing[name]!r}"
        for name in sorted(reference)
        if reference[name] != outgoing[name]
    )
    if drifted:
        raise ValueError(
            "Every labeled chunk must write the levels, dtypes, and row shapes the "
            f"store holds; in {chunk}, {drifted}."
        )


def _check_storable_dtypes(outgoing: _FieldSchema) -> None:
    """Raise when a chunk carries a floating-point dtype no store can hold.

    Only the chunk defining a fresh store's schema is checked; every later
    chunk is already held to that schema.
    """
    unstorable = ", ".join(
        f"{name} arrives as {dtype!r}"
        for name, (_, dtype, _) in sorted(outgoing.items())
        if dtype.is_floating_point and dtype not in _STORABLE_DTYPES
    )
    if unstorable:
        raise ValueError(
            "Every floating-point field must arrive in a dtype an ALCHEMI Zarr "
            f"store can hold; {unstorable}, and the storable dtypes are "
            f"{list(_STORABLE_DTYPES)!r}."
        )


def _split_per_graph(
    batch: Batch, field: str, values: torch.Tensor, level: SignalLevel
) -> list[torch.Tensor]:
    """Split a concatenated teacher tensor into one entry per graph.

    Raises
    ------
    ValueError
        If *values* does not hold one row per atom or per graph. The split
        would otherwise drop the surplus rows before
        :meth:`~nvalchemi.data.Batch.add_key` or the store's schema checks
        could see them.
    """
    expected = batch.num_nodes if level == "node" else batch.num_graphs
    if values.shape[:1] != (expected,):
        shape = tuple(values.shape)
        unit = "atom" if level == "node" else "graph"
        raise ValueError(
            f"Teacher label {field!r} at level {level!r} has shape {shape!r}; "
            f"expected {expected!r} rows, one per {unit}."
        )
    if level == "node":
        return list(torch.split(values, batch.num_nodes_list, dim=0))
    return [values[index : index + 1] for index in range(batch.num_graphs)]


def _strip_unstorable(
    batch: Batch, keep: frozenset[str], ephemeral: frozenset[str]
) -> None:
    """Drop *ephemeral* and any field that appeared during labeling, keeping *keep*.

    An edge group left with no fields is dropped too, so the store's edge
    pointers never record edges no array backs.
    """
    for key in ephemeral | (frozenset(_batch_schema(batch)) - keep):
        if key in batch:
            del batch[key]
    edges = batch._storage.groups.get("edges")
    if edges is not None and next(edges.keys(), None) is None:
        batch._storage.groups.pop("edges")


def _chunk_batches(
    dataset: BatchDatasetProtocol,
    chunks: Sequence[list[int]],
    prefetch: bool | Literal["auto"],
) -> Iterator[Batch]:
    """Yield the batch of every chunk, reading one chunk ahead while pipelining is on.

    Under ``prefetch="auto"`` the first :data:`_AUTO_PROBE_CHUNKS` chunks are
    read sequentially, with each load and the caller's processing of the
    yielded batch timed separately. The last of them is the reference: reading
    ahead starts only when its load took at least
    :data:`_PIPELINE_LOAD_FRACTION` of its processing, since a shorter load
    has too little to hide. The first chunk read entirely ahead is then timed
    against the reference per atom, and the run falls back to sequential reads
    when it was not faster, which is what a fast local store looks like once
    the read-ahead thread contends with the scoring. The first chunk is never
    the reference because it carries CUDA warm-up and the store's creation. A
    read left pending when the caller stops early is cancelled.
    """
    pipelined = prefetch is True
    pending = False
    reference = 0.0
    try:
        for position, indices in enumerate(chunks):
            started = perf_counter()
            if pending:
                (batch,) = dataset.get_fused_batches()
                pending = False
            else:
                batch = dataset.load_batches([indices])[0]
            probing = prefetch == "auto" and position <= _AUTO_PROBE_CHUNKS + 1
            if probing and batch.device.type == "cuda":
                torch.cuda.synchronize(batch.device)
            loaded = perf_counter()
            if pipelined and position + 1 < len(chunks):
                dataset.prefetch_fused_batches([chunks[position + 1]])
                pending = True
            yield batch
            if not probing:
                continue
            elapsed = perf_counter() - started
            if position == _AUTO_PROBE_CHUNKS - 1:
                load = loaded - started
                reference = elapsed / batch.num_nodes
                pipelined = load >= _PIPELINE_LOAD_FRACTION * (elapsed - load)
            elif position == _AUTO_PROBE_CHUNKS + 1 and pipelined:
                pipelined = elapsed / batch.num_nodes < reference
    finally:
        if pending:
            dataset.cancel_prefetch()


def label_dataset(
    dataset: BatchDatasetProtocol,
    scorer: TeacherScorer,
    store: StoreLike,
    *,
    batch_size: int = 32,
    device: torch.device | str | None = None,
    resume: bool = True,
    keep_neighbors: bool = False,
    prefetch: bool | Literal["auto"] = "auto",
) -> int:
    """Label *dataset* with teacher signals and persist the result to *store*.

    Walks *dataset* in contiguous chunks of *batch_size* samples, scores each
    chunk with *scorer*, attaches every returned signal as a batch field, and
    writes the augmented chunk to a Zarr store holding the original fields plus
    the teacher fields, readable through the ordinary
    :class:`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrReader` /
    :class:`~nvalchemi.data.datapipes.dataset.Dataset` path.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Source dataset; only ``__len__`` and ``load_batches`` are used.
    scorer : TeacherScorer
        Scorer producing the teacher signals for each chunk.
    store : StoreLike
        Destination Zarr store: a path, a zarr store instance, or a dict.
    batch_size : int, optional
        Number of samples scored per forward pass. Default ``32``.
    device : torch.device | str | None, optional
        Device to move each chunk to before scoring. Default ``None``
        (score on whatever device the dataset emits).
    resume : bool, optional
        If ``True`` (default), an existing store is treated as a partial run:
        the first ``len(store)`` samples are skipped and labeling continues
        from there. A store already holding every sample is a no-op; one
        holding more samples than *dataset* has is refused. If ``False``, an
        existing store is an error.
    keep_neighbors : bool, optional
        If ``False`` (default), a source neighbor list is dropped rather than
        stored, because the cutoff it was built at lives on the batch and not
        in the store. ``True`` carries a sparse (``COO``) source list over; the
        dense tensors are dropped either way. Default ``False``.
    prefetch : bool | Literal["auto"], optional
        Whether to read each chunk while the previous one is scored and
        written. ``False`` reads, scores, and writes one chunk at a time.
        ``True`` reads one chunk ahead through the dataset's fused-prefetch
        surface (``prefetch_fused_batches`` / ``get_fused_batches``), falling
        back to the sequential loop with a :class:`UserWarning` when the
        dataset offers none. ``"auto"`` (default) reads the first two chunks
        sequentially and times the second one's load against the scoring and
        writing of its chunk; when the load took at least half of that
        processing it reads ahead, then times the first chunk read entirely
        ahead against the sequential one per atom and falls back to
        sequential reads if reading ahead was not faster. A dataset without
        the surface stays sequential silently. Default ``"auto"``.

    Returns
    -------
    int
        Number of samples labeled by this call; ``0`` when a resumed store
        already covers the whole dataset.

    Raises
    ------
    ValueError
        If *batch_size* is not positive, *prefetch* is not ``True``, ``False``,
        or ``"auto"``, *scorer* declares or returns a batch
        field outside the ``teacher_*`` namespace, *store* exists but cannot be
        read as an ALCHEMI Zarr store, *resume* is ``False`` and *store*
        exists, *store* holds soft-deleted samples or more samples than
        *dataset* has, *store* holds arrays that disagree about how many
        samples it contains, a teacher label does not hold one row per atom or
        per graph, a chunk carries a floating-point field in a dtype a store
        cannot hold, or a chunk would write a different field set, level,
        dtype, or row shape than the store holds.
    TypeError
        If *scorer* declares ``label_fields`` as a single string.

    Examples
    --------
    >>> from nvalchemi.training.distillation import label_dataset
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", "forces"])  # doctest: +SKIP
    >>> label_dataset(dataset, scorer, "labeled.zarr", batch_size=64)  # doctest: +SKIP
    1024

    Notes
    -----
    The first chunk defines the store schema, and every later chunk — on
    fresh and resumed runs alike — must write the same fields, levels, dtypes,
    and row shapes, since the writer would otherwise misalign, cast, or
    truncate labels silently. Each label is held to the chunk's atom or graph
    count before it is attached, because the split into per-graph rows would
    otherwise drop whatever a scorer returned beyond it. Resuming assumes
    stored sample *i* is dataset sample *i*: soft-deleted samples, a store
    longer than the dataset, and a store whose arrays disagree with its
    committed sample count (what an interrupted append leaves) are refused,
    while drift within the dataset's length is undetectable. Labels are
    attached with ``overwrite=True``, so a scorer is held to the ``teacher_*``
    namespace both by its declared ``label_fields`` and by every chunk it
    returns, to protect the reference fields it would otherwise replace.

    Labels stored in float16 or float64 read back at the reading dataset's
    ``positions`` dtype, because a dataset coerces every floating-point field
    it loads (:meth:`~nvalchemi.data.AtomicData.check_fp_dtype_consistency`);
    the stored dtype governs the store's size, not what training sees. Build
    the student's neighbor list from the stored positions with a
    :class:`~nvalchemi.hooks.NeighborListHook` at ``BEFORE_FORWARD``.

    Reading ahead overlaps the next chunk's load with the current chunk's
    scoring and write, so it saves up to one load per chunk when the store is
    slow to read (a network or object store, or shared storage) or when
    per-sample validation dominates the load. The dataset's prefetch thread
    decodes and validates the chunk while the main thread launches the
    teacher's kernels, and a dataset that targets a CUDA device also moves
    every sample there from that thread; on a fast local store the contention
    can cost more than the load it hides, and labeling runs a little slower
    than the sequential loop. ``"auto"`` therefore measures both forms on the
    first chunks rather than assuming; the per-chunk writes, resume
    bookkeeping, and store contents are the same in every mode. A dataset
    that emits host-resident chunks, with *device* passed here for the move,
    keeps the transfer on the main thread and reads ahead faster than one that
    transfers from the prefetch thread.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}.")
    if prefetch not in (True, False, "auto"):
        raise ValueError(f"prefetch must be True, False, or 'auto'; got {prefetch!r}.")
    if prefetch is not False and not all(
        callable(getattr(dataset, name, None)) for name in _PREFETCH_METHODS
    ):
        if prefetch is True:
            warnings.warn(
                f"{type(dataset).__name__} offers no fused-prefetch surface, so "
                "labeling reads each chunk sequentially.",
                UserWarning,
                stacklevel=2,
            )
        prefetch = False

    declared = scorer_fields(scorer)
    if declared is not None:
        _reject_foreign_fields(declared, "A scorer's label_fields")

    state = _existing_store_state(store)
    if state is None and isinstance(store, (str, Path)) and Path(store).exists():
        raise ValueError(
            "Store path exists but is not a readable ALCHEMI Zarr store; got "
            f"{store!s}."
        )
    if state is not None and not resume:
        raise ValueError(
            f"Store already exists with {state.active!r} samples and resume is False; "
            "pass resume=True to continue labeling or write to a fresh store."
        )
    if state is not None and state.active != state.total:
        raise ValueError(
            f"Store holds {state.total - state.active!r} soft-deleted samples, so a "
            "resumed run cannot line up with the dataset; defragment the store or "
            "label into a fresh one."
        )

    total = len(dataset)
    start = state.active if state is not None else 0
    schema = state.schema if state is not None else None
    if start > total:
        raise ValueError(
            f"Store holds {start!r} samples but the dataset has {total!r}, so it was "
            "labeled from a different, longer dataset; resume against that dataset "
            "or label into a fresh store."
        )
    if start == total:
        return 0

    writer = AtomicDataZarrWriter(store)
    ephemeral = _DENSE_NEIGHBOR_KEYS if keep_neighbors else _NEIGHBOR_KEYS
    chunks = [
        list(range(begin, min(begin + batch_size, total)))
        for begin in range(start, total, batch_size)
    ]
    labeled = 0
    for indices, batch in zip(chunks, _chunk_batches(dataset, chunks, prefetch)):
        if device is not None:
            batch = batch.to(device)
        loaded_fields = frozenset(_batch_schema(batch))
        labels = scorer.label(batch)
        _reject_foreign_fields(labels, "Teacher labels")
        for field, (values, level) in labels.items():
            if level == "system":
                _ensure_system_group(batch)
            batch.add_key(
                field,
                _split_per_graph(batch, field, values, level),
                level=level,
                overwrite=True,
            )
        _strip_unstorable(batch, loaded_fields | frozenset(labels), ephemeral)
        outgoing = _batch_schema(batch)
        if schema is None:
            _check_storable_dtypes(outgoing)
            writer.write(batch)
            schema = outgoing
        else:
            _check_chunk_schema(schema, outgoing, indices)
            writer.append(batch)
        labeled += batch.num_graphs
    return labeled
