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
"""Ordered structure sampling for a propagator that admits structures over time.

A propagator run with in-flight batching reads structures twice over: once to
build the batch its first step propagates from, and again whenever a structure
graduates and the room it frees is backfilled with a fresh one. This module
serves both from a single position over the rows one rank owns, so a structure
is propagated once and a restart resumes where it stopped, and it decides what
fits a batch through one :class:`FitPolicy` predicate rather than a fixed set
of budget arguments.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import torch

from nvalchemi.data.datapipes.samplers import distributed_shard
from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol

__all__ = ["FitPolicy", "OrderedStructureSampler", "StructureSource", "WithinBudget"]


class FitPolicy(Protocol):
    """Decide whether the batch being drawn still fits once a candidate joins it.

    Called by :meth:`OrderedStructureSampler.draw` with the atom and edge totals
    the drawn structures would hold with the candidate included, so a policy is
    a stateless predicate over running totals: :class:`WithinBudget` bounds
    them, and a memory estimate or any other axis is one more class of this
    shape.
    """

    def __call__(self, num_atoms: int, num_edges: int) -> bool:
        """Return whether a drawn batch totaling *num_atoms* and *num_edges* fits."""
        ...


@dataclasses.dataclass(frozen=True)
class WithinBudget:
    """Fit policy admitting a batch while its totals stay within the given bounds.

    Parameters
    ----------
    atoms : int | None, optional
        Total atoms the drawn batch may hold. Default ``None`` (unbounded).
    edges : int | None, optional
        Total stored edges the drawn batch may hold. Default ``None``
        (unbounded). The edge count a dataset reports is whatever it stored,
        not the neighbor list a propagator rebuilds every step, so bound it only
        when the stored count is the one that matters.

    Examples
    --------
    >>> from nvalchemi.dynamics import WithinBudget
    >>> WithinBudget(atoms=10)(num_atoms=8, num_edges=0)
    True
    >>> WithinBudget(atoms=10)(num_atoms=12, num_edges=0)
    False
    """

    atoms: int | None = None
    edges: int | None = None

    def __call__(self, num_atoms: int, num_edges: int) -> bool:
        """Return whether *num_atoms* and *num_edges* both stay within the bounds."""
        return (self.atoms is None or num_atoms <= self.atoms) and (
            self.edges is None or num_edges <= self.edges
        )


@runtime_checkable
class StructureSource(Protocol):
    """Structures a propagator starts from, served in order from one position.

    These are the members an in-flight batching driver reads, so an object
    providing them can stand in for a dataset-backed sampler: :meth:`probe`
    hands construction-time checks one row; :meth:`shard` narrows the source
    to the rows one rank owns and reopens the position; :meth:`initial_batch`
    builds the batch the first step propagates from; :meth:`draw` serves the
    structures a backfill starts fresh from; :attr:`exhausted` reports a
    position with nothing left; and :meth:`state_dict` /
    :meth:`load_state_dict` carry the position through a restart.
    :class:`OrderedStructureSampler` is the reference implementation, over a
    dataset.

    Examples
    --------
    >>> from nvalchemi.dynamics import OrderedStructureSampler, StructureSource
    >>> isinstance(OrderedStructureSampler(dataset), StructureSource)  # doctest: +SKIP
    True
    """

    @property
    def exhausted(self) -> bool:
        """Whether the source has no structure left to hand out."""
        ...

    def shard(self, rank: int, world_size: int) -> None:
        """Narrow the source to the rows *rank* of *world_size* owns and reopen it."""
        ...

    def probe(self) -> Batch:
        """Return one row as the one-graph batch it would be propagated as."""
        ...

    def initial_batch(self) -> Batch:
        """Return the batch the first step propagates from, advancing the position."""
        ...

    def draw(
        self,
        *,
        limit: int | None = None,
        fits: FitPolicy | None = None,
        on_miss: Literal["stop", "skip"] = "stop",
    ) -> list[AtomicData]:
        """Serve the next structures while they pass *fits*."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return the position a restart resumes this source from."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Resume this source at the position *state* recorded."""
        ...


class OrderedStructureSampler:
    """Structures served in dataset order from one position, for in-flight batching.

    The reference :class:`StructureSource`, over a dataset. A run reads the
    sampler once to build the batch the first step propagates from, and a
    driver layered on top draws from it again for every structure it graduates
    and backfills; both go through the one position here, so no structure is
    propagated twice within one pass over the rows. An *unbudgeted* sampler
    serves every row it owns as one batch, so the trajectory count is the
    dataset's; a *budgeted* one packs the initial batch while structures fit
    and leaves the remainder in row order for :meth:`draw`. Both are one
    :meth:`draw` call under a :class:`WithinBudget` policy with
    ``on_miss="stop"``, while a backfill filling the room a graduation freed
    passes its own policy with ``on_miss="skip"``.

    :meth:`shard` narrows the sampler to the rows one rank owns, dealt strided
    and unpadded so the shards are disjoint and no structure is propagated
    twice; :attr:`next_row` counts positions in :attr:`rows`. A ``system_id``
    is not a position — ids number the structures the run has started, past
    any structure a policy passed over — so :attr:`next_system_id` is tracked
    separately from :attr:`next_row`.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Structures, indexed in the order they are served.
    max_atoms : int | None, optional
        Total atoms the initial batch may hold. Default ``None``, which serves
        every row this sampler owns.
    max_edges : int | None, optional
        Total stored edges the initial batch may hold. Default ``None``.
    max_batch_size : int | None, optional
        Total structures the initial batch may hold. Default ``None``.

    Raises
    ------
    ValueError
        If a budget is set and not positive.

    Examples
    --------
    >>> from nvalchemi.dynamics import OrderedStructureSampler, WithinBudget
    >>> sampler = OrderedStructureSampler(dataset, max_atoms=10_000)  # doctest: +SKIP
    >>> batch = sampler.initial_batch()  # doctest: +SKIP
    >>> fresh = sampler.draw(limit=2, fits=WithinBudget(atoms=64))  # doctest: +SKIP
    >>> sampler.next_row  # doctest: +SKIP
    6
    """

    def __init__(
        self,
        dataset: BatchDatasetProtocol,
        *,
        max_atoms: int | None = None,
        max_edges: int | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        """Open the sampler at the first row of *dataset*."""
        declared = {
            "max_atoms": max_atoms,
            "max_edges": max_edges,
            "max_batch_size": max_batch_size,
        }
        for name, value in declared.items():
            if value is not None and value <= 0:
                raise ValueError(
                    f"{type(self).__name__} {name} bounds a batch and must be "
                    f"positive when set; got {value!r}. Leave it None to serve "
                    "every row."
                )
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.max_edges = max_edges
        self.max_batch_size = max_batch_size
        self._rows: tuple[int, ...] = tuple(range(len(dataset)))
        self._next_row = 0
        self._next_system_id = 0
        self._rank = 0
        self._world_size = 1

    def __len__(self) -> int:
        """Return the number of rows this sampler owns."""
        return len(self._rows)

    @property
    def rows(self) -> tuple[int, ...]:
        """Dataset rows this sampler serves, in the order it serves them."""
        return self._rows

    @property
    def next_row(self) -> int:
        """Position in :attr:`rows` of the next structure served."""
        return self._next_row

    @property
    def next_system_id(self) -> int:
        """``system_id`` the next structure handed out is stamped with."""
        return self._next_system_id

    @property
    def exhausted(self) -> bool:
        """Whether the shard has no structure left to hand out."""
        return self._next_row >= len(self._rows)

    def shard(self, rank: int, world_size: int) -> None:
        """Narrow this sampler to the rows rank *rank* of *world_size* owns.

        Rows are dealt out strided by
        :func:`~nvalchemi.data.datapipes.distributed_shard` — rank ``r`` takes
        every ``world_size``-th structure from offset ``r`` — so the shards are
        disjoint, cover the dataset, and differ by at most one structure. The
        deal balances the count, not the work, so sort the dataset by atom
        count when structures differ widely in size. It is unpadded, since a
        padded structure would be propagated twice. The position and the next
        ``system_id`` are reset, so installing a shard on a sampler that has
        already run restarts it from its first row rather than resuming it.

        Parameters
        ----------
        rank : int
            Global rank claiming a shard.
        world_size : int
            Ranks the dataset is dealt across. A single-rank run gets the whole
            dataset, unchanged.

        Raises
        ------
        ValueError
            If *world_size* is not positive or *rank* falls outside it.
        """
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(
                "A shard is dealt to one rank of a world, so the rank has "
                f"to name a position in it; got rank={rank!r} of "
                f"world_size={world_size!r}."
            )
        self._rank = rank
        self._world_size = world_size
        self._rows = tuple(
            distributed_shard(
                list(range(len(self.dataset))),
                num_replicas=world_size,
                rank=rank,
                drop_last=False,
                pad=False,
            )
        )
        self._next_row = 0
        self._next_system_id = 0

    def probe(self) -> Batch:
        """Return the first row of the shard, as the one-graph batch it loads as.

        The row is loaded through the dataset's own collation rather than read
        as an :class:`~nvalchemi.data.AtomicData`, because that is what fills
        in the ``velocities`` and ``atomic_masses`` a store need not have kept
        and a propagator still reads.

        Returns
        -------
        Batch
            One graph, for a check that has to run before a run is paid for.

        Raises
        ------
        ValueError
            If this sampler owns no rows at all.
        """
        if not self._rows:
            raise ValueError(
                f"{type(self).__name__} has to hold at least one structure; got a "
                f"{type(self.dataset).__name__} of length "
                f"{len(self.dataset)!r} sharded to no rows."
            )
        return self.dataset.load_batches([[self._rows[0]]])[0]

    def initial_batch(self) -> Batch:
        """Return the batch the first step propagates from, advancing the position.

        The batch enters the run carrying none of the propagator's bookkeeping,
        so this sampler installs its own: a structure loaded from a store a
        dynamics sink filled arrives holding the ``status`` it graduated with,
        which :meth:`~nvalchemi.dynamics.base.BaseDynamics.step` would freeze
        at ``exit_status`` for a step that moves nothing.

        Returns
        -------
        Batch
            Initial batch, stamped with clean bookkeeping and numbered from
            :attr:`next_system_id`.

        Raises
        ------
        ValueError
            If the sampler has nothing left to serve, or if the structure at
            :attr:`next_row` is larger than the declared budget.
        """
        budget = WithinBudget(atoms=self.max_atoms, edges=self.max_edges)
        rows = self._scan_rows(
            limit=self.max_batch_size,
            fits=None if budget == WithinBudget() else budget,
            on_miss="stop",
        )
        if not rows:
            raise ValueError(
                "A run has to propagate something; got no structure at row "
                f"{self._next_row!r} of {len(self._rows)!r} fitting "
                f"max_atoms={self.max_atoms!r}, max_edges={self.max_edges!r}, "
                f"and max_batch_size={self.max_batch_size!r}. Widen the budget, "
                "or pass a dataset holding a structure that fits it."
            )
        state = self.dataset.load_batches([rows])[0]
        for key in BaseDynamics._bookkeeping_keys:
            if key in state:
                del state[key]
        self._stamp_bookkeeping(state)
        return state

    def draw(
        self,
        *,
        limit: int | None = None,
        fits: FitPolicy | None = None,
        on_miss: Literal["stop", "skip"] = "stop",
    ) -> list[AtomicData]:
        """Serve the next structures while they pass *fits*.

        Parameters
        ----------
        limit : int | None, optional
            Most structures to serve. Default ``None`` (the rest of the shard).
        fits : FitPolicy | None, optional
            Policy called with the atom and edge totals the drawn structures
            would hold with each candidate included. Default ``None`` (every
            structure fits).
        on_miss : {"stop", "skip"}, optional
            What a candidate that does not fit does to the scan. ``"stop"``
            ends the draw and leaves :attr:`next_row` on it, which is how an
            initial batch is packed; ``"skip"`` passes over it and goes on,
            which is how a backfill fills the room a graduation freed without
            one oversized structure starving every refill behind it. Default
            ``"stop"``.

        Returns
        -------
        list[AtomicData]
            Structures in row order, each stamped with its own ``system_id``.
            Empty once the shard is exhausted, or once the first candidate
            misses under ``on_miss="stop"``.
        """
        drawn: list[AtomicData] = []
        for index in self._scan_rows(limit=limit, fits=fits, on_miss=on_miss):
            data, _ = self.dataset[index]
            data.add_system_property(
                "system_id",
                torch.tensor([[self._next_system_id]], dtype=torch.long),
            )
            self._next_system_id += 1
            drawn.append(data)
        return drawn

    def state_dict(self) -> dict[str, int]:
        """Return the position a restart resumes this sampler from.

        Returns
        -------
        dict[str, int]
            ``next_row``, ``next_system_id``, and the ``rank`` and
            ``world_size`` both were counted in. The dataset and the declared
            budgets are configuration, not state, and are left out.
        """
        return {
            "next_row": self._next_row,
            "next_system_id": self._next_system_id,
            "rank": self._rank,
            "world_size": self._world_size,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        """Resume this sampler at the position *state* recorded.

        Parameters
        ----------
        state : Mapping[str, int]
            Bundle written by :meth:`state_dict`, on the shard this sampler is
            already narrowed to.

        Raises
        ------
        KeyError
            If *state* lacks ``next_row``, including a bundle written under
            the former ``cursor`` key, which is not read.
        ValueError
            If *state* was written for another rank or another world size,
            whose position counts rows in a different shard.
        """
        if "next_row" not in state:
            raise KeyError(
                f"{type(self).__name__} state is resumed from 'next_row'; got "
                f"keys {sorted(state)!r}."
            )
        rank = int(state["rank"])
        world_size = int(state["world_size"])
        if (rank, world_size) != (self._rank, self._world_size):
            raise ValueError(
                "The restart bundle's position was written for rank "
                f"{rank!r} of {world_size!r}; this rank is {self._rank!r} of "
                f"{self._world_size!r}. Restart on the world that wrote it, or "
                "start over from the first row."
            )
        self._next_row = int(state["next_row"])
        self._next_system_id = int(state["next_system_id"])

    def _scan_rows(
        self,
        *,
        limit: int | None,
        fits: FitPolicy | None,
        on_miss: Literal["stop", "skip"],
    ) -> list[int]:
        """Advance the position and return the rows the policy admitted."""
        rows: list[int] = []
        atoms = edges = 0
        while self._next_row < len(self._rows) and (limit is None or len(rows) < limit):
            index = self._rows[self._next_row]
            if fits is not None:
                num_atoms, num_edges = self.dataset.get_metadata(index)
                if not fits(atoms + num_atoms, edges + num_edges):
                    if on_miss == "stop":
                        break
                    self._next_row += 1
                    continue
                atoms += num_atoms
                edges += num_edges
            rows.append(index)
            self._next_row += 1
        return rows

    def _stamp_bookkeeping(self, state: Batch) -> None:
        """Give *state* the graph-level fields an in-flight run maintains.

        ``status`` is what a status-migrating
        :class:`~nvalchemi.dynamics.base.ConvergenceHook` writes and a driver
        graduates on, and ``system_id`` numbers the structures the way a
        backfill continues numbering them.
        """
        state["status"] = torch.zeros(
            state.num_graphs, 1, dtype=torch.long, device=state.device
        )
        state["system_id"] = torch.arange(
            self._next_system_id,
            self._next_system_id + state.num_graphs,
            dtype=torch.long,
            device=state.device,
        ).unsqueeze(-1)
        self._next_system_id += state.num_graphs
