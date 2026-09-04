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
"""Seed source of an on-policy segment loop, shaped like a sampler.

:meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` graduates converged
graphs and backfills fresh ones through a five-member sampler surface. This
module adapts a plain seed dataset to that surface, and checks that its
structures carry what the propagator reads before its first force evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
    from nvalchemi.dynamics.base import ConvergenceHook


def _seed_field_requirements(dynamics: BaseDynamics) -> tuple[str, ...]:
    """Return the batch fields *dynamics* reads before its first force evaluation.

    A propagator opens its step with ``pre_update``, which runs on the outputs
    of the *previous* step: the fields its ``__needs_keys__`` model outputs
    populate have to be on the seed batch already, zero-filled if nothing has
    computed them yet. It also reads whatever it updates in place, which is its
    ``__provides_keys__`` state other than ``positions`` — ``velocities`` for
    the integrators and the fixed-cell optimizers, and ``cell`` on top of that
    for the variable-cell ones, which invert it before the first force
    evaluation. A propagator that carries momentum divides forces by masses, so
    it reads ``atomic_masses`` too.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator the seed structures are propagated by.

    Returns
    -------
    tuple[str, ...]
        Sorted batch field names the seed structures have to carry.
    """
    fields = {
        dynamics._OUTPUT_KEY_TO_BATCH_ATTR.get(key, key)
        for key in dynamics.__needs_keys__
    }
    fields |= dynamics.__provides_keys__ - {"positions"}
    if "velocities" in fields:
        fields.add("atomic_masses")
    return tuple(sorted(fields))


def _check_seed_fields(state: Batch, dynamics: BaseDynamics) -> None:
    """Reject a seed batch the propagator cannot take its first step from.

    Parameters
    ----------
    state : Batch
        Batch the first segment would propagate from.
    dynamics : BaseDynamics
        Propagator the batch is seeded for.

    Raises
    ------
    ValueError
        If *state* is missing a field *dynamics* reads before its first force
        evaluation.
    """
    missing = [
        field for field in _seed_field_requirements(dynamics) if field not in state
    ]
    if not missing:
        return
    raise ValueError(
        f"Seed structures must carry the fields {type(dynamics).__name__} "
        f"propagates from; got missing {missing!r}. It reads the batch fields of "
        f"__needs_keys__={sorted(dynamics.__needs_keys__)!r} before evaluating "
        f"the model for the first time, and updates "
        f"__provides_keys__={sorted(dynamics.__provides_keys__)!r} in place from "
        "them, so a seed structure has to arrive with all of them — zeros are "
        "enough for the model outputs, AtomicData fills velocities and "
        "atomic_masses in itself unless a store dropped them, and a cell has to "
        "be carried because nothing fills that in for an aperiodic structure."
    )


def _stamp_bookkeeping(state: Batch) -> None:
    """Give *state* the graph-level fields the refill cycle maintains.

    ``status`` is what a status-migrating
    :class:`~nvalchemi.dynamics.base.ConvergenceHook` writes and what
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` graduates on, and
    ``system_id`` numbers the structures the way
    :class:`~nvalchemi.dynamics.sampler.SizeAwareSampler` does, so the ids a
    backfill hands out continue the seeded ones. Both guards fire on every seed
    batch a segment loop builds, whichever source built it, because the loop
    strips the propagator's bookkeeping before stamping it: they stand for the
    invariant that a run numbers and statuses its own trajectories rather than
    for a batch that arrives legitimately carrying either.

    Parameters
    ----------
    state : Batch
        Seed batch, mutated in place.
    """
    if "status" not in state:
        state["status"] = torch.zeros(
            state.num_graphs, 1, dtype=torch.long, device=state.device
        )
    if "system_id" not in state:
        state["system_id"] = torch.arange(
            state.num_graphs, dtype=torch.long, device=state.device
        ).unsqueeze(-1)


def _check_seed_status(state: Batch, criterion: ConvergenceHook) -> None:
    """Reject a criterion that migrates off a status no seed graph holds.

    :meth:`~nvalchemi.dynamics.base.ConvergenceHook.__call__` migrates only the
    graphs sitting on its ``source_status``, so a criterion aimed at another one
    leaves the lifecycle inert in the worst way: nothing freezes, nothing
    graduates, and the same criterion installed as the detector keeps cutting
    segments short over structures that are still being propagated and
    re-captured. Nothing warns, because there is no exhaustion to warn about.

    Parameters
    ----------
    state : Batch
        Seed batch, already stamped with the run's own bookkeeping.
    criterion : ConvergenceHook
        Criterion driving the trajectory lifecycle.

    Raises
    ------
    ValueError
        If no seed graph carries the criterion's ``source_status``.
    """
    statuses = sorted({int(value) for value in state["status"].view(-1).tolist()})
    if criterion.source_status in statuses:
        return
    raise ValueError(
        "A converged graph migrates off the status its seed carries, and the "
        "run stamps that status itself rather than reading it from the seed "
        f"structures; got source_status={criterion.source_status!r} against "
        f"seed statuses {statuses!r}, so nothing would ever freeze or "
        "graduate. Pass source_status=0, or pass the fmax threshold itself and "
        "let the shorthand wire it up."
    )


class _SeedSampler:
    """Sampler-shaped view of a seed dataset, handing out structures in order.

    :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` reads five
    members off whatever sits on ``dynamics.sampler`` — ``max_atoms``,
    ``max_edges``, ``max_batch_size``, ``request_replacements_budget``, and
    ``exhausted`` — and this class is the smallest thing that answers them for a
    run seeded from a dataset rather than from a
    :class:`~nvalchemi.dynamics.sampler.SizeAwareSampler`. Structures are served
    sequentially from the cursor the seeded batch left behind, so no structure
    is propagated twice within one pass.

    That cursor opens *at the end* of an ordinary seed run: a ``seed_dataset``
    is propagated whole, so the initial batch consumed every structure and there
    is no remainder to serve. Without ``recycle``, this sampler therefore hands
    out nothing and exists to answer the surface ``refill_check`` requires —
    which raises outright on ``sampler is None`` — while the batch narrows one
    trajectory per graduation. With ``recycle`` the cursor wraps to the
    beginning and the run keeps its trajectory count.

    The size envelope is the seeded batch itself: ``max_batch_size`` is the
    number of trajectories the run started with, so a backfill never widens the
    frame past it, and ``max_atoms`` is the atom count it started with, so a
    backfill never grows it beyond the footprint the device already held.
    ``max_edges`` stays ``None`` deliberately: the edges of a live frame are the
    neighbor list a propagator rebuilds every step, while the edge count a
    dataset reports is whatever it stored, and budgeting the first against the
    second would reject every replacement of a run whose neighbor list is denser
    than its store.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Seed structures, indexed in the order they were stored.
    consumed : int
        Structures the initial batch already took off the front.
    recycle : bool
        Whether a cursor at the end wraps to the beginning instead of
        reporting the source exhausted.
    max_atoms : int
        Total atoms a refilled batch may hold.
    max_batch_size : int
        Total structures a refilled batch may hold.
    """

    def __init__(
        self,
        dataset: BatchDatasetProtocol,
        *,
        consumed: int,
        recycle: bool,
        max_atoms: int,
        max_batch_size: int,
    ) -> None:
        """Open the cursor past the structures the initial batch consumed."""
        self._dataset = dataset
        self._cursor = consumed
        self._recycle = recycle
        self._next_system_id = consumed
        self.max_atoms = max_atoms
        self.max_edges: int | None = None
        self.max_batch_size = max_batch_size

    @property
    def exhausted(self) -> bool:
        """Whether the seed source has no structure left to hand out."""
        return not self._recycle and self._cursor >= len(self._dataset)

    def request_replacements_budget(
        self,
        atom_budget: int | None = None,
        edge_budget: int | None = None,
        max_count: int | None = None,
    ) -> list[AtomicData]:
        """Return the next structures that fit the freed slot and atom budget.

        A structure too large for the budget is skipped rather than allowed to
        block the queue, the way
        :meth:`~nvalchemi.dynamics.sampler.SizeAwareSampler.request_replacements_budget`
        passes over a candidate that does not fit — the budget after a
        graduation is exactly what graduated, so on a heterogeneous seed set a
        large structure at the cursor would otherwise starve every refill behind
        it. The scan gives up after one pass over the dataset, counting every
        structure it reaches rather than only the ones it skipped: a recycling
        cursor that wrapped mid-scan would otherwise serve a structure it had
        already served in the same call, and two copies of one seed entering the
        batch together relax in lockstep into duplicate frames.

        Parameters
        ----------
        atom_budget : int | None, optional
            Atoms the graduated structures freed. Default ``None``
            (unconstrained).
        edge_budget : int | None, optional
            Edges the graduated structures freed. Default ``None``, and
            ignored either way because this sampler declares no edge budget.
        max_count : int | None, optional
            Slots the graduated structures freed. Default ``None``, which caps
            the request at one pass over the seed dataset.

        Returns
        -------
        list[AtomicData]
            Structures to append to the active batch, oldest cursor position
            first, each stamped with its own ``system_id``. Empty once the
            source is exhausted, or once nothing a pass over it reaches fits
            the atom budget.
        """
        replacements: list[AtomicData] = []
        atoms = atom_budget
        wanted = len(self._dataset) if max_count is None else max_count
        scanned = 0
        while len(replacements) < wanted and scanned < len(self._dataset):
            if self._cursor >= len(self._dataset):
                if not self._recycle:
                    break
                self._cursor = 0
            index = self._cursor
            self._cursor += 1
            scanned += 1
            num_atoms, _ = self._dataset.get_metadata(index)
            if atoms is not None and num_atoms > atoms:
                continue
            data, _ = self._dataset[index]
            data.add_system_property(
                "system_id",
                torch.tensor([[self._next_system_id]], dtype=torch.long),
            )
            self._next_system_id += 1
            replacements.append(data)
            if atoms is not None:
                atoms -= num_atoms
        return replacements
