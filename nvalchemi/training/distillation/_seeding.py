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


def _seed_field_requirements(dynamics: BaseDynamics) -> tuple[str, ...]:
    """Return the batch fields *dynamics* reads before its first force evaluation.

    A propagator opens its step with ``pre_update``, which runs on the outputs
    of the *previous* step: the fields its ``__needs_keys__`` model outputs
    populate have to be on the seed batch already, zero-filled if nothing has
    computed them yet. A propagator that carries momentum — one declaring
    ``velocities`` in ``__provides_keys__`` — additionally divides forces by
    masses, so it reads both of those too.

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
        BaseDynamics._OUTPUT_KEY_TO_BATCH_ATTR.get(key, key)
        for key in dynamics.__needs_keys__
    }
    if "velocities" in dynamics.__provides_keys__:
        fields |= {"velocities", "atomic_masses"}
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
        "enough for the model outputs, and AtomicData fills velocities and "
        "atomic_masses in itself unless a store dropped them."
    )


def _stamp_bookkeeping(state: Batch) -> None:
    """Give *state* the graph-level fields the refill cycle maintains.

    ``status`` is what a status-migrating
    :class:`~nvalchemi.dynamics.base.ConvergenceHook` writes and what
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` graduates on, and
    ``system_id`` numbers the structures the way
    :class:`~nvalchemi.dynamics.sampler.SizeAwareSampler` does, so the ids a
    backfill hands out continue the seeded ones. A batch built by that sampler
    already carries both and is left alone.

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

    The size envelope is the seeded batch itself: ``max_batch_size`` is the
    number of trajectories the run started with, which keeps that count stable
    across graduations, and ``max_atoms`` is the atom count it started with, so
    a backfill never grows the frame beyond the footprint the device already
    held. ``max_edges`` stays ``None`` deliberately: the edges of a live frame
    are the neighbor list a propagator rebuilds every step, while the edge count
    a dataset reports is whatever it stored, and budgeting the first against the
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
            source is exhausted, or once the next structure no longer fits the
            atom budget.
        """
        replacements: list[AtomicData] = []
        atoms = atom_budget
        for _ in range(max_count if max_count is not None else len(self._dataset)):
            if self._cursor >= len(self._dataset):
                if not self._recycle:
                    break
                self._cursor = 0
            num_atoms, _ = self._dataset.get_metadata(self._cursor)
            if atoms is not None and num_atoms > atoms:
                break
            data, _ = self._dataset[self._cursor]
            data.add_system_property(
                "system_id",
                torch.tensor([[self._next_system_id]], dtype=torch.long),
            )
            self._cursor += 1
            self._next_system_id += 1
            replacements.append(data)
            if atoms is not None:
                atoms -= num_atoms
        return replacements
