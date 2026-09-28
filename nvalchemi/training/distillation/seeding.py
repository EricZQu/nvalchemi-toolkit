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
"""Initial structures of an on-policy segment loop, as a recipe can name them.

The segment loop seeds its trajectories from an
:class:`~nvalchemi.dynamics.OrderedStructureSampler`; this module adds the one
thing a recipe needs from it, a spec round-trip through the store the
structures are read from, and keeps the loop's historical names importable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.dynamics.structure_sampler import (
    FitPolicy,
    OrderedStructureSampler,
    StructureSource,
    WithinBudget,
)
from nvalchemi.training._spec_utils import (
    DatasetRef,
    dataset_from_spec_dict,
    dataset_spec_dict,
)

__all__ = ["FitPolicy", "InitialStructures", "InitialStructuresSource", "WithinBudget"]

InitialStructuresSource = StructureSource
"""The loop's historical name for :class:`~nvalchemi.dynamics.StructureSource`."""

_IN_MEMORY_REMEDY = (
    "Write the samples to a store with "
    "nvalchemi.training.distillation.label_dataset (or an AtomicDataZarrWriter) "
    "and point the recipe at that path, or re-supply "
    "OnPolicyConfig.initial_structures at construction."
)
"""Sentence the in-memory refusal ends on, naming the distillation writer."""


class _InitialStructuresSpec(BaseModel):
    """Recipe block a :class:`InitialStructures` is rebuilt from.

    Validating the block before anything is opened refuses a budget that is
    not a positive count and a misspelled setting where a recipe is read rather
    than inside the run it describes — a misspelling in particular, since a
    source is unbudgeted by default and one that never reached a field silently
    generates under no budget at all.
    """

    dataset: Annotated[
        DatasetRef,
        Field(description="Store the initial structures are read from."),
    ]
    max_atoms: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Total atoms the initial batch may hold.",
        ),
    ] = None
    max_edges: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Total stored edges the initial batch may hold.",
        ),
    ] = None
    max_batch_size: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Total structures the initial batch may hold.",
        ),
    ] = None

    model_config = ConfigDict(extra="forbid")


class InitialStructures(OrderedStructureSampler):
    """An :class:`~nvalchemi.dynamics.OrderedStructureSampler` that also travels in a recipe.

    The sampler is the loop's reference :class:`InitialStructuresSource`; this
    subclass adds :meth:`to_spec_dict` and :meth:`from_spec_dict`, which name
    the sampler by the store its dataset reads and the budgets declared on it,
    so :class:`~nvalchemi.training.distillation.OnPolicyConfig` can be written
    to a recipe and rebuilt from one. A streaming source with no stable
    position to serialize leaves both out, stays runtime-only, and is refused
    by name wherever a recipe is written from a config holding it.

    Examples
    --------
    >>> from nvalchemi.training.distillation import InitialStructures, WithinBudget
    >>> structures = InitialStructures(dataset, max_atoms=10_000)  # doctest: +SKIP
    >>> state = structures.initial_batch()  # doctest: +SKIP
    >>> fresh = structures.draw(limit=2, fits=WithinBudget(atoms=64))  # doctest: +SKIP
    >>> rebuilt = InitialStructures.from_spec_dict(structures.to_spec_dict())  # doctest: +SKIP
    """

    def to_spec_dict(self) -> dict[str, Any]:
        """Return the JSON-ready reference a recipe names this source by.

        Returns
        -------
        dict[str, Any]
            The store the structures are read from and the budgets the caller
            declared. The position is state and belongs to a restart bundle
            instead, and the rank shard is a launcher fact that belongs to
            neither.

        Raises
        ------
        ValueError
            If the dataset holds its samples in memory, which no recipe
            can name.
        """
        return {
            "dataset": dataset_spec_dict(
                self.dataset,
                field="OnPolicyConfig.initial_structures",
                remedy=_IN_MEMORY_REMEDY,
            ),
            "max_atoms": self.max_atoms,
            "max_edges": self.max_edges,
            "max_batch_size": self.max_batch_size,
        }

    @classmethod
    def from_spec_dict(cls, spec: Mapping[str, Any]) -> InitialStructures:
        """Rebuild the source :meth:`to_spec_dict` described.

        Parameters
        ----------
        spec : Mapping[str, Any]
            Reference produced by :meth:`to_spec_dict`.

        Returns
        -------
        InitialStructures
            Source over the referenced store, opened at its first row.

        Raises
        ------
        pydantic.ValidationError
            If *spec* carries a key no source takes, names no store to read
            the structures from, or gives a budget that is not a positive count. It
            derives from :class:`ValueError`, so a caller that already reports
            a bad recipe reports this one the same way.
        """
        validated = _InitialStructuresSpec.model_validate(spec)
        return cls(
            dataset_from_spec_dict(
                validated.dataset.model_dump(),
                field="OnPolicyConfig.initial_structures",
            ),
            max_atoms=validated.max_atoms,
            max_edges=validated.max_edges,
            max_batch_size=validated.max_batch_size,
        )
