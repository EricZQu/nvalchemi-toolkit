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
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.structure_sampler import (
    FitPolicy,
    OrderedStructureSampler,
    StructureSource,
    WithinBudget,
)

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol

__all__ = ["FitPolicy", "InitialStructures", "InitialStructuresSource", "WithinBudget"]

InitialStructuresSource = StructureSource
"""The loop's historical name for :class:`~nvalchemi.dynamics.StructureSource`."""


def _dataset_spec_dict(dataset: BatchDatasetProtocol, field: str) -> dict[str, Any]:
    """Return the store reference a path-backed dataset round-trips as.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Dataset to reference. Only a dataset reading a filesystem or URI store
        can be named in a recipe; one holding its samples in memory cannot.
    field : str
        Name of the recipe field being serialized, quoted in the error.

    Returns
    -------
    dict[str, Any]
        ``{"path": ..., "device": ...}`` reference the rebuild reopens.

    Raises
    ------
    ValueError
        If *dataset* is not backed by a store a path names.
    """
    store = getattr(getattr(dataset, "reader", None), "store", None)
    if not isinstance(store, (str, Path)):
        raise ValueError(
            f"{field} is a {type(dataset).__name__} holding its samples in "
            "memory, which no recipe can name: a spec references a dataset by "
            "the store it reads. Write the samples to a store with "
            "nvalchemi.training.distillation.label_dataset (or an "
            "AtomicDataZarrWriter) and point the recipe at that path, or "
            f"re-supply {field} at construction."
        )
    return {"path": str(store), "device": str(getattr(dataset, "target_device", "cpu"))}


def _dataset_from_spec_dict(spec: Mapping[str, Any]) -> BatchDatasetProtocol:
    """Reopen the dataset :func:`_dataset_spec_dict` referenced.

    Parameters
    ----------
    spec : Mapping[str, Any]
        Reference produced by :func:`_dataset_spec_dict`.

    Returns
    -------
    BatchDatasetProtocol
        Dataset over the referenced store. The reader it opens stays open for
        the caller to close.

    Raises
    ------
    pydantic.ValidationError
        If *spec* names no store to read, or carries a key that is not part of
        a store reference.
    """
    from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset

    reference = _DatasetRef.model_validate(spec)
    return Dataset(AtomicDataZarrReader(reference.path), device=reference.device)


class _DatasetRef(BaseModel):
    """Store reference a recipe names one dataset by."""

    path: Annotated[
        str,
        Field(description="Filesystem path or URI of the store to read."),
    ]
    device: Annotated[
        str,
        Field(
            default="cpu",
            description="Device the dataset collates the rows it serves onto.",
        ),
    ] = "cpu"

    model_config = ConfigDict(extra="forbid")


class _InitialStructuresSpec(BaseModel):
    """Recipe block a :class:`InitialStructures` is rebuilt from.

    Validating the block before anything is opened refuses a budget that is
    not a positive count and a misspelled setting where a recipe is read rather
    than inside the run it describes — a misspelling in particular, since a
    source is unbudgeted by default and one that never reached a field silently
    generates under no budget at all.
    """

    dataset: Annotated[
        _DatasetRef,
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


def _required_structure_fields(dynamics: BaseDynamics) -> tuple[str, ...]:
    """Return the batch fields *dynamics* updates in place from its first step.

    A propagator primes its model outputs — ``BEFORE_COMPUTE``, ``compute``,
    ``AFTER_COMPUTE`` — before its first ``pre_update``, so the fields its
    ``__needs_keys__`` outputs land in need not be on the initial batch.
    Whatever it updates in place has to be — its ``__provides_keys__`` other
    than ``positions`` — plus ``atomic_masses`` for a propagator carrying
    momentum.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator the initial structures are propagated by.

    Returns
    -------
    tuple[str, ...]
        Sorted batch field names the initial structures have to carry.
    """
    fields = dynamics.__provides_keys__ - {"positions"}
    if "velocities" in fields:
        fields.add("atomic_masses")
    return tuple(sorted(fields))


def _check_structure_fields(state: Batch, dynamics: BaseDynamics) -> None:
    """Reject an initial batch the propagator cannot take its first step from.

    Parameters
    ----------
    state : Batch
        Batch the first segment would propagate from, or the one-row probe
        standing in for it at construction.
    dynamics : BaseDynamics
        Propagator the batch is seeded for.

    Raises
    ------
    ValueError
        If *state* is missing a field *dynamics* updates in place from its
        first step.
    """
    missing = [
        field for field in _required_structure_fields(dynamics) if field not in state
    ]
    if not missing:
        return
    raise ValueError(
        f"Initial structures must carry the fields {type(dynamics).__name__} "
        f"propagates from; got missing {missing!r}. It primes the model outputs "
        f"of __needs_keys__={sorted(dynamics.__needs_keys__)!r} itself before "
        f"its first step, but updates "
        f"__provides_keys__={sorted(dynamics.__provides_keys__)!r} in place from "
        "what the structures carry, so an initial structure has to arrive with "
        "all of those — AtomicData fills velocities and atomic_masses in itself "
        "unless a store dropped them, and a cell has to be carried because "
        "nothing fills that in for an aperiodic structure."
    )


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
            "dataset": _dataset_spec_dict(
                self.dataset, "OnPolicyConfig.initial_structures"
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
            _dataset_from_spec_dict(validated.dataset.model_dump()),
            max_atoms=validated.max_atoms,
            max_edges=validated.max_edges,
            max_batch_size=validated.max_batch_size,
        )
