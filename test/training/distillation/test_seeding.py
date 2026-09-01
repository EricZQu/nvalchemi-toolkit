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
"""Tests for :mod:`nvalchemi.training.distillation._seeding`."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.optimizers.fire import FIRE, FIREVariableCell
from nvalchemi.training.distillation._seeding import (
    _check_seed_fields,
    _check_seed_status,
    _seed_field_requirements,
    _SeedSampler,
    _stamp_bookkeeping,
)
from test.training.conftest import _build_atomic_data, _build_demo_model
from test.training.distillation.conftest import _build_periodic_batch


def _make_sized_dataset(sizes: Sequence[int]) -> InMemoryDataset:
    """Return a dataset whose structures carry the given atom counts, in order."""
    return InMemoryDataset(
        in_memory_batch=Batch.from_data_list(
            [
                _build_atomic_data(n_atoms=size, seed=index)
                for index, size in enumerate(sizes)
            ]
        )
    )


def _make_sampler(
    dataset: InMemoryDataset, *, consumed: int, recycle: bool = False
) -> _SeedSampler:
    """Return a sampler over *dataset* with the whole dataset as its envelope."""
    return _SeedSampler(
        dataset,
        consumed=consumed,
        recycle=recycle,
        max_atoms=sum(dataset.get_metadata(index)[0] for index in range(len(dataset))),
        max_batch_size=len(dataset),
    )


def _served_sizes(replacements: list) -> list[int]:
    """Return the atom count of every structure a request handed back."""
    return [int(data.positions.shape[0]) for data in replacements]


def _make_variable_cell_seed() -> Batch:
    """Return a periodic seed batch carrying what a variable-cell FIRE reads."""
    batch = _build_periodic_batch(n_systems=2, n_atoms=4)
    batch["forces"] = torch.zeros(batch.num_nodes, 3)
    batch["stress"] = torch.zeros(batch.num_graphs, 3, 3)
    return batch


class _RenamedForceFIRE(FIRE):
    """FIRE writing the model's forces to a batch field of its own naming."""

    _OUTPUT_KEY_TO_BATCH_ATTR = {"forces": "reference_forces"}


class TestSeedFieldRequirements:
    def test_a_fixed_cell_optimizer_reads_forces_and_the_momentum_state(self) -> None:
        """FIRE opens on forces it has not computed and on velocities it updates."""
        requirements = _seed_field_requirements(FIRE(_build_demo_model(), dt=0.1))

        assert requirements == ("atomic_masses", "forces", "velocities")

    def test_a_variable_cell_propagator_also_reads_the_cell(self) -> None:
        """A cell is state the propagator updates in place, and inverts first."""
        requirements = _seed_field_requirements(
            FIREVariableCell(_build_demo_model(), dt=0.1)
        )

        assert requirements == (
            "atomic_masses",
            "cell",
            "forces",
            "stress",
            "velocities",
        )

    def test_a_seed_batch_without_a_cell_is_rejected(self) -> None:
        """An aperiodic seed cannot start a variable-cell relaxation."""
        dynamics = FIREVariableCell(_build_demo_model(), dt=0.1)
        seed = _make_variable_cell_seed()
        del seed["cell"]

        with pytest.raises(ValueError, match="missing \\['cell'\\]"):
            _check_seed_fields(seed, dynamics)

    def test_a_periodic_seed_carrying_the_declared_fields_is_accepted(self) -> None:
        """The same batch with its cell passes, so the check is not blanket."""
        dynamics = FIREVariableCell(_build_demo_model(), dt=0.1)

        _check_seed_fields(_make_variable_cell_seed(), dynamics)

    def test_the_propagators_own_output_map_names_the_batch_field(self) -> None:
        """A propagator renaming an output is checked against the name it reads."""
        requirements = _seed_field_requirements(
            _RenamedForceFIRE(_build_demo_model(), dt=0.1)
        )

        assert "reference_forces" in requirements
        assert "forces" not in requirements


class TestSeedSampler:
    def test_the_cursor_serves_the_structures_the_seeded_batch_left_behind(
        self,
    ) -> None:
        """Structures come back in dataset order, starting where the batch stopped."""
        dataset = _make_sized_dataset([3, 4, 5, 6])
        sampler = _make_sampler(dataset, consumed=1)

        replacements = sampler.request_replacements_budget(max_count=2)

        assert _served_sizes(replacements) == [4, 5]
        assert [int(data.system_id) for data in replacements] == [1, 2]

    def test_a_whole_dataset_seed_hands_out_nothing_without_recycling(self) -> None:
        """The initial batch consumes every structure, so the cursor opens spent."""
        dataset = _make_sized_dataset([3, 4, 5, 6])
        sampler = _make_sampler(dataset, consumed=len(dataset))

        assert sampler.exhausted is True
        assert sampler.request_replacements_budget(max_count=2) == []

    def test_a_recycled_cursor_wraps_to_the_first_structure(self) -> None:
        """Recycling restarts at row 0 and keeps numbering system ids forward."""
        dataset = _make_sized_dataset([3, 4, 5, 6])
        sampler = _make_sampler(dataset, consumed=len(dataset), recycle=True)

        replacements = sampler.request_replacements_budget(max_count=2)

        assert sampler.exhausted is False
        assert _served_sizes(replacements) == [3, 4]
        assert [int(data.system_id) for data in replacements] == [4, 5]

    def test_an_over_budget_structure_is_skipped_for_the_next_that_fits(self) -> None:
        """A large structure at the cursor must not starve the ones behind it."""
        dataset = _make_sized_dataset([9, 3, 4])
        sampler = _make_sampler(dataset, consumed=0)

        replacements = sampler.request_replacements_budget(atom_budget=8, max_count=2)

        assert _served_sizes(replacements) == [3, 4]

    def test_a_budget_nothing_fits_returns_nothing_rather_than_looping(self) -> None:
        """A recycling cursor gives up after one pass when no structure fits."""
        dataset = _make_sized_dataset([9, 8, 7])
        sampler = _make_sampler(dataset, consumed=0, recycle=True)

        assert sampler.request_replacements_budget(atom_budget=2, max_count=3) == []

    def test_an_unbounded_request_stops_at_one_pass_over_the_dataset(self) -> None:
        """Without a slot count the request is capped at the dataset's length."""
        dataset = _make_sized_dataset([3, 4, 5])
        sampler = _make_sampler(dataset, consumed=0, recycle=True)

        replacements = sampler.request_replacements_budget()

        assert _served_sizes(replacements) == [3, 4, 5]

    def test_a_wrapped_cursor_never_serves_a_structure_twice_in_one_request(
        self,
    ) -> None:
        """Skipping to the end and wrapping must not re-serve the row already given."""
        dataset = _make_sized_dataset([4, 12, 13, 14])
        sampler = _make_sampler(dataset, consumed=0, recycle=True)

        replacements = sampler.request_replacements_budget(atom_budget=15, max_count=2)

        assert _served_sizes(replacements) == [4]


class TestSeedStatusContract:
    def _stamped_batch(self) -> Batch:
        """Return a two-system seed batch carrying the run's own bookkeeping."""
        state = Batch.from_data_list(
            [_build_atomic_data(n_atoms=3, seed=index) for index in range(2)]
        )
        _stamp_bookkeeping(state)
        return state

    def test_the_stamped_status_is_the_one_the_shorthand_migrates_off(self) -> None:
        """Seeds enter on status 0, which is what the fmax shorthand reads."""
        state = self._stamped_batch()

        assert state["status"].view(-1).tolist() == [0, 0]
        _check_seed_status(
            state,
            ConvergenceHook.from_fmax(0.05, source_status=0, target_status=1),
        )

    def test_a_criterion_aimed_at_an_unseeded_status_raises(self) -> None:
        """A criterion migrating off status 1 would freeze and graduate nothing."""
        state = self._stamped_batch()

        with pytest.raises(ValueError, match=r"source_status=1 against seed statuses"):
            _check_seed_status(
                state,
                ConvergenceHook.from_fmax(0.05, source_status=1, target_status=2),
            )
