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
"""Tests for multi-rank on-policy distillation."""

from __future__ import annotations

import os
import socket
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
import torch
from torch import distributed as dist

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.data.datapipes.multidataset import MultiDataset
from nvalchemi.dynamics.base import BaseDynamics, DistributedPipeline, FusedStage
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import CheckpointHook, TrainingStage, ValidationConfig
from nvalchemi.training.distillation import strategy as distillation_strategy
from nvalchemi.training.distillation.replay import build_mixed_loader
from nvalchemi.training.distillation.strategy import (
    _RANK_SEED_STRIDE,
    DistillationStrategy,
    _rank_local_propagator_seed,
)
from nvalchemi.training.distributed import destroy_distributed, init_distributed
from nvalchemi.training.hooks import DDPHook
from nvalchemi.training.runtime import unwrap_model
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import _build_direct_force_teacher
from test.training.distillation.test_on_policy import (
    _LANGEVIN_KWARGS,
    _REFERENCE_ELEMENT,
    _SEED_ELEMENT,
    _make_batch,
    _make_composed_propagator,
    _make_on_policy_strategy,
    _make_recording_student,
    _make_reference_dataset,
    _make_scorer,
    _make_seed_dataset,
    _make_system,
)

_WORKER_STEPS = 4
"""Optimizer steps every spawned rank takes, as two segments of two."""

_SEGMENT_KWARGS: dict[str, Any] = {
    "num_steps": _WORKER_STEPS,
    "steps_per_segment": 2,
    "segment_steps": 2,
}
"""Segment budget shared by the in-process and the spawned runs."""


def _free_port() -> int:
    """Return an available localhost TCP port for process-group setup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_distributed_strategy(
    *, rank: int = 0, world_size: int = 2, **overrides: Any
) -> DistillationStrategy:
    """Return an on-policy strategy that believes it is *rank* of *world_size*."""
    hooks = list(overrides.pop("hooks", []))
    return _make_on_policy_strategy(
        distributed_manager=_FakeManager(world_size=world_size, rank=rank),
        hooks=[DDPHook(), *hooks],
        **{**_SEGMENT_KWARGS, **overrides},
    )


def _make_replica_seed_dataset(n_systems: int = 4) -> InMemoryDataset:
    """Return *n_systems* byte-identical copies of one structure.

    Seeding a world with replicas is how a run asks for one trajectory per rank
    from a single geometry, and it is the shape where sharding alone separates
    nothing: only the propagator's own RNG stream can make the ranks' frames
    differ.
    """
    return InMemoryDataset(
        in_memory_batch=Batch.from_data_list(
            [_make_system(_SEED_ELEMENT, 500) for _ in range(n_systems)]
        )
    )


def _make_composed_anchor() -> MultiDataset:
    """Return a host-memory anchor split across a composition declaring no device.

    A :class:`MultiDataset` reports neither a ``target_device`` nor a resident
    batch, so its placement is only knowable by drawing from it.
    """
    scorer = _make_scorer(_build_direct_force_teacher(seed=2))
    return MultiDataset(
        _make_reference_dataset(scorer, 4, base_seed=700),
        _make_reference_dataset(scorer, 4, base_seed=740),
    )


def _seed_geometry(system: AtomicData) -> tuple[float, ...]:
    """Return the positions telling one structure of a seed dataset from another."""
    return tuple(round(float(value), 6) for value in system.positions.flatten())


def _loaded_seed_rows(state: Batch) -> list[int]:
    """Return the seed-dataset rows *state* was loaded from, matched by geometry."""
    rows = {
        _seed_geometry(system): index
        for index, system in enumerate(
            _make_seed_dataset().in_memory_batch.to_data_list()
        )
    }
    return sorted(rows[_seed_geometry(system)] for system in state.to_data_list())


def _make_annealing_propagator(student: BaseModelMixin) -> FusedStage:
    """Return a hot sampling stage composed with a cold one, both stochastic.

    A fused stage exposes no seed of its own — each thermostat holds its own,
    in a sub-stage — so this is the propagator a root-only seed probe misses.
    """
    hot = NVTLangevin(student, n_steps=2, **{**_LANGEVIN_KWARGS, "temperature": 1000.0})
    return hot + NVTLangevin(student, **_LANGEVIN_KWARGS)


def _replay_energies(strategy: DistillationStrategy) -> list[float]:
    """Return the teacher energy of every frame a run generated and stored."""
    frames = strategy.replay_buffer.dataset.in_memory_batch
    return sorted(round(float(value), 6) for value in frames.teacher_energy.flatten())


def _student_state(strategy: DistillationStrategy) -> dict[str, torch.Tensor]:
    """Return the trained student's state dict, detached on the host."""
    student = unwrap_model(strategy.models["student"])
    return {
        key: value.detach().cpu().clone() for key, value in student.state_dict().items()
    }


def _own_seed_rows(backend: str) -> dict[str, Any]:
    """Return the seed rows this rank claims and the ones its seed batch came from."""
    init_distributed(backend=backend)
    try:
        strategy = _make_on_policy_strategy(**_SEGMENT_KWARGS)
        state = strategy._seed_state(strategy.on_policy)
        return {
            "shard": list(strategy.seed_shard),
            "loaded": _loaded_seed_rows(state),
        }
    finally:
        destroy_distributed()


def _run_worker(
    rank: int,
    world_size: int,
    port: int,
    backend: str,
    local_rank: int,
    composed: bool,
    seeds_only: bool,
    result_queue: Any,
) -> None:
    """Run one rank of the segment loop and report what it produced."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(local_rank),
        }
    )
    torch.manual_seed(0)
    if seeds_only:
        result_queue.put((rank, _own_seed_rows(backend)))
        return
    recorder = _RecordingValidationHook()
    overrides: dict[str, Any] = {}
    if composed:
        student = _build_demo_model()
        overrides = {
            "student": student,
            "config_overrides": {
                "dynamics": _make_annealing_propagator(student),
                "seed_dataset": _make_replica_seed_dataset(),
            },
        }
    strategy = _make_on_policy_strategy(
        device="cuda" if backend == "nccl" else "cpu",
        hooks=[recorder, DDPHook(backend=backend, find_unused_parameters=True)],
        validation_config=ValidationConfig(
            validation_data=[_make_batch(_REFERENCE_ELEMENT, 2, base_seed=900)],
            every_n_steps=2,
        ),
        **_SEGMENT_KWARGS,
        **overrides,
    )
    strategy.run()
    result_queue.put(
        (
            rank,
            {
                "state": {
                    key: value.tolist()
                    for key, value in _student_state(strategy).items()
                },
                "energies": _replay_energies(strategy),
                "device": str(strategy.devices[0]),
                "steps": strategy.step_count,
                "validations": recorder.calls,
            },
        )
    )


def _run_ranks(
    world_size: int,
    *,
    backend: str = "gloo",
    local_ranks: tuple[int, ...] | None = None,
    composed: bool = False,
    seeds_only: bool = False,
) -> dict[int, dict[str, Any]]:
    """Spawn *world_size* ranks of the segment loop and collect their results.

    ``local_ranks`` names the node-local rank each process reports, which is
    what decides its device: ``None`` places rank ``r`` on device ``r`` (one
    node), while zeros everywhere is the one-rank-per-node placement.
    ``composed`` swaps the bare thermostat for a fused one and the distinct seed
    structures for replicas of a single geometry. ``seeds_only`` stops each rank
    after its seed batch, which is all the sharding contract needs and skips the
    generation and training the rest of the spawned runs pay for.
    """
    ranks = local_ranks or tuple(range(world_size))
    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_run_worker,
            args=(
                rank,
                world_size,
                port,
                backend,
                ranks[rank],
                composed,
                seeds_only,
                result_queue,
            ),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results: dict[int, dict[str, Any]] = {}
    try:
        for _ in range(world_size):
            rank, payload = result_queue.get(timeout=600)
            results[rank] = payload
        for proc in procs:
            proc.join(timeout=60)
            assert proc.exitcode == 0
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
    return results


def _assert_one_student(results: dict[int, dict[str, Any]]) -> None:
    """Assert every rank came out of the run holding the same student weights."""
    reference = results[min(results)]["state"]
    for result in results.values():
        for key, value in reference.items():
            torch.testing.assert_close(
                torch.as_tensor(result["state"][key]), torch.as_tensor(value)
            )


def _assert_disjoint_frames(results: dict[int, dict[str, Any]]) -> None:
    """Assert no two ranks generated — and paid the teacher for — the same frame."""
    generated = [frozenset(result["energies"]) for result in results.values()]
    assert all(generated)
    assert not frozenset.intersection(*generated)


class _FakeManager:
    """Distributed manager reporting a fixed world size, rank, and node-local rank."""

    def __init__(
        self, *, world_size: int = 2, rank: int = 0, local_rank: int | None = None
    ) -> None:
        """Report a world of *world_size* ranks, seen from *rank*."""
        self.world_size = world_size
        self.rank = rank
        self.global_rank = rank
        self.local_rank = rank if local_rank is None else local_rank
        self.device = torch.device("cpu")
        self.broadcast_buffers = False
        self.find_unused_parameters = True

    def is_initialized(self) -> bool:
        """Report communication as established for any multi-rank world."""
        return self.world_size > 1


class _RecordingDDP(torch.nn.Module):
    """Data-parallel stand-in counting the forwards routed through the wrapper."""

    calls: ClassVar[list[dict[str, Any]]] = []
    forwards: ClassVar[int] = 0

    def __init__(self, module: torch.nn.Module, **kwargs: Any) -> None:
        """Wrap *module* and record the data-parallel options it was given."""
        super().__init__()
        self.module = module
        type(self).calls.append(kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Count the pass whose gradients a real wrapper would all-reduce."""
        type(self).forwards += 1
        return self.module(*args, **kwargs)


class _RecordingValidationHook:
    """Count the validation passes a run closes."""

    frequency = 1
    stage = TrainingStage.AFTER_VALIDATION

    def __init__(self) -> None:
        """Start with no validation passes seen."""
        self.calls = 0

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Count one closed validation pass."""
        self.calls += 1


class _ModelProbe:
    """Report the strategy's live models at every forward pass."""

    frequency = 1
    stage = TrainingStage.BEFORE_FORWARD

    def __init__(self) -> None:
        """Start with an empty trace of wrapped model names."""
        self.wrapped: list[list[str]] = []

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        """Record which models a data-parallel wrapper is holding."""
        self.wrapped.append(
            sorted(
                key
                for key, model in ctx.workflow.models.items()
                if isinstance(model, _RecordingDDP)
            )
        )


class _StubPropagator:
    """Propagator stand-in exposing no RNG seed at all."""


class _PropertySeededPropagator:
    """Propagator stand-in exposing its seed through a getter-only property."""

    def __init__(self) -> None:
        """Hold the writable private seed the read-only public name forwards."""
        self._random_seed = 7

    @property
    def random_seed(self) -> int:
        """Return the seed under the public name the probe reaches first."""
        return self._random_seed


class _GeneratorKick(BaseDynamics):
    """Stochastic stage drawing its noise from a torch.Generator it holds."""

    __needs_keys__: set[str] = set()
    __provides_keys__: set[str] = {"velocities"}

    def __init__(self, model: BaseModelMixin, **kwargs: Any) -> None:
        """Hold a generator no seed-attribute probe can offset."""
        super().__init__(model=model, **kwargs)
        self.generator = torch.Generator().manual_seed(1234)

    def pre_update(self, batch: Batch) -> None:
        """Kick the velocities from a stream the rank offset cannot reach."""
        batch.velocities.add_(
            torch.randn(batch.velocities.shape, generator=self.generator)
        )

    def post_update(self, batch: Batch) -> None:
        """Leave the post-force half of the step alone."""


@pytest.fixture(autouse=True)
def _reset_recording_ddp() -> None:
    """Reset the data-parallel stand-in's counters before every test."""
    _RecordingDDP.calls.clear()
    _RecordingDDP.forwards = 0


class TestSeedSharding:
    def test_ranks_take_disjoint_strided_shards_that_cover_the_seeds(self) -> None:
        """Every seed structure is propagated once, by exactly one rank."""
        shards = [
            _make_distributed_strategy(rank=rank, world_size=3)._seed_shard(8)
            for rank in range(3)
        ]

        assert shards == [[0, 3, 6], [1, 4, 7], [2, 5]]
        assert sorted(index for shard in shards for index in shard) == list(range(8))

    def test_shards_follow_the_global_rank_rather_than_the_node_local_one(self) -> None:
        """Across nodes the node-local rank repeats, so sharding on it would too."""
        strategy = _make_on_policy_strategy(
            num_steps=2,
            distributed_manager=_FakeManager(world_size=4, rank=3, local_rank=1),
        )

        assert strategy._seed_shard(8) == [3, 7]

    def test_a_single_rank_run_propagates_every_seed(self) -> None:
        """Sharding is a no-op on one process, so single-rank runs are unchanged."""
        strategy = _make_on_policy_strategy(num_steps=2)

        assert strategy._seed_shard(4) == [0, 1, 2, 3]

    def test_the_seed_shard_names_the_rows_this_rank_propagates(self) -> None:
        """The rows a refill may draw from are public, disjoint, and complete."""
        strategies = [
            _make_distributed_strategy(rank=rank, world_size=3) for rank in range(3)
        ]
        num_seeds = len(strategies[0].on_policy.seed_dataset)
        shards = [strategy.seed_shard for strategy in strategies]

        assert shards == [
            tuple(strategy._seed_shard(num_seeds)) for strategy in strategies
        ]
        assert all(
            not set(left) & set(right) for left, right in combinations(shards, 2)
        )
        assert sorted(row for shard in shards for row in shard) == list(
            range(num_seeds)
        )

    def test_a_sampler_seeded_run_owns_no_seed_rows(self) -> None:
        """A sampler packs from its own dataset, so there are no rows to deal out."""
        strategy = _make_distributed_strategy(
            config_overrides={
                "seed_dataset": None,
                "sampler": SizeAwareSampler(
                    _make_seed_dataset(), max_atoms=64, max_batch_size=4
                ),
            }
        )

        assert strategy.seed_shard == ()

    def test_the_seed_shard_is_available_before_any_seed_state_is_drawn(self) -> None:
        """A restart refills without ever seeding, and still may only serve its rows."""
        strategy = _make_distributed_strategy(rank=1)

        assert strategy.seed_shard == (1, 3)

    def test_each_rank_generates_and_labels_its_own_frames(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frames, and the teacher passes paying for them, never leave their rank."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        energies = []
        for rank in range(2):
            strategy = _make_distributed_strategy(rank=rank)
            scorer = strategy.on_policy.teacher_scorer
            with patch.object(scorer, "label", wraps=scorer.label) as labeled:
                strategy.run()
            assert {call.args[0].num_graphs for call in labeled.call_args_list} == {2}
            energies.append(_replay_energies(strategy))

        assert len(energies[0]) == len(energies[1])
        assert not set(energies[0]) & set(energies[1])

    def test_fewer_seeds_than_ranks_is_rejected(self) -> None:
        """A rank dealt no structure of its own would have nothing to propagate."""
        strategy = _make_distributed_strategy(
            world_size=8, config_overrides={"seed_dataset": _make_seed_dataset(4)}
        )

        with pytest.raises(ValueError, match="at least one for each"):
            strategy.run()

    def test_a_size_aware_sampler_is_rejected_on_more_than_one_rank(self) -> None:
        """A sampler bin-packs from its own dataset with no view of the world."""
        strategy = _make_distributed_strategy(
            config_overrides={
                "seed_dataset": None,
                "sampler": SizeAwareSampler(
                    _make_seed_dataset(), max_atoms=64, max_batch_size=4
                ),
            }
        )

        with pytest.raises(ValueError, match="pass seed_dataset instead"):
            strategy.run()


class TestRankSeedStreams:
    def test_the_mixture_seed_is_moved_onto_this_rank_stride(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ranks sharing a base seed would otherwise draw the same anchor samples."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        strategy = _make_distributed_strategy(rank=1, config_overrides={"seed": 5})

        with patch.object(
            distillation_strategy, "build_mixed_loader", wraps=build_mixed_loader
        ) as built:
            strategy.run()

        assert {call.kwargs["seed"] for call in built.call_args_list} == {
            5 + _RANK_SEED_STRIDE
        }

    def test_a_single_rank_run_keeps_the_configured_mixture_seed(self) -> None:
        """The offset is zero on rank zero, so single-process draws are unchanged."""
        strategy = _make_on_policy_strategy(num_steps=2, config_overrides={"seed": 5})

        with patch.object(
            distillation_strategy, "build_mixed_loader", wraps=build_mixed_loader
        ) as built:
            strategy.run()

        assert {call.kwargs["seed"] for call in built.call_args_list} == {5}

    def test_seed_streams_follow_the_global_rank_rather_than_the_node_local_one(
        self,
    ) -> None:
        """Every node's rank zero would otherwise draw one stream."""
        strategy = _make_on_policy_strategy(
            num_steps=2,
            distributed_manager=_FakeManager(world_size=4, rank=3, local_rank=1),
        )

        assert strategy._rank_seed_offset() == 3 * _RANK_SEED_STRIDE

    def test_the_propagator_seed_is_moved_onto_this_rank_stride(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing but the loop puts the propagator on this rank's own stream."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        strategy = _make_distributed_strategy(rank=1)

        with patch.object(
            distillation_strategy,
            "_rank_local_propagator_seed",
            wraps=distillation_strategy._rank_local_propagator_seed,
        ) as entered:
            strategy.run()

        assert {call.args[1] for call in entered.call_args_list} == {_RANK_SEED_STRIDE}

    def test_a_stochastic_propagator_draws_from_its_own_rank_stream(self) -> None:
        """A counter-based thermostat would otherwise kick every rank identically."""
        dynamics = NVTLangevin(_make_recording_student(), **_LANGEVIN_KWARGS)
        base = dynamics._random_seed

        with _rank_local_propagator_seed(dynamics, _RANK_SEED_STRIDE):
            offset = dynamics._random_seed

        assert offset == base + _RANK_SEED_STRIDE
        assert dynamics._random_seed == base

    def test_a_composed_propagator_moves_every_stochastic_sub_stage(self) -> None:
        """A fused stage holds no seed itself; the thermostat drawing the noise does."""
        fused = _make_annealing_propagator(_make_recording_student())
        bases = [sub._random_seed for _, sub in fused.sub_stages]

        with _rank_local_propagator_seed(fused, _RANK_SEED_STRIDE):
            offsets = [sub._random_seed for _, sub in fused.sub_stages]

        assert not hasattr(fused, "_random_seed")
        assert offsets == [base + _RANK_SEED_STRIDE for base in bases]
        assert [sub._random_seed for _, sub in fused.sub_stages] == bases

    def test_one_integrator_composed_twice_is_strided_once(self) -> None:
        """Two sub-stages can be the same object, which must not take two strides."""
        dynamics = NVTLangevin(_make_recording_student(), **_LANGEVIN_KWARGS)
        base = dynamics._random_seed

        with _rank_local_propagator_seed(dynamics + dynamics, _RANK_SEED_STRIDE):
            offset = dynamics._random_seed

        assert offset == base + _RANK_SEED_STRIDE
        assert dynamics._random_seed == base

    def test_a_rank_keyed_stage_map_is_walked_as_a_pipeline_holds_it(self) -> None:
        """A pipeline keys its stages by rank, so its propagators sit in a mapping."""
        dynamics = NVTLangevin(_make_recording_student(), **_LANGEVIN_KWARGS)
        base = dynamics._random_seed

        with _rank_local_propagator_seed(
            DistributedPipeline(stages={0: dynamics}), _RANK_SEED_STRIDE
        ):
            offset = dynamics._random_seed

        assert offset == base + _RANK_SEED_STRIDE
        assert dynamics._random_seed == base

    def test_a_propagator_hiding_its_seed_is_left_alone(self) -> None:
        """Nothing is written to a propagator naming its seed somewhere else."""
        propagator = _StubPropagator()

        with _rank_local_propagator_seed(propagator, _RANK_SEED_STRIDE):
            pass

        assert not hasattr(propagator, "random_seed")

    def test_a_seed_read_through_a_property_is_moved_under_its_writable_name(
        self,
    ) -> None:
        """A getter-only public name would raise where the offsets are applied."""
        propagator = _PropertySeededPropagator()

        with _rank_local_propagator_seed(propagator, _RANK_SEED_STRIDE):
            offset = propagator.random_seed

        assert offset == 7 + _RANK_SEED_STRIDE
        assert propagator._random_seed == 7

    def test_rank_zero_leaves_the_propagator_seed_untouched(self) -> None:
        """A zero offset must not rewrite the seed a single-process run reads."""
        dynamics = NVTLangevin(_make_recording_student(), **_LANGEVIN_KWARGS)

        with _rank_local_propagator_seed(dynamics, 0):
            assert dynamics._random_seed == _LANGEVIN_KWARGS["random_seed"]


class TestSharedStreamReport:
    def test_a_generator_stage_beside_a_seeded_one_is_named(self) -> None:
        """One seed found in the tree must not pass the stages left beside it."""
        student = _build_demo_model()
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={
                "dynamics": _GeneratorKick(student, n_steps=1)
                + NVTLangevin(student, **_LANGEVIN_KWARGS)
            },
        )

        with pytest.warns(UserWarning, match="shared random stream") as reported:
            strategy._warn_shared_propagator_streams(strategy.on_policy)

        message = str(reported[0].message)
        assert "'_GeneratorKick'" in message
        assert "NVTLangevin" not in message
        assert "FusedStage" not in message

    def test_a_fully_seeded_composition_is_reported_as_nothing(self) -> None:
        """Every stage of an annealing propagator holds a seed of its own."""
        student = _build_demo_model()
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={"dynamics": _make_annealing_propagator(student)},
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy._warn_shared_propagator_streams(strategy.on_policy)

    def test_a_deterministic_stage_beside_a_seeded_one_is_not_named(self) -> None:
        """A relax-then-sample propagator has no stream to separate in its first half."""
        student = _build_demo_model()
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={
                "dynamics": DemoDynamics(student, n_steps=1)
                + NVTLangevin(student, **_LANGEVIN_KWARGS)
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy._warn_shared_propagator_streams(strategy.on_policy)

    def test_rank_zero_reports_a_propagator_it_could_not_separate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rank zero applies no offset, and is the rank whose stderr survives."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        student = _build_demo_model()
        strategy = _make_distributed_strategy(
            rank=0,
            student=student,
            config_overrides={"dynamics": DemoDynamics(student, n_steps=1)},
        )

        with pytest.warns(UserWarning, match="could not be moved onto per-rank"):
            strategy.run()

        assert strategy.step_count == _WORKER_STEPS

    def test_a_single_rank_run_reports_nothing(self) -> None:
        """One process draws one stream by definition, so the report would be noise."""
        student = _build_demo_model()
        strategy = _make_on_policy_strategy(
            num_steps=2,
            student=student,
            config_overrides={"dynamics": DemoDynamics(student, n_steps=1)},
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy._warn_shared_propagator_streams(strategy.on_policy)


class TestReplayPlacementAcrossRanks:
    def test_an_anchor_pinned_to_one_indexed_device_is_reported(self) -> None:
        """Every rank would stage its buffer and collate its mixture on that device."""
        strategy = _make_distributed_strategy(rank=1)
        strategy.devices = [torch.device("cuda:1")]
        strategy.reference_dataset.target_device = torch.device("cuda:0")

        with pytest.warns(UserWarning, match="not this rank's own device"):
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda:0")

    def test_an_index_less_anchor_device_is_left_alone(self) -> None:
        """'cuda' names whichever device the launcher pinned this rank to."""
        strategy = _make_distributed_strategy(rank=1)
        strategy.devices = [torch.device("cuda:1")]
        strategy.reference_dataset.target_device = torch.device("cuda")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda")

    def test_a_host_memory_anchor_is_left_alone(self) -> None:
        """A mixture collated on the host concentrates nothing on one accelerator."""
        strategy = _make_distributed_strategy(rank=1)
        strategy.devices = [torch.device("cuda:1")]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cpu")

    def test_a_single_rank_run_stages_where_the_anchor_is(self) -> None:
        """One process holds one buffer, so an indexed anchor concentrates nothing."""
        strategy = _make_on_policy_strategy(num_steps=2)
        strategy.reference_dataset.target_device = torch.device("cuda:0")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda:0")

    def test_a_composed_anchor_is_measured_rather_than_reported_as_unconstrained(
        self,
    ) -> None:
        """A composition declares no device, and host memory is a placement too."""
        strategy = _make_distributed_strategy(
            rank=1, reference_dataset=_make_composed_anchor()
        )
        strategy.devices = [torch.device("cuda:1")]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cpu")

    def test_an_anchor_moved_after_setup_is_staged_where_it_now_is(self) -> None:
        """The documented remedy runs after construction, so nothing may be memoized."""
        strategy = _make_distributed_strategy(rank=1)
        strategy.devices = [torch.device("cuda:1")]
        strategy.reference_dataset.target_device = torch.device("cuda:1")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device = strategy._resolve_replay_device(strategy.on_policy)

        assert device == torch.device("cuda:1")


class TestGradientSynchronization:
    def test_only_the_student_is_wrapped_for_the_all_reduce(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The frozen teacher is replicated per rank and stays out of the collective."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        probe = _ModelProbe()
        strategy = _make_distributed_strategy(hooks=[probe])

        strategy.run()

        assert _RecordingDDP.calls == [
            {
                "find_unused_parameters": True,
                "broadcast_buffers": False,
                "static_graph": False,
            }
        ]
        assert probe.wrapped and all(keys == ["student"] for keys in probe.wrapped)

    def test_generation_bypasses_the_wrapper_that_training_goes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only gradient steps cross ranks; propagating and labeling stay local."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        student = _make_recording_student()
        strategy = _make_distributed_strategy(student=student)

        strategy.run()

        assert _RecordingDDP.forwards == strategy.step_count == _WORKER_STEPS
        assert len(student.forwards) > _RecordingDDP.forwards

    def test_an_unsynchronized_student_is_rejected(self) -> None:
        """Without a wrapper every rank trains, and generates from, its own policy."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_FakeManager(world_size=2)
        )

        with pytest.raises(ValueError, match="gradients have to be synchronized"):
            strategy.run()

    def test_a_single_rank_run_needs_no_wrapper(self) -> None:
        """The contract is about ranks, so a lone process runs the loop bare."""
        strategy = _make_on_policy_strategy(
            num_steps=2, distributed_manager=_FakeManager(world_size=1)
        )

        strategy.run()

        assert strategy.step_count == 2


class TestRankLocalModelModes:
    def test_the_mode_context_is_given_the_module_the_wrapper_owns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Handed the wrapper instead, it would take a composition's mode off it."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        student = _build_demo_model()
        composed = _make_composed_propagator(student, _build_demo_model())
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )

        with patch.object(
            distillation_strategy,
            "_eval_propagator_model",
            wraps=distillation_strategy._eval_propagator_model,
        ) as entered:
            strategy.run()

        assert entered.call_args_list
        assert all(call.args[1] is student for call in entered.call_args_list)

    def test_a_composed_propagator_generates_in_eval_mode_under_a_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training goes through the wrapper, so nothing else takes the composition down."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        student = _build_demo_model()
        composed = _make_composed_propagator(student, _build_demo_model())
        strategy = _make_distributed_strategy(
            student=student,
            config_overrides={"dynamics": NVTLangevin(composed, **_LANGEVIN_KWARGS)},
        )

        strategy.run()

        assert composed.forwards
        assert all(reading == (False, False) for reading in composed.forwards)
        assert composed.training is True


class TestRankConsistentBookkeeping:
    def test_validation_runs_on_every_rank_while_only_rank_zero_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Validation all-reduces its metrics, so no rank may skip it."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        validated: list[int] = []
        written: list[list[str]] = []
        for rank in range(2):
            recorder = _RecordingValidationHook()
            checkpoints = tmp_path / f"rank{rank}"
            strategy = _make_distributed_strategy(
                rank=rank,
                hooks=[
                    recorder,
                    CheckpointHook(checkpoints, step_interval=2, async_save=False),
                ],
                validation_config=ValidationConfig(
                    validation_data=[_make_batch(_REFERENCE_ELEMENT, 2, base_seed=900)],
                    every_n_steps=2,
                ),
            )
            strategy.run()
            validated.append(recorder.calls)
            written.append(sorted(path.name for path in checkpoints.glob("*")))

        assert validated[0] == validated[1] > 0
        assert written[0]
        assert not written[1]

    def test_what_rank_zero_wrote_restores_unwrapped_and_resumes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A wrapper owns the student at save time, so a restart must not need one."""
        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _RecordingDDP)
        checkpoints = tmp_path / "rank0"
        strategy = _make_distributed_strategy(
            hooks=[CheckpointHook(checkpoints, step_interval=2, async_save=False)]
        )
        strategy.run()
        trained = _student_state(strategy)

        resumed = _make_distributed_strategy(num_steps=_WORKER_STEPS + 2)
        resumed.restore_checkpoint(checkpoints)
        restored = _student_state(resumed)
        resumed.run()

        assert set(restored) == set(trained)
        for key, value in trained.items():
            torch.testing.assert_close(restored[key], value)
        assert resumed.step_count == _WORKER_STEPS + 2


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_two_cpu_ranks_train_one_student_from_disjoint_trajectories() -> None:
    """Two real ranks self-label their own frames and end on one set of weights."""
    results = _run_ranks(2)

    assert set(results) == {0, 1}
    assert all(result["steps"] == _WORKER_STEPS for result in results.values())
    assert results[0]["validations"] == results[1]["validations"] > 0
    _assert_disjoint_frames(results)
    _assert_one_student(results)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_two_cpu_ranks_own_disjoint_seed_rows() -> None:
    """The rows a real rank reports owning are the rows its seed batch came from."""
    results = _run_ranks(2, seeds_only=True)

    assert set(results) == {0, 1}
    shards = [results[rank]["shard"] for rank in sorted(results)]
    assert [results[rank]["loaded"] for rank in sorted(results)] == shards
    assert not set(shards[0]) & set(shards[1])
    assert sorted(shards[0] + shards[1]) == list(range(len(_make_seed_dataset())))


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_a_composed_propagator_seeded_from_replicas_still_diverges_per_rank() -> None:
    """Sharding replicas separates nothing, so the sub-stage thermostats have to."""
    results = _run_ranks(2, composed=True)

    assert set(results) == {0, 1}
    _assert_disjoint_frames(results)
    _assert_one_student(results)


@pytest.mark.slow
@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_ranks_placed_one_per_node_train_one_student() -> None:
    """The multi-node shape: node-local rank zero everywhere, one global world."""
    results = _run_ranks(4, local_ranks=(0, 0, 0, 0))

    assert set(results) == {0, 1, 2, 3}
    _assert_disjoint_frames(results)
    _assert_one_student(results)


@pytest.mark.multigpu
def test_two_gpu_ranks_train_one_student_from_disjoint_trajectories() -> None:
    """The single-node placement: a teacher replica and a student rank per GPU."""
    results = _run_ranks(2, backend="nccl")

    assert set(results) == {0, 1}
    assert {result["device"] for result in results.values()} == {"cuda:0", "cuda:1"}
    _assert_disjoint_frames(results)
    _assert_one_student(results)


@pytest.mark.multigpu
@pytest.mark.slow
def test_four_gpu_ranks_train_one_student_from_disjoint_trajectories() -> None:
    """Scaling the node out changes the world size and nothing else."""
    results = _run_ranks(4, backend="nccl")

    assert set(results) == {0, 1, 2, 3}
    _assert_disjoint_frames(results)
    _assert_one_student(results)
