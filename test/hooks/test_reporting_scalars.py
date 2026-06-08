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
"""Tests for reporting scalar extraction and JSONL output."""

from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

import nvalchemi.hooks.reporting._distributed as reporting_distributed
from nvalchemi.hooks import TrainContext
from nvalchemi.hooks.reporting import (
    JSONLMode,
    JSONLReporter,
    ReportingState,
    ScalarSnapshot,
    collect_scalars,
    extract_loss_scalars,
    extract_scalars,
)
from nvalchemi.hooks.reporting._distributed import reduce_scalar_snapshot


class _ReportStage(Enum):
    AFTER_OPTIMIZER_STEP = auto()


def _ctx(
    *,
    global_rank: int = 2,
    loss: torch.Tensor | None = None,
    losses: dict[str, object] | None = None,
    optimizers: list[torch.optim.Optimizer] | None = None,
    lr_schedulers: list[torch.optim.lr_scheduler.LRScheduler | None] | None = None,
) -> TrainContext:
    return TrainContext(
        batch=object(),
        global_rank=global_rank,
        step_count=17,
        batch_count=19,
        epoch_step_count=3,
        epoch=5,
        loss=loss,
        losses=losses,
        optimizers=optimizers or [],
        lr_schedulers=lr_schedulers or [],
    )


def _fake_physicsnemo_modules(
    *,
    device: str | torch.device = "cpu",
    initialized: bool = True,
) -> tuple[ModuleType, ModuleType]:
    physicsnemo_module = ModuleType("physicsnemo")
    distributed_module = ModuleType("physicsnemo.distributed")

    class FakeDistributedManager:
        @classmethod
        def is_initialized(cls) -> bool:
            return initialized

        def __init__(self) -> None:
            self.device = torch.device(device)

    distributed_module.DistributedManager = FakeDistributedManager
    physicsnemo_module.distributed = distributed_module
    return physicsnemo_module, distributed_module


def _install_fake_physicsnemo_manager(
    *,
    device: str | torch.device = "cpu",
    initialized: bool = True,
) -> None:
    physicsnemo_module, distributed_module = _fake_physicsnemo_modules(
        device=device,
        initialized=initialized,
    )
    sys.modules["physicsnemo"] = physicsnemo_module
    sys.modules["physicsnemo.distributed"] = distributed_module


def _distributed_reduce_worker(rank: int, init_file: str, output_dir: str) -> None:
    _install_fake_physicsnemo_manager()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=30),
    )
    try:
        snapshot = ScalarSnapshot(
            stage="AFTER_OPTIMIZER_STEP",
            scalars={
                "loss/total": float(rank + 1),
                "metric": float((rank + 1) * 10),
            },
            global_rank=rank,
        )
        results: dict[str, object] = {}
        for reduction in (
            "mean",
            dist.ReduceOp.SUM,
            dist.ReduceOp.MIN,
            dist.ReduceOp.MAX,
        ):
            reduced = reduce_scalar_snapshot(
                snapshot,
                reduction,
                reporter_name="TestReporter",
            )
            name = reduction if isinstance(reduction, str) else str(reduction).lower()
            results[name.rsplit(".", maxsplit=1)[-1]] = reduced.scalars

        mismatched_snapshot = ScalarSnapshot(
            stage="AFTER_OPTIMIZER_STEP",
            scalars={f"rank/{rank}": float(rank)},
            global_rank=rank,
        )
        try:
            reduce_scalar_snapshot(
                mismatched_snapshot,
                dist.ReduceOp.SUM,
                reporter_name="TestReporter",
            )
        except ValueError as exc:
            results["mismatch"] = str(exc)
        else:
            results["mismatch"] = "missing-error"

        output_path = Path(output_dir) / f"rank-{rank}.json"
        output_path.write_text(json.dumps(results, sort_keys=True), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def test_extract_loss_scalars_handles_simple_training_losses() -> None:
    ctx = _ctx(
        loss=torch.tensor(1.5),
        losses={
            "energy": torch.tensor(0.4),
            "force": torch.tensor(0.1),
        },
    )

    scalars = extract_loss_scalars(ctx)

    assert scalars == pytest.approx(
        {
            "loss/total": 1.5,
            "loss/energy": 0.4,
            "loss/force": 0.1,
        }
    )


def test_extract_loss_scalars_handles_composed_loss_output() -> None:
    ctx = _ctx(
        loss=torch.tensor(99.0),
        losses={
            "total_loss": torch.tensor(3.0),
            "per_component_total": {
                "energy": torch.tensor(1.0),
                "force": torch.tensor([2.0]),
            },
            "per_component_weight": {"energy": 0.25, "force": 0.75},
            "per_component_raw_weight": {"energy": 1.0, "force": 3.0},
            "per_component_sample": {
                "energy": torch.tensor([1.0, 3.0]),
                "force": torch.tensor([2.0, 6.0]),
            },
        },
    )

    scalars = extract_loss_scalars(ctx)

    assert scalars == pytest.approx(
        {
            "loss/total": 3.0,
            "loss/energy/total": 1.0,
            "loss/force/total": 2.0,
            "loss/energy/weight": 0.25,
            "loss/force/weight": 0.75,
            "loss/energy/raw_weight": 1.0,
            "loss/force/raw_weight": 3.0,
            "loss/energy/sample_mean": 2.0,
            "loss/force/sample_mean": 4.0,
        }
    )


def test_extract_scalars_flattens_nested_mapping() -> None:
    scalars = extract_scalars(
        {
            "outer": {
                "inner": torch.tensor(2.0),
                "flag": True,
            },
            "plain": 3,
        },
        prefix="custom",
    )

    assert scalars == {
        "custom/outer/inner": 2.0,
        "custom/outer/flag": 1.0,
        "custom/plain": 3.0,
    }


def test_extract_scalars_rejects_non_scalar_tensor() -> None:
    with pytest.raises(ValueError, match="'vector' must be scalar"):
        extract_scalars({"vector": torch.tensor([1.0, 2.0])})


def test_collect_scalars_includes_metadata_custom_scalars_and_lrs() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.125)
    ctx = _ctx(loss=torch.tensor(2.5), optimizers=[optimizer])
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)

    snapshot = collect_scalars(
        ctx,
        _ReportStage.AFTER_OPTIMIZER_STEP,
        state,
        custom_scalars={
            "metric": lambda context, stage: torch.tensor(4.5),  # noqa: ARG005
            "nested": lambda context, stage: {"value": 6.0},  # noqa: ARG005
        },
    )

    assert snapshot.stage == "AFTER_OPTIMIZER_STEP"
    assert snapshot.event_count == 1
    assert snapshot.step_count == 17
    assert snapshot.batch_count == 19
    assert snapshot.epoch_step_count == 3
    assert snapshot.epoch == 5
    assert snapshot.global_rank == 2
    assert snapshot.elapsed_s is not None
    assert snapshot.scalars == pytest.approx(
        {
            "loss/total": 2.5,
            "optimizer/lr": 0.125,
            "metric": 4.5,
            "nested/value": 6.0,
        }
    )


def test_collect_scalars_extracts_scheduler_lrs() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.125)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    ctx = _ctx(optimizers=[optimizer], lr_schedulers=[scheduler])

    snapshot = collect_scalars(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)

    assert snapshot.scalars == pytest.approx(
        {
            "optimizer/lr": 0.125,
            "scheduler/lr": 0.125,
        }
    )


def test_collect_scalars_preserves_scheduler_slot_indices() -> None:
    first_parameter = torch.nn.Parameter(torch.tensor(1.0))
    second_parameter = torch.nn.Parameter(torch.tensor(2.0))
    first_optimizer = torch.optim.SGD([first_parameter], lr=0.125)
    second_optimizer = torch.optim.SGD([second_parameter], lr=0.25)
    scheduler = torch.optim.lr_scheduler.StepLR(
        second_optimizer,
        step_size=1,
        gamma=0.5,
    )
    ctx = _ctx(
        optimizers=[first_optimizer, second_optimizer],
        lr_schedulers=[None, scheduler],
    )

    snapshot = collect_scalars(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)

    assert snapshot.scalars == pytest.approx(
        {
            "optimizer/0/lr": 0.125,
            "optimizer/1/lr": 0.25,
            "scheduler/1/lr": 0.25,
        }
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed gloo backend is required",
)
def test_reduce_scalar_snapshot_uses_initialized_process_group(tmp_path) -> None:
    output_dir = tmp_path / "distributed-results"
    output_dir.mkdir()

    mp.spawn(
        _distributed_reduce_worker,
        args=(str(tmp_path / "distributed-init"), str(output_dir)),
        nprocs=2,
        join=True,
    )

    rank_results = [
        json.loads((output_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    expected = {
        "mean": {"loss/total": 1.5, "metric": 15.0},
        "sum": {"loss/total": 3.0, "metric": 30.0},
        "min": {"loss/total": 1.0, "metric": 10.0},
        "max": {"loss/total": 2.0, "metric": 20.0},
    }
    for results in rank_results:
        for reduction, expected_scalars in expected.items():
            assert results[reduction] == pytest.approx(expected_scalars)
        assert "same scalar keys" in results["mismatch"]


def test_reduce_scalar_snapshot_batches_scalar_collective(monkeypatch) -> None:
    all_reduce_sizes: list[int] = []

    def fake_all_gather_object(
        gathered_keys: list[tuple[str, ...]],
        keys: tuple[str, ...],
    ) -> None:
        gathered_keys[:] = [keys, keys]

    def fake_all_reduce(values: torch.Tensor, op: dist.ReduceOp) -> None:
        all_reduce_sizes.append(values.numel())
        values.mul_(2.0)

    monkeypatch.setattr(reporting_distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(reporting_distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(reporting_distributed.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        reporting_distributed.dist,
        "all_gather_object",
        fake_all_gather_object,
    )
    monkeypatch.setattr(reporting_distributed.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        reporting_distributed,
        "_collective_device",
        lambda: torch.device("cpu"),
    )
    snapshot = ScalarSnapshot(
        stage="AFTER_OPTIMIZER_STEP",
        scalars={"a": 1.0, "b": 2.0, "c": 3.0},
    )

    reduced = reduce_scalar_snapshot(
        snapshot,
        dist.ReduceOp.SUM,
        reporter_name="TestReporter",
    )

    assert all_reduce_sizes == [3]
    assert reduced.scalars == pytest.approx({"a": 2.0, "b": 4.0, "c": 6.0})


def test_collective_device_uses_physicsnemo_distributed_manager(monkeypatch) -> None:
    physicsnemo_module, distributed_module = _fake_physicsnemo_modules(device="cpu")
    monkeypatch.setitem(sys.modules, "physicsnemo", physicsnemo_module)
    monkeypatch.setitem(sys.modules, "physicsnemo.distributed", distributed_module)

    assert reporting_distributed._collective_device() == torch.device("cpu")


def test_collective_device_requires_initialized_physicsnemo_manager(
    monkeypatch,
) -> None:
    physicsnemo_module, distributed_module = _fake_physicsnemo_modules(
        initialized=False,
    )
    monkeypatch.setitem(sys.modules, "physicsnemo", physicsnemo_module)
    monkeypatch.setitem(sys.modules, "physicsnemo.distributed", distributed_module)

    with pytest.raises(RuntimeError, match="DistributedManager to be initialized"):
        reporting_distributed._collective_device()


def test_collect_scalars_can_include_training_progress() -> None:
    ctx = _ctx(loss=torch.tensor(2.5))
    ctx.workflow = SimpleNamespace(num_steps=20, num_epochs=10)
    state = ReportingState(started_at_s=time.monotonic() - 10.0)
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)

    snapshot = collect_scalars(
        ctx,
        _ReportStage.AFTER_OPTIMIZER_STEP,
        state,
        include_progress=True,
    )

    assert snapshot.scalars["training/progress_fraction"] == pytest.approx(17 / 20)
    assert snapshot.scalars["training/remaining_steps"] == pytest.approx(3.0)
    assert snapshot.scalars["training/target_epochs"] == pytest.approx(10.0)
    assert snapshot.scalars["training/steps_per_s"] > 0
    assert snapshot.scalars["training/eta_s"] > 0


def test_jsonl_reporter_writes_scalar_snapshot(tmp_path) -> None:
    output_path = tmp_path / "reports" / "metrics.jsonl"
    ctx = _ctx(global_rank=0, loss=torch.tensor(2.5))
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)
    reporter = JSONLReporter(
        output_path,
        custom_scalars={"metric": lambda context, stage: 9.0},  # noqa: ARG005
        mode="w",
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)
    reporter.close()
    reporter.close()

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["stage"] == "AFTER_OPTIMIZER_STEP"
    assert record["event_count"] == 1
    assert record["step_count"] == 17
    assert record["global_rank"] == 0
    assert record["scalars"] == pytest.approx(
        {
            "loss/total": 2.5,
            "metric": 9.0,
        }
    )


def test_jsonl_reporter_context_manager_closes_file(tmp_path) -> None:
    output_path = tmp_path / "metrics.jsonl"
    ctx = _ctx(global_rank=0)
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)

    with JSONLReporter(
        output_path,
        include_losses=False,
        include_optimizer_lrs=False,
        mode="w",
    ) as reporter:
        reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["scalars"] == {}


def test_jsonl_reporter_defaults_to_rank_zero_only(tmp_path) -> None:
    output_path = tmp_path / "metrics.jsonl"
    ctx = _ctx(global_rank=1, loss=torch.tensor(2.5))
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)
    reporter = JSONLReporter(output_path, mode="w")

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)

    assert reporter.rank_zero_only is True
    assert not output_path.exists()


def test_jsonl_reporter_requires_rank_token_for_all_rank_writes(tmp_path) -> None:
    with pytest.raises(ValueError, match="must contain '\\{rank\\}'"):
        JSONLReporter(tmp_path / "metrics.jsonl", rank_zero_only=False)


def test_jsonl_reporter_expands_rank_token_for_all_rank_writes(tmp_path) -> None:
    output_template = tmp_path / "metrics.rank-{rank}.jsonl"
    ctx = _ctx(global_rank=3, loss=torch.tensor(2.5))
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)
    reporter = JSONLReporter(
        output_template,
        mode=JSONLMode.WRITE,
        rank_zero_only=False,
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)
    reporter.close()

    output_path = tmp_path / "metrics.rank-3.jsonl"
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["global_rank"] == 3
    assert records[0]["scalars"] == pytest.approx({"loss/total": 2.5})


def test_jsonl_reporter_reduction_uses_all_rank_dispatch_and_rank_zero_write(
    tmp_path,
) -> None:
    output_path = tmp_path / "metrics.jsonl"
    ctx = _ctx(global_rank=0, loss=torch.tensor(2.5))
    state = ReportingState()
    state.mark_event(ctx, _ReportStage.AFTER_OPTIMIZER_STEP)
    reporter = JSONLReporter(
        output_path,
        mode="w",
        rank_reduction="mean",
    )

    reporter.report(ctx, _ReportStage.AFTER_OPTIMIZER_STEP, state)
    reporter.close()

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert reporter.rank_zero_only is False
    assert records[0]["scalars"] == pytest.approx({"loss/total": 2.5})


def test_jsonl_reporter_validates_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        JSONLReporter(tmp_path / "metrics.jsonl", mode="r")
