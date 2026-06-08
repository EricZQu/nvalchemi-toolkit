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
"""JSON Lines reporting sink."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TextIO

from torch import distributed as dist

from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks.reporting._distributed import (
    normalize_rank_reduction,
    reduce_scalar_snapshot,
)
from nvalchemi.hooks.reporting._scalars import (
    ScalarCallback,
    collect_scalars,
)
from nvalchemi.hooks.reporting._state import ReportingState


class JSONLMode(str, Enum):
    """File mode used by :class:`JSONLReporter`.

    Attributes
    ----------
    APPEND : JSONLMode
        Append to an existing JSONL file, creating it when needed.
    WRITE : JSONLMode
        Truncate an existing JSONL file, creating it when needed.
    EXCLUSIVE : JSONLMode
        Create a new JSONL file and fail if it already exists.
    """

    APPEND = "a"
    WRITE = "w"
    EXCLUSIVE = "x"


class JSONLReporter:
    """Write scalar reporting snapshots as JSON Lines.

    Parameters
    ----------
    path : str | Path
        Destination ``.jsonl`` file.
    custom_scalars : Mapping[str, ScalarCallback] | None, optional
        Additional scalar callbacks passed to :func:`collect_scalars`.
    include_losses : bool, default True
        When ``True``, include loss scalars from the hook context.
    include_optimizer_lrs : bool, default True
        When ``True``, include optimizer learning rates from the hook context.
    mode : {"a", "w", "x"}, default "a"
        File open mode. ``"a"`` appends, ``"w"`` truncates, and ``"x"``
        requires that the file does not already exist.
    rank_reduction : torch.distributed.ReduceOp | {"none", "mean", "sum", "min", "max"} | None, default None
        Optional distributed reduction applied to scalars before writing. String
        values are normalized to :class:`torch.distributed.ReduceOp`. Reduction
        requires every rank to call this reporter; only rank zero writes the
        reduced snapshot.
    flush : bool, default True
        Flush the file handle after every record.
    mkdir : bool, default True
        Create the parent directory before opening the file.
    rank_zero_only : bool, default True
        Request rank-zero-only dispatch from :class:`ReportingOrchestrator`.
        When ``False`` and ``rank_reduction="none"``, ``path`` must contain
        ``"{rank}"`` or ``"{global_rank}"`` so every rank writes its own file.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        custom_scalars: Mapping[str, ScalarCallback] | None = None,
        include_losses: bool = True,
        include_optimizer_lrs: bool = True,
        mode: JSONLMode | str = JSONLMode.APPEND,
        rank_reduction: dist.ReduceOp | str | None = None,
        flush: bool = True,
        mkdir: bool = True,
        rank_zero_only: bool = True,
    ) -> None:
        try:
            self.mode = JSONLMode(mode)
        except ValueError as exc:
            raise ValueError(
                "JSONLReporter mode must be one of 'a', 'w', or 'x'."
            ) from exc
        self.rank_reduction = rank_reduction
        self._rank_reduction_op, _ = normalize_rank_reduction(rank_reduction)
        self.path = Path(path)
        self.custom_scalars = custom_scalars
        self.include_losses = include_losses
        self.include_optimizer_lrs = include_optimizer_lrs
        self.flush = flush
        self.mkdir = mkdir
        self._write_rank_zero_only = (
            rank_zero_only or self._rank_reduction_op is not None
        )
        self.rank_zero_only = rank_zero_only and self._rank_reduction_op is None
        self.requires_all_ranks = self._rank_reduction_op is not None
        self._file: TextIO | None = None
        self._open_path: Path | None = None
        if not self._write_rank_zero_only and not self._has_rank_token:
            raise ValueError(
                "JSONLReporter path must contain '{rank}' or '{global_rank}' "
                "when rank_zero_only=False and rank_reduction='none'."
            )

    def __enter__(self) -> JSONLReporter:
        """Return this reporter; files are opened lazily on first write."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the JSONL file."""
        self.close()

    def close(self) -> None:
        """Close the JSONL file if it is open."""
        if self._file is None:
            return
        self._file.close()
        self._file = None
        self._open_path = None

    def report(self, ctx: HookContext, stage: Enum, state: ReportingState) -> None:
        """Write one scalar snapshot.

        Parameters
        ----------
        ctx : HookContext
            Workflow hook context.
        stage : Enum
            Hook stage being reported.
        state : ReportingState
            Shared reporting state from the orchestrator.
        """
        snapshot = collect_scalars(
            ctx,
            stage,
            state,
            custom_scalars=self.custom_scalars,
            include_losses=self.include_losses,
            include_optimizer_lrs=self.include_optimizer_lrs,
        )
        if self._rank_reduction_op is not None:
            snapshot = reduce_scalar_snapshot(
                snapshot,
                self.rank_reduction,
                reporter_name=type(self).__name__,
            )
            if not self._is_rank_zero(ctx):
                return
        elif self._write_rank_zero_only and not self._is_rank_zero(ctx):
            return

        self._open(self._resolve_path(ctx.global_rank))
        if self._file is None:
            raise RuntimeError("JSONLReporter failed to open its output file.")
        self._file.write(json.dumps(snapshot.as_dict(), sort_keys=True))
        self._file.write("\n")
        if self.flush:
            self._file.flush()

    @property
    def _has_rank_token(self) -> bool:
        path = str(self.path)
        return "{rank}" in path or "{global_rank}" in path

    def _open(self, path: Path) -> None:
        if self._file is not None and self._open_path == path:
            return
        if self._file is not None:
            self.close()
        if self.mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open(self.mode.value, encoding="utf-8")
        self._open_path = path

    def _resolve_path(self, global_rank: int) -> Path:
        path = str(self.path)
        path = path.replace("{global_rank}", str(global_rank))
        path = path.replace("{rank}", str(global_rank))
        return Path(path)

    def _is_rank_zero(self, ctx: HookContext) -> bool:
        return ctx.global_rank == 0
