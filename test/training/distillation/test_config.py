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
"""Tests for :mod:`nvalchemi.training.distillation.config`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.optimizers.fire import FIRE
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.training.distillation import InProcessTeacherScorer, OnPolicyConfig
from test.training.conftest import _build_demo_model
from test.training.distillation.conftest import _build_small_dataset


def _make_config_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid ``OnPolicyConfig`` payload with *overrides* applied."""
    kwargs: dict[str, Any] = {
        "dynamics": DemoDynamics(_build_demo_model(), n_steps=10, dt=0.5),
        "teacher_scorer": InProcessTeacherScorer(
            _build_demo_model(), ["energy", "forces"]
        ),
        "seed_dataset": _build_small_dataset(),
        "replay_ratio": 0.25,
        "steps_per_segment": 4,
    }
    kwargs.update(overrides)
    return kwargs


class TestOnPolicyConfigDefaults:
    def test_defaults_cover_every_optional_knob(self) -> None:
        """A minimal config resolves the segment, labeling, and replay defaults."""
        config = OnPolicyConfig(**_make_config_kwargs())

        assert config.batch_size == 8
        assert config.segment_steps == 100
        assert config.label_frequency == 100
        assert config.replay_capacity is None
        assert config.replay_eviction == "fifo"
        assert config.sampler is None
        assert config.weight_sync_frequency == 1

    def test_relaxation_optimizer_is_accepted_as_the_propagator(self) -> None:
        """The knob is ``dynamics``, so a FIRE relaxation drives the loop too."""
        propagator = FIRE(_build_demo_model(), dt=0.1, n_steps=10)

        config = OnPolicyConfig(**_make_config_kwargs(dynamics=propagator))

        assert config.dynamics is propagator

    def test_size_aware_sampler_is_accepted(self) -> None:
        """An inflight-batching sampler over the seeds round-trips unchanged."""
        seeds = _build_small_dataset()
        sampler = SizeAwareSampler(seeds, max_atoms=64, max_batch_size=4)

        config = OnPolicyConfig(
            **_make_config_kwargs(seed_dataset=seeds, sampler=sampler)
        )

        assert config.sampler is sampler


class TestOnPolicyConfigValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"replay_ratio": -0.1},
            {"replay_ratio": 1.5},
            {"segment_steps": 0},
            {"steps_per_segment": 0},
            {"batch_size": 0},
            {"label_frequency": 0},
            {"replay_capacity": 0},
            {"replay_eviction": "oldest"},
            {"unknown_knob": 1},
        ],
        ids=[
            "negative_ratio",
            "ratio_above_one",
            "zero_segment_steps",
            "zero_training_steps",
            "zero_batch_size",
            "zero_label_frequency",
            "zero_replay_capacity",
            "unknown_eviction",
            "extra_field",
        ],
    )
    def test_out_of_range_knobs_are_rejected(self, overrides: dict[str, Any]) -> None:
        """Every declarative constraint fails at construction, not mid-run."""
        with pytest.raises(ValidationError):
            OnPolicyConfig(**_make_config_kwargs(**overrides))

    def test_weight_sync_frequency_above_one_raises(self) -> None:
        """The reserved sync knob is held at 1 while the propagator shares a module."""
        with pytest.raises(ValidationError, match="weight_sync_frequency must be 1"):
            OnPolicyConfig(**_make_config_kwargs(weight_sync_frequency=2))

    def test_scorer_must_satisfy_the_teacher_scorer_protocol(self) -> None:
        """A stand-in without ``label`` and ``signals`` is not a scorer."""
        with pytest.raises(ValidationError):
            OnPolicyConfig(**_make_config_kwargs(teacher_scorer=object()))

    def test_async_knobs_are_not_configurable_yet(self) -> None:
        """``async_mode`` and ``staleness_threshold`` land with the remote scorer."""
        with pytest.raises(ValidationError):
            OnPolicyConfig(**_make_config_kwargs(async_mode=True))
