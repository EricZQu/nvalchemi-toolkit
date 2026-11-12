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
"""
CUDA-graph capture safety of the dynamics ops.

Tests verify:
- the Warp-backed integrator ops are captured by ``torch.cuda.graph`` (the graph
  is not empty) and a replayed NVE / velocity-Verlet step is bitwise the eager one
- the device-seed Langevin half step reproduces the Python-int-seed op bit for
  bit, advances its seed, and draws fresh noise on every graph replay
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.dynamics._ops.langevin import (
    langevin_half_step,
    langevin_half_step_device_seed,
)
from nvalchemi.dynamics._ops.velocity_verlet import (
    vv_position_update,
    vv_velocity_finalize,
)

N = 64


def _system(device: str, dtype: torch.dtype = torch.float32):
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(N, 3, generator=g, dtype=dtype).to(device)
    vel = 0.1 * torch.randn(N, 3, generator=g, dtype=dtype).to(device)
    mass = (1.0 + torch.rand(N, generator=g, dtype=dtype)).to(device)
    dt = torch.full((1,), 0.05, dtype=dtype, device=device)
    bidx = torch.zeros(N, dtype=torch.int32, device=device)
    return pos, vel, mass, dt, bidx


def _force(pos: torch.Tensor) -> torch.Tensor:
    """Deterministic analytic force (pure torch, capture-safe)."""
    return -pos * (1.0 + 0.2 * pos.pow(2).sum(-1, keepdim=True))


def _vv_step(pos, vel, frc, mass, dt, bidx) -> None:
    vv_position_update(pos, vel, frc, mass, dt, bidx)
    frc.copy_(_force(pos))
    vv_velocity_finalize(vel, frc, mass, dt, bidx)


class TestCaptureSafeVelocityVerlet:
    def test_graph_is_not_empty_and_matches_eager(self, gpu_device):
        steps = 25
        pos, vel, mass, dt, bidx = _system(gpu_device)
        frc = _force(pos)
        # eager reference
        pe, ve, fe = pos.clone(), vel.clone(), frc.clone()
        for _ in range(steps):
            _vv_step(pe, ve, fe, mass, dt, bidx)
        # one captured step, replayed ``steps`` times (static buffers)
        ps, vs, fs = pos.clone(), vel.clone(), frc.clone()
        _vv_step(ps, vs, fs, mass, dt, bidx)  # warm-up: loads the Warp module
        ps.copy_(pos)
        vs.copy_(vel)
        fs.copy_(frc)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _vv_step(ps, vs, fs, mass, dt, bidx)
        ps.copy_(pos)
        vs.copy_(vel)
        fs.copy_(frc)
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.equal(ps, pos), "captured graph left the integrator out"
        for _ in range(steps - 1):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ps, pe)
        assert torch.equal(vs, ve)
        assert torch.equal(fs, fe)

    def test_nve_pre_post_update_capture(self, gpu_device):
        """The toolkit NVE integrator's own ``pre_update`` / ``post_update``."""
        from torch import nn

        from nvalchemi.data import AtomicData, Batch
        from nvalchemi.dynamics import NVE
        from nvalchemi.models.base import BaseModelMixin, ModelConfig

        class Toy(nn.Module, BaseModelMixin):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    active_outputs={"energy", "forces"},
                    autograd_outputs=frozenset(),
                    autograd_inputs=frozenset(),
                    required_inputs=frozenset(),
                    optional_inputs=frozenset(),
                    supports_pbc=False,
                    needs_pbc=False,
                    neighbor_config=None,
                )

            @property
            def embedding_shapes(self):
                return {}

            def compute_embeddings(self, data, **kwargs):
                raise NotImplementedError

            def forward(self, data, **kwargs):
                raise NotImplementedError

        def make() -> Batch:
            pos, vel, mass, _, _ = _system(gpu_device)
            data = AtomicData(
                positions=pos,
                atomic_numbers=torch.full((N,), 6, dtype=torch.long, device=gpu_device),
                cell=(torch.eye(3) * 50.0).unsqueeze(0).to(gpu_device),
                pbc=torch.zeros(1, 3, dtype=torch.bool, device=gpu_device),
                energy=torch.zeros(1, 1, device=gpu_device),
                forces=torch.zeros(N, 3, device=gpu_device),
            )
            data.use_default_masses()
            data.use_default_velocities()
            b = Batch.from_data_list([data], device=gpu_device)
            b.velocities.copy_(vel)
            b.forces.copy_(_force(b.positions))
            return b

        def step(dyn, b) -> None:
            dyn.pre_update(b)
            b.forces.copy_(_force(b.positions))
            dyn.post_update(b)

        steps = 10
        eager, graphed = make(), make()
        d1, d2 = NVE(Toy(), dt=1.0), NVE(Toy(), dt=1.0)
        d1._ensure_state_initialized(eager)
        d2._ensure_state_initialized(graphed)
        for _ in range(steps):
            step(d1, eager)
        p0 = graphed.positions.clone()
        v0, f0 = graphed.velocities.clone(), graphed.forces.clone()
        step(d2, graphed)  # warm-up
        graphed.positions.copy_(p0)
        graphed.velocities.copy_(v0)
        graphed.forces.copy_(f0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step(d2, graphed)
        graphed.positions.copy_(p0)
        graphed.velocities.copy_(v0)
        graphed.forces.copy_(f0)
        for _ in range(steps):
            graph.replay()
        torch.cuda.synchronize()
        assert not torch.equal(graphed.positions, p0)
        assert torch.equal(graphed.positions, eager.positions)
        assert torch.equal(graphed.velocities, eager.velocities)


class TestDeviceSeedLangevin:
    def _args(self, device: str, dtype: torch.dtype = torch.float32):
        pos, vel, mass, dt, bidx = _system(device, dtype)
        frc = _force(pos)
        kt = torch.full((1,), 0.0257, dtype=dtype, device=device)
        fr = torch.full((1,), 0.5, dtype=dtype, device=device)
        return pos, vel, frc, mass, dt, kt, fr, bidx

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_matches_int_seed_op_bitwise(self, device, dtype):
        pos, vel, frc, mass, dt, kt, fr, bidx = self._args(device, dtype)
        p1, v1 = pos.clone(), vel.clone()
        p2, v2 = pos.clone(), vel.clone()
        seed = torch.tensor([1234], dtype=torch.int32, device=device)
        langevin_half_step(p1, v1, frc, mass, dt, kt, fr, 1234, bidx)
        langevin_half_step_device_seed(p2, v2, frc, mass, dt, kt, fr, seed, bidx)
        assert torch.equal(p1, p2)
        assert torch.equal(v1, v2)

    def test_seed_advances_by_one_per_call(self, device):
        pos, vel, frc, mass, dt, kt, fr, bidx = self._args(device)
        seed = torch.tensor([7], dtype=torch.int32, device=device)
        for k in range(3):
            langevin_half_step_device_seed(pos, vel, frc, mass, dt, kt, fr, seed, bidx)
            assert int(seed.item()) == 8 + k

    def test_consecutive_calls_follow_the_int_seed_sequence(self, device):
        pos, vel, frc, mass, dt, kt, fr, bidx = self._args(device)
        pa, va, pb, vb = pos.clone(), vel.clone(), pos.clone(), vel.clone()
        seed = torch.tensor([100], dtype=torch.int32, device=device)
        for k in range(4):
            langevin_half_step(pa, va, frc, mass, dt, kt, fr, 100 + k, bidx)
            langevin_half_step_device_seed(pb, vb, frc, mass, dt, kt, fr, seed, bidx)
        assert torch.equal(pa, pb)
        assert torch.equal(va, vb)

    def test_int_seed_graph_repeats_device_seed_graph_does_not(self, gpu_device):
        """The reason for the device seed: an int seed is baked into the graph."""
        pos, vel, frc, mass, dt, kt, fr, bidx = self._args(gpu_device)
        zero_f = torch.zeros_like(frc)
        # int seed: warm-up, capture, two replays from the same state
        v_int = torch.zeros_like(vel)
        p_int = pos.clone()
        langevin_half_step(p_int, v_int, zero_f, mass, dt, kt, fr, 5, bidx)
        torch.cuda.synchronize()
        g_int = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_int):
            langevin_half_step(p_int, v_int, zero_f, mass, dt, kt, fr, 5, bidx)
        v_int.zero_()
        g_int.replay()
        first = v_int.clone()
        v_int.zero_()
        g_int.replay()
        torch.cuda.synchronize()
        assert torch.equal(first, v_int), "an int seed is expected to repeat"
        # device seed: same experiment, the noise must differ
        v_dev = torch.zeros_like(vel)
        p_dev = pos.clone()
        seed = torch.tensor([5], dtype=torch.int32, device=gpu_device)
        langevin_half_step_device_seed(
            p_dev, v_dev, zero_f, mass, dt, kt, fr, seed, bidx
        )
        torch.cuda.synchronize()
        seed.fill_(5)
        g_dev = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_dev):
            langevin_half_step_device_seed(
                p_dev, v_dev, zero_f, mass, dt, kt, fr, seed, bidx
            )
        seed.fill_(5)
        v_dev.zero_()
        g_dev.replay()
        first = v_dev.clone()
        v_dev.zero_()
        g_dev.replay()
        torch.cuda.synchronize()
        assert not torch.equal(first, v_dev)

    def test_graph_replays_equal_the_eager_seed_sequence(self, gpu_device):
        pos, vel, frc, mass, dt, kt, fr, bidx = self._args(gpu_device)
        zero_f = torch.zeros_like(frc)
        seed = torch.tensor([42], dtype=torch.int32, device=gpu_device)
        pg, vg = pos.clone(), vel.clone()
        langevin_half_step_device_seed(
            pg.clone(), vg.clone(), zero_f, mass, dt, kt, fr, seed, bidx
        )
        torch.cuda.synchronize()
        seed.fill_(42)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            langevin_half_step_device_seed(pg, vg, zero_f, mass, dt, kt, fr, seed, bidx)
        pg.copy_(pos)
        vg.copy_(vel)
        seed.fill_(42)
        pe, ve = pos.clone(), vel.clone()
        for k in range(5):
            graph.replay()
            langevin_half_step(pe, ve, zero_f, mass, dt, kt, fr, 42 + k, bidx)
        torch.cuda.synchronize()
        assert torch.equal(pg, pe)
        assert torch.equal(vg, ve)

    def test_replayed_noise_has_the_right_variance(self, gpu_device):
        """200 replays of the pure O step (zero force, no drift): <v^2> -> kT / m."""
        n = 4096
        dev = gpu_device
        pos = torch.zeros(n, 3, device=dev)
        vel = torch.zeros(n, 3, device=dev)
        frc = torch.zeros(n, 3, device=dev)
        mass = torch.ones(n, device=dev)
        dt = torch.full((1,), 0.05, device=dev)
        kt = torch.full((1,), 0.0257, device=dev)
        fr = torch.full((1,), 0.5, device=dev)
        bidx = torch.zeros(n, dtype=torch.int32, device=dev)
        seed = torch.tensor([1], dtype=torch.int32, device=dev)
        langevin_half_step_device_seed(
            pos.clone(), vel.clone(), frc, mass, dt, kt, fr, seed, bidx
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            langevin_half_step_device_seed(pos, vel, frc, mass, dt, kt, fr, seed, bidx)
        vel.zero_()
        for _ in range(200):
            graph.replay()
        torch.cuda.synchronize()
        # the half step ends on an O step, so after many replays v is stationary N(0, kT/m)
        assert vel.pow(2).mean().item() == pytest.approx(0.0257, rel=0.05)


class TestNVTLangevinSeed:
    def test_trajectory_matches_int_seed_ops(self, device):
        """``NVTLangevin`` now feeds the device seed; its noise stream is ``seed + step_count`` as before."""
        from torch import nn

        from nvalchemi.data import AtomicData, Batch
        from nvalchemi.dynamics import NVTLangevin
        from nvalchemi.dynamics._ops.langevin import langevin_finalize
        from nvalchemi.models.base import BaseModelMixin, ModelConfig

        class Toy(nn.Module, BaseModelMixin):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    active_outputs={"energy", "forces"},
                    autograd_outputs=frozenset(),
                    autograd_inputs=frozenset(),
                    required_inputs=frozenset(),
                    optional_inputs=frozenset(),
                    supports_pbc=False,
                    needs_pbc=False,
                    neighbor_config=None,
                )

            @property
            def embedding_shapes(self):
                return {}

            def compute_embeddings(self, data, **kwargs):
                raise NotImplementedError

            def forward(self, data, **kwargs):
                from collections import OrderedDict

                f = _force(data.positions)
                return OrderedDict(energy=f.new_zeros(data.num_graphs, 1), forces=f)

        pos, vel, mass, _, _ = _system(device)
        data = AtomicData(
            positions=pos.clone(),
            atomic_numbers=torch.full((N,), 6, dtype=torch.long, device=device),
            cell=(torch.eye(3) * 50.0).unsqueeze(0).to(device),
            pbc=torch.zeros(1, 3, dtype=torch.bool, device=device),
            energy=torch.zeros(1, 1, device=device),
            forces=torch.zeros(N, 3, device=device),
        )
        data.use_default_masses()
        data.use_default_velocities()
        batch = Batch.from_data_list([data], device=device)
        batch.velocities.copy_(vel)
        masses = batch.atomic_masses.clone()
        dyn = NVTLangevin(
            Toy(), dt=1.0, temperature=300.0, friction=0.01, random_seed=9
        )
        dyn.compute(batch)
        steps = 6
        # reference: the same BAOAB sequence with the Python-int-seed op
        pr, vr, fr_ = (
            batch.positions.clone(),
            batch.velocities.clone(),
            batch.forces.clone(),
        )
        dyn._ensure_state_initialized(batch)
        st = dyn._state
        for k in range(steps):
            langevin_half_step(
                pr,
                vr,
                fr_,
                masses,
                st.dt,
                st.temperature,
                st.friction,
                9 + k,
                batch.batch_idx.int(),
            )
            fr_ = _force(pr)
            langevin_finalize(vr, fr_, masses, st.dt, batch.batch_idx.int())
        dyn.run(batch, n_steps=steps)
        assert torch.allclose(batch.positions, pr, atol=1e-6)
        assert torch.allclose(batch.velocities, vr, atol=1e-6)


class TestWrapPositionsCapture:
    def test_wrap_positions_in_graph(self, gpu_device):
        from nvalchemi.hooks.periodic import wrap_positions_into_cell

        g = torch.Generator().manual_seed(1)
        pos = (torch.rand(32, 3, generator=g) * 30 - 5).to(gpu_device)
        cell = (torch.eye(3) * 10).unsqueeze(0).to(gpu_device)
        pbc = torch.ones(1, 3, dtype=torch.bool, device=gpu_device)
        bidx = torch.zeros(32, dtype=torch.long, device=gpu_device)
        ref = pos.clone()
        wrap_positions_into_cell(ref, cell, pbc, bidx)
        x = pos.clone()
        wrap_positions_into_cell(x.clone(), cell, pbc, bidx)  # warm-up
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrap_positions_into_cell(x, cell, pbc, bidx)
        x.copy_(pos)
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.equal(x, pos)
        assert torch.equal(x, ref)


class TestNeighborRebuildDetectionCapture:
    """Pins the upstream behaviour the hooks rely on: the rebuild-detection op launches on the torch stream."""

    def test_rebuild_flags_inside_graph(self, gpu_device):
        from nvalchemiops.torch.neighbors.rebuild_detection import (
            batch_neighbor_list_needs_rebuild,
        )

        g = torch.Generator().manual_seed(2)
        ref = torch.rand(2 * N, 3, generator=g).to(gpu_device)
        bidx = torch.repeat_interleave(torch.arange(2, dtype=torch.int32), N).to(gpu_device)
        cur = ref.clone()
        cur[N:] += 1.0  # system 1 moved far past the skin, system 0 did not

        def _run(r, c):
            return batch_neighbor_list_needs_rebuild(
                r, c, bidx, 0.25, update_reference_positions=True, num_systems=2
            )

        _run(ref.clone(), cur)  # warm-up: loads the Warp module
        torch.cuda.synchronize()
        ref_s = ref.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flags = _run(ref_s, cur)
        ref_s.copy_(ref)  # reset the buffer the capture mutates
        graph.replay()
        torch.cuda.synchronize()
        assert flags.tolist() == [False, True]  # the captured launch ran, and ran per system
        assert torch.equal(ref_s[:N], ref[:N])  # system 0 reference untouched
        assert torch.equal(ref_s[N:], cur[N:])  # system 1 reference advanced to the new positions
