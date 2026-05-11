# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Tests for the Stochastic Gumbel Annealing quantizer and its codec hooks."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    CheckerboardLatentCodec,
    EntropyBottleneckLatentCodec,
    GaussianConditionalLatentCodec,
    LRPGaussianLatentCodec,
)
from compressai.layers import CheckerboardMaskedConv2d
from compressai.models.mlic import MLICPlusPlus, MLICv2
from compressai.ops import SGAQuantizer


def _set_iter(sga: SGAQuantizer, it: int, total_it: int) -> None:
    """Helper to keep the SGA state mutation explicit in test bodies."""
    sga.set_iter(it, total_it)


class _ConstantResidual(nn.Module):
    def __init__(self, out_channels: int, value: float) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_full(
            (x.shape[0], self.out_channels, x.shape[2], x.shape[3]),
            self.value,
        )


class TestSGAQuantizer:
    def test_fallback_to_round_when_iter_unset(self) -> None:
        sga = SGAQuantizer()
        x = torch.tensor([0.3, -0.7, 1.4, -2.1])
        assert torch.equal(sga(x), torch.round(x))

    def test_set_iter_then_reset_falls_back_to_round(self) -> None:
        sga = SGAQuantizer()
        x = torch.tensor([0.3, -0.7, 1.4, -2.1])
        sga.set_iter(0, 100)
        sga.set_iter(None, None)
        assert torch.equal(sga(x), torch.round(x))

    def test_relaxed_sample_is_differentiable(self) -> None:
        sga = SGAQuantizer()
        sga.set_iter(0, 100)
        x = torch.randn(8, requires_grad=True)
        y = sga(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.norm() > 0

    def test_temperature_schedule_decays(self) -> None:
        sga = SGAQuantizer()
        # Constant during the t < t0 warm-up phase, then exponential decay.
        t_warmup = sga.annealed_temperature(0, 100)
        t_after = sga.annealed_temperature(80, 100)
        assert t_warmup == sga.upper_temperature
        assert t_after < t_warmup
        assert t_after > 0


class TestEntropyBottleneckLatentCodecSGA:
    def test_sga_mode_round_trip_with_iter_unset(self) -> None:
        sga = SGAQuantizer()
        codec = EntropyBottleneckLatentCodec(
            entropy_bottleneck=EntropyBottleneck(4),
            quantizer="sga",
            sga=sga,
        )
        codec.eval()
        x = torch.randn(1, 4, 8, 8)
        out = codec(x)
        # With iter unset, SGA falls back to round; y_hat must be exact round.
        medians = codec.entropy_bottleneck._get_medians()
        expected = torch.round(x - medians) + medians
        assert torch.allclose(out["y_hat"], expected, atol=1e-5)

    def test_sga_mode_grad_flows(self) -> None:
        sga = SGAQuantizer()
        codec = EntropyBottleneckLatentCodec(
            entropy_bottleneck=EntropyBottleneck(4),
            quantizer="sga",
            sga=sga,
        )
        codec.train()
        sga.set_iter(0, 100)
        x = nn.Parameter(torch.randn(1, 4, 8, 8))
        out = codec(x)
        loss = (-torch.log2(out["likelihoods"]["y"])).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.norm().item() > 0

    def test_invalid_quantizer_raises(self) -> None:
        try:
            EntropyBottleneckLatentCodec(
                entropy_bottleneck=EntropyBottleneck(4), quantizer="foo"
            )
        except ValueError:
            return
        raise AssertionError("Expected ValueError for invalid quantizer")

    def test_sga_without_module_raises(self) -> None:
        try:
            EntropyBottleneckLatentCodec(
                entropy_bottleneck=EntropyBottleneck(4), quantizer="sga"
            )
        except ValueError:
            return
        raise AssertionError("Expected ValueError for sga without module")


class TestGaussianConditionalLatentCodecSGA:
    @staticmethod
    def _ctx_params(
        batch: int = 1,
        channels: int = 4,
        height: int = 8,
        width: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scales = torch.rand(batch, channels, height, width) + 0.5
        means = torch.randn(batch, channels, height, width)
        return torch.cat([scales, means], dim=1), means

    def test_sga_mode_round_trip_with_iter_unset(self) -> None:
        sga = SGAQuantizer()
        codec = GaussianConditionalLatentCodec(quantizer="sga", sga=sga).eval()
        y = torch.randn(1, 4, 8, 8)
        ctx_params, means = self._ctx_params()
        out = codec(y, ctx_params)
        expected = torch.round(y - means) + means
        assert torch.allclose(out["y_hat"], expected, atol=1e-5)

    def test_sga_mode_grad_flows(self) -> None:
        sga = SGAQuantizer()
        codec = GaussianConditionalLatentCodec(quantizer="sga", sga=sga).train()
        sga.set_iter(0, 100)
        y = nn.Parameter(torch.randn(1, 4, 8, 8))
        ctx_params, _ = self._ctx_params()
        out = codec(y, ctx_params)
        loss = (
            out["y_hat"].pow(2).mean() + (-torch.log2(out["likelihoods"]["y"])).mean()
        )
        loss.backward()
        assert y.grad is not None
        assert y.grad.norm().item() > 0

    def test_invalid_quantizer_raises(self) -> None:
        try:
            GaussianConditionalLatentCodec(quantizer="foo")
        except ValueError:
            return
        raise AssertionError("Expected ValueError for invalid quantizer")

    def test_sga_without_module_raises(self) -> None:
        try:
            GaussianConditionalLatentCodec(quantizer="sga")
        except ValueError:
            return
        raise AssertionError("Expected ValueError for sga without module")


class TestLRPGaussianLatentCodecSGA:
    def test_sga_mode_is_inherited_before_lrp(self) -> None:
        sga = SGAQuantizer()
        lrp_value = 0.25
        lrp_scale = 0.5
        codec = LRPGaussianLatentCodec(
            lrp_transform=_ConstantResidual(4, lrp_value),
            lrp_scale=lrp_scale,
            quantizer="sga",
            sga=sga,
        ).eval()
        y = torch.randn(1, 4, 8, 8)
        scales = torch.rand(1, 4, 8, 8) + 0.5
        means = torch.randn(1, 4, 8, 8)
        ctx_params = torch.cat([scales, means], dim=1)
        out = codec(y, ctx_params)

        residual = lrp_scale * torch.tanh(torch.tensor(lrp_value))
        expected = torch.round(y - means) + means + residual
        assert torch.allclose(out["y_hat"], expected, atol=1e-5)

    def test_sga_mode_grad_flows_through_lrp_output(self) -> None:
        sga = SGAQuantizer()
        codec = LRPGaussianLatentCodec(
            lrp_transform=_ConstantResidual(4, 0.1),
            quantizer="sga",
            sga=sga,
        ).train()
        sga.set_iter(0, 100)
        y = nn.Parameter(torch.randn(1, 4, 8, 8))
        scales = torch.rand(1, 4, 8, 8) + 0.5
        means = torch.randn(1, 4, 8, 8)
        ctx_params = torch.cat([scales, means], dim=1)
        out = codec(y, ctx_params)
        loss = (
            out["y_hat"].pow(2).mean() + (-torch.log2(out["likelihoods"]["y"])).mean()
        )
        loss.backward()
        assert y.grad is not None
        assert y.grad.norm().item() > 0


class TestCheckerboardLatentCodecSGA:
    @staticmethod
    def _make(
        *,
        quantizer: str = "ste",
        sga: Optional[SGAQuantizer] = None,
        forward_method: str = "twopass",
    ) -> CheckerboardLatentCodec:
        y_channels = 4
        side_channels = 6
        ctx_channels = 5
        return CheckerboardLatentCodec(
            latent_codec={
                "y": GaussianConditionalLatentCodec(
                    quantizer="ste",
                    scale_table=[0.11, 0.5, 1.0, 2.0, 4.0],
                ),
            },
            context_prediction=CheckerboardMaskedConv2d(
                y_channels, ctx_channels, kernel_size=5, stride=1, padding=2
            ),
            entropy_parameters=nn.Conv2d(
                ctx_channels + side_channels, 2 * y_channels, 1
            ),
            forward_method=forward_method,
            quantizer=quantizer,
            sga=sga,
        )

    def test_sga_mode_matches_ste_with_iter_unset(self) -> None:
        torch.manual_seed(0)
        ste_codec = self._make(quantizer="ste").eval()
        sga_codec = self._make(quantizer="sga", sga=SGAQuantizer()).eval()
        sga_codec.load_state_dict(ste_codec.state_dict())

        y = torch.randn(1, 4, 8, 8)
        side_params = torch.randn(1, 6, 8, 8)
        with torch.no_grad():
            ste_out = ste_codec(y, side_params)
            sga_out = sga_codec(y, side_params)

        assert torch.allclose(sga_out["y_hat"], ste_out["y_hat"], atol=1e-5)
        assert torch.allclose(
            sga_out["likelihoods"]["y"],
            ste_out["likelihoods"]["y"],
            atol=1e-5,
        )

    def test_sga_mode_grad_flows_through_checkerboard_y_hat(self) -> None:
        torch.manual_seed(1)
        sga = SGAQuantizer()
        codec = self._make(quantizer="sga", sga=sga).train()
        sga.set_iter(0, 100)
        y = nn.Parameter(torch.randn(1, 4, 8, 8))
        side_params = torch.randn(1, 6, 8, 8)
        out = codec(y, side_params)
        loss = (
            out["y_hat"].pow(2).mean() + (-torch.log2(out["likelihoods"]["y"])).mean()
        )
        loss.backward()
        assert y.grad is not None
        assert y.grad.norm().item() > 0

    def test_invalid_quantizer_raises(self) -> None:
        try:
            self._make(quantizer="foo")
        except ValueError:
            return
        raise AssertionError("Expected ValueError for invalid quantizer")

    def test_sga_without_module_raises(self) -> None:
        try:
            self._make(quantizer="sga")
        except ValueError:
            return
        raise AssertionError("Expected ValueError for sga without module")

    def test_sga_onepass_raises(self) -> None:
        codec = self._make(
            quantizer="sga",
            sga=SGAQuantizer(),
            forward_method="onepass",
        )
        try:
            codec(torch.randn(1, 4, 8, 8), torch.randn(1, 6, 8, 8))
        except ValueError:
            return
        raise AssertionError("Expected ValueError for onepass SGA")


class TestMlicSGARefine:
    def test_set_sga_mode_propagates_to_all_leaves(self) -> None:
        m = MLICPlusPlus(N=8, M=16, slice_num=4, context_window=3)
        sga = SGAQuantizer()
        m.set_sga_mode(sga)
        # All four MultiContextCheckerboardLatentCodec leaves + one
        # EntropyBottleneckLatentCodec for z share the same SGA instance.
        sga_ids = set()
        for module in m.modules():
            if hasattr(module, "sga") and module.sga is not None:
                sga_ids.add(id(module.sga))
        assert sga_ids == {id(sga)}

    def test_set_sga_mode_none_restores_defaults(self) -> None:
        m = MLICPlusPlus(N=8, M=16, slice_num=4, context_window=3)
        sga = SGAQuantizer()
        m.set_sga_mode(sga)
        m.set_sga_mode(None)
        z_codec = m.latent_codec.latent_codec["z"]
        assert z_codec.quantizer == "noise"
        assert z_codec.sga is None
        for module in m.modules():
            if module.__class__.__name__ == "MultiContextCheckerboardLatentCodec":
                assert module.quantizer == "ste"
                assert module.sga is None

    def test_refine_extract_returns_y_z_shapes(self) -> None:
        m = MLICPlusPlus(N=8, M=16, slice_num=4, context_window=3)
        m.eval()
        x = torch.randn(1, 3, 64, 64)
        y, z = m.refine_extract(x)
        assert y.shape == (1, 16, 4, 4)
        assert z.shape == (1, 8, 1, 1)

    def test_refine_forward_no_sga_matches_full_forward(self) -> None:
        torch.manual_seed(0)
        m = MLICPlusPlus(N=8, M=16, slice_num=4, context_window=3)
        m.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            full = m(x)
            y, z = m.refine_extract(x)
            refine = m.refine_forward(y, z)
        # When SGA mode is off, refine_forward should match the standard
        # forward path (both use STE on the y leaves and noise on z).
        assert torch.allclose(refine["x_hat"], full["x_hat"], atol=1e-5)

    def test_refine_loop_decreases_rd_loss(self) -> None:
        torch.manual_seed(42)
        m = MLICPlusPlus(N=8, M=16, slice_num=4, context_window=3)
        m.train()
        x = torch.randn(1, 3, 64, 64)
        y_init, z_init = m.refine_extract(x)

        sga = SGAQuantizer()
        m.set_sga_mode(sga)
        y = nn.Parameter(y_init.clone())
        z = nn.Parameter(z_init.clone())
        opt = torch.optim.Adam([y, z], lr=5e-3)

        total_it = 50
        first_loss = None
        last_loss = None
        for it in range(total_it):
            sga.set_iter(it, total_it)
            opt.zero_grad()
            out = m.refine_forward(y, z)
            bpp = (
                -torch.log2(out["likelihoods"]["y"]).sum()
                - torch.log2(out["likelihoods"]["z"]).sum()
            ) / x.numel()
            mse = F.mse_loss(x, out["x_hat"])
            loss = bpp + 0.025 * mse * 255**2
            loss.backward()
            assert y.grad is not None and y.grad.norm() > 0
            opt.step()
            if it == 0:
                first_loss = loss.item()
            last_loss = loss.item()

        assert last_loss is not None and first_loss is not None
        assert last_loss < first_loss

    def test_mlicv2_sga_interface_smoke(self) -> None:
        # MLICv2 wires GSC into every leaf; on a fresh untrained model the
        # selective mask is all-False, which detaches y_hat from y. We only
        # smoke-test the interface here (set_sga_mode, refine_extract,
        # refine_forward shapes). A real refinement run requires a trained
        # checkpoint where GSC outputs a meaningful mask.
        m = MLICv2(N=8, M=16, slice_num=4, context_window=3)
        m.eval()
        x = torch.randn(1, 3, 64, 64)
        y, z = m.refine_extract(x)
        assert y.shape == (1, 16, 4, 4)
        assert z.shape == (1, 8, 1, 1)
        sga = SGAQuantizer()
        m.set_sga_mode(sga)
        sga.set_iter(0, 100)
        with torch.no_grad():
            out = m.refine_forward(y, z)
        assert out["x_hat"].shape == (1, 3, 64, 64)
        assert out["likelihoods"]["y"].shape == (1, 16, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 8, 1, 1)
