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

from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    CheckerboardLatentCodec,
    EntropyBottleneckLatentCodec,
    GaussianConditionalLatentCodec,
    HyperpriorLatentCodec,
    MultiContextCheckerboardLatentCodec,
)
from compressai.layers import CheckerboardMaskedConv2d
from compressai.models._helpers.sga import SGARefinementMixin
from compressai.models.mlic import MLICPlusPlus, MLICv2, MLICv2Plus
from compressai.ops import SGAQuantizer


class _ZeroEntropyParameters(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(
            x.shape[0],
            self.out_channels,
            x.shape[2],
            x.shape[3],
        )


def _make_multi_context_codec(
    *,
    quantizer: str = "ste",
    sga: Optional[SGAQuantizer] = None,
) -> MultiContextCheckerboardLatentCodec:
    return MultiContextCheckerboardLatentCodec(
        entropy_parameters_anchor=_ZeroEntropyParameters(8),
        entropy_parameters_nonanchor=_ZeroEntropyParameters(8),
        scale_table=[0.11, 0.5, 1.0, 2.0, 4.0],
        quantizer=quantizer,
        sga=sga,
    )


class TestSGAQuantizer:
    def test_fallback_to_round_when_iter_unset(self) -> None:
        sga = SGAQuantizer()
        x = torch.tensor([0.3, -0.7, 1.4, -2.1])
        assert torch.equal(sga(x), torch.round(x))

    def test_reset_iter_falls_back_to_round(self) -> None:
        sga = SGAQuantizer()
        x = torch.tensor([0.3, -0.7, 1.4, -2.1])
        sga.set_iter(800, 2000)
        sga.set_iter(None, None)
        assert torch.equal(sga(x), torch.round(x))

    def test_temperature_schedule_matches_official_defaults(self) -> None:
        sga = SGAQuantizer()
        assert sga.annealed_temperature(0) == sga.upper_temperature
        assert sga.annealed_temperature(700) == sga.upper_temperature
        assert sga.annealed_temperature(800) < sga.upper_temperature

    def test_relaxed_sample_is_differentiable(self) -> None:
        torch.manual_seed(0)
        sga = SGAQuantizer()
        sga.set_iter(800)
        x = torch.randn(8, requires_grad=True)
        y = sga(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.norm() > 0


class TestEntropyBottleneckLatentCodecSGA:
    def test_sga_requires_module(self) -> None:
        with pytest.raises(ValueError):
            EntropyBottleneckLatentCodec(
                entropy_bottleneck=EntropyBottleneck(4),
                quantizer="sga",
            )

    def test_sga_round_fallback_matches_hard_round(self) -> None:
        codec = EntropyBottleneckLatentCodec(
            entropy_bottleneck=EntropyBottleneck(4),
            quantizer="sga",
            sga=SGAQuantizer(),
        ).eval()
        y = torch.randn(1, 4, 8, 8)
        out = codec(y)
        medians = codec.entropy_bottleneck._get_medians()
        assert torch.allclose(out["y_hat"], torch.round(y - medians) + medians)
        assert out["likelihoods"]["y"].shape == y.shape


class TestGaussianConditionalLatentCodecSGA:
    def test_sga_requires_module(self) -> None:
        with pytest.raises(ValueError):
            GaussianConditionalLatentCodec(quantizer="sga")

    def test_sga_round_fallback_matches_hard_round(self) -> None:
        codec = GaussianConditionalLatentCodec(quantizer="sga", sga=SGAQuantizer())
        y = torch.randn(1, 4, 8, 8)
        scales = torch.rand_like(y) + 0.5
        means = torch.randn_like(y)
        out = codec(y, torch.cat([scales, means], dim=1))
        assert torch.allclose(out["y_hat"], torch.round(y - means) + means)
        assert out["likelihoods"]["y"].shape == y.shape


class TestMultiContextCheckerboardLatentCodecSGA:
    def test_sga_requires_module(self) -> None:
        with pytest.raises(ValueError):
            _make_multi_context_codec(quantizer="sga")

    def test_sga_round_fallback_matches_ste(self) -> None:
        torch.manual_seed(1)
        ste_codec = _make_multi_context_codec().eval()
        sga_codec = _make_multi_context_codec(
            quantizer="sga",
            sga=SGAQuantizer(),
        ).eval()
        sga_codec.load_state_dict(ste_codec.state_dict())

        y = torch.randn(1, 4, 8, 8)
        side_params = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            ste_out = ste_codec(y, side_params)
            sga_out = sga_codec(y, side_params)
        assert torch.allclose(sga_out["y_hat"], ste_out["y_hat"])
        assert torch.allclose(sga_out["likelihoods"]["y"], ste_out["likelihoods"]["y"])

    def test_sga_mode_backpropagates_through_y_hat(self) -> None:
        torch.manual_seed(2)
        sga = SGAQuantizer()
        sga.set_iter(800)
        codec = _make_multi_context_codec(quantizer="sga", sga=sga).train()
        y = nn.Parameter(torch.randn(1, 4, 8, 8))
        out = codec(y, torch.randn(1, 8, 8, 8))
        out["y_hat"].pow(2).mean().backward()
        assert y.grad is not None
        assert y.grad.norm() > 0


class TestCheckerboardLatentCodecSGA:
    @staticmethod
    def _make(
        *,
        quantizer: str = "ste",
        sga: Optional[SGAQuantizer] = None,
    ) -> CheckerboardLatentCodec:
        return CheckerboardLatentCodec(
            latent_codec={
                "y": GaussianConditionalLatentCodec(
                    scale_table=[0.11, 0.5, 1.0, 2.0, 4.0],
                )
            },
            entropy_parameters=_ZeroEntropyParameters(8),
            context_prediction=CheckerboardMaskedConv2d(
                4,
                4,
                kernel_size=5,
                padding=2,
            ),
            quantizer=quantizer,
            sga=sga,
        )

    def test_sga_round_fallback_matches_ste(self) -> None:
        torch.manual_seed(3)
        ste_codec = self._make().eval()
        sga_codec = self._make(quantizer="sga", sga=SGAQuantizer()).eval()
        sga_codec.load_state_dict(ste_codec.state_dict())
        y = torch.randn(1, 4, 8, 8)
        side_params = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            ste_out = ste_codec(y, side_params)
            sga_out = sga_codec(y, side_params)
        assert torch.allclose(sga_out["y_hat"], ste_out["y_hat"])
        assert torch.allclose(sga_out["likelihoods"]["y"], ste_out["likelihoods"]["y"])


class _TinyHyperpriorModel(SGARefinementMixin, nn.Module):
    def __init__(self, z_quantizer: str = "noise") -> None:
        super().__init__()
        self.g_a = nn.Conv2d(3, 4, 1)
        self.g_s = nn.Conv2d(4, 3, 1)
        self.latent_codec = HyperpriorLatentCodec(
            h_a=nn.Conv2d(4, 2, 1),
            h_s=nn.Conv2d(2, 8, 1),
            latent_codec={
                "z": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(2),
                    quantizer=z_quantizer,
                ),
                "y": GaussianConditionalLatentCodec(quantizer="ste"),
            },
        )


class TestSGARefinementMixin:
    def test_restores_model_specific_default_quantizers(self) -> None:
        model = _TinyHyperpriorModel(z_quantizer="noise")
        sga = SGAQuantizer()

        model.set_sga_mode(sga)
        assert model.latent_codec.z.quantizer == "sga"
        assert model.latent_codec.y.quantizer == "sga"

        model.set_sga_mode(None)
        assert model.latent_codec.z.quantizer == "noise"
        assert model.latent_codec.z.sga is None
        assert model.latent_codec.y.quantizer == "ste"
        assert model.latent_codec.y.sga is None

    def test_refine_forward_smoke_for_generic_hyperprior_model(self) -> None:
        model = _TinyHyperpriorModel(z_quantizer="ste").eval()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            y, z = model.refine_extract(x)
            out = model.refine_forward(y, z)
        assert y.shape == (1, 4, 8, 8)
        assert z.shape == (1, 2, 8, 8)
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == y.shape
        assert out["likelihoods"]["z"].shape == z.shape


class TestMlicSGARefinement:
    def test_sga_interface_is_mlicv2plus_only(self) -> None:
        assert not hasattr(
            MLICPlusPlus(N=8, M=16, slice_num=4, context_window=3),
            "set_sga_mode",
        )
        assert not hasattr(
            MLICv2(N=8, M=16, slice_num=4, context_window=3), "set_sga_mode"
        )
        assert hasattr(
            MLICv2Plus(N=8, M=16, slice_num=4, context_window=3),
            "set_sga_mode",
        )

    def test_set_sga_mode_propagates_and_restores_defaults(self) -> None:
        model = MLICv2Plus(N=8, M=16, slice_num=4, context_window=3)
        sga = SGAQuantizer()
        model.set_sga_mode(sga)

        z_codec = model.latent_codec.latent_codec["z"]
        assert z_codec.quantizer == "sga"
        assert z_codec.sga is sga
        leaf_sgas = {
            id(module.sga)
            for module in model.modules()
            if isinstance(module, MultiContextCheckerboardLatentCodec)
        }
        assert leaf_sgas == {id(sga)}

        model.set_sga_mode(None)
        assert z_codec.quantizer == "ste"
        assert z_codec.sga is None
        for module in model.modules():
            if isinstance(module, MultiContextCheckerboardLatentCodec):
                assert module.quantizer == "ste"
                assert module.sga is None

    def test_refine_forward_without_sga_matches_forward(self) -> None:
        torch.manual_seed(3)
        model = MLICv2Plus(N=8, M=16, slice_num=4, context_window=3).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            full = model(x)
            y, z = model.refine_extract(x)
            refined = model.refine_forward(y, z)
        assert y.shape == (1, 16, 4, 4)
        assert z.shape == (1, 8, 1, 1)
        assert torch.allclose(refined["x_hat"], full["x_hat"])
        assert torch.allclose(refined["likelihoods"]["y"], full["likelihoods"]["y"])
        assert torch.allclose(refined["likelihoods"]["z"], full["likelihoods"]["z"])

    def test_mlicv2plus_refine_sga_interface_smoke(self) -> None:
        pytest.importorskip("timm")
        model = MLICv2Plus(N=8, M=16, slice_num=4, context_window=3).eval()
        x = torch.randn(1, 3, 64, 64)
        y, z = model.refine_extract(x)
        sga = SGAQuantizer()
        sga.set_iter(800)
        model.set_sga_mode(sga)
        with torch.no_grad():
            out = model.refine_forward(y, z)
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == y.shape
        assert out["likelihoods"]["z"].shape == z.shape
