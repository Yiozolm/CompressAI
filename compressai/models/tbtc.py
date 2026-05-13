from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, conv, deconv
from compressai.layers.lic.tbtc import (
    TBTCAnalysisTransform,
    TBTCHyperAnalysisTransform,
    TBTCHyperSynthesisTransform,
    TBTCSynthesisTransform,
)
from compressai.ops import quantize_ste
from compressai.registry import register_model

from .base import CompressionModel

__all__ = [
    "TBTCChARMBlockHalf",
    "TBTCConvChARM",
    "TBTCConvHyperprior",
    "TBTCSwinTChARM",
    "TBTCSwinTHyperprior",
]


def _ste_quantize_with_mean(y: Tensor, mean: Tensor) -> Tensor:
    return quantize_ste(y - mean) + mean


def _default_swint_config() -> Dict[str, Dict[str, Any]]:
    return {
        "g_a": {
            "input_dim": 3,
            "embed_dim": [128, 192, 256, 320],
            "embed_out_dim": [192, 256, 320, None],
            "depths": [2, 2, 6, 2],
            "head_dim": [32, 32, 32, 32],
            "window_size": [8, 8, 8, 8],
        },
        "g_s": {
            "embed_dim": [320, 256, 192, 128],
            "embed_out_dim": [256, 192, 128, 3],
            "depths": [2, 6, 2, 2],
            "head_dim": [32, 32, 32, 32],
            "window_size": [8, 8, 8, 8],
        },
        "h_a": {
            "input_dim": 320,
            "embed_dim": [192, 192],
            "embed_out_dim": [192, None],
            "depths": [5, 1],
            "head_dim": [32, 32],
            "window_size": [4, 4],
        },
        "h_s": {
            "embed_dim": [192, 192],
            "embed_out_dim": [192, 640],
            "depths": [1, 5],
            "head_dim": [32, 32],
            "window_size": [4, 4],
        },
    }


def _validate_quality(quality: str) -> str:
    quality = quality.upper()
    if quality not in ("S", "M", "L"):
        raise ValueError('quality must be one of "S", "M", "L"')
    return quality


class TBTCChARMBlockHalf(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        c1 = (out_dim - in_dim) // 3 + in_dim
        c2 = 2 * (out_dim - in_dim) // 3 + in_dim
        self.layers = nn.Sequential(
            conv(in_dim, c1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(c1, c2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(c2, out_dim, kernel_size=3, stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


@register_model("zyc2022-conv-hyperprior")
class TBTCConvHyperprior(CompressionModel):
    """Conv-Hyperprior from Transformer-Based Transform Coding (ICLR 2022)."""

    def __init__(self, main_dim: int = 320, hyper_dim: int = 192, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.main_dim = int(main_dim)
        self.hyper_dim = int(hyper_dim)
        self.M = self.main_dim
        self.N = self.hyper_dim
        self.entropy_bottleneck = EntropyBottleneck(self.hyper_dim)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            conv(3, self.main_dim),
            GDN(self.main_dim),
            conv(self.main_dim, self.main_dim),
            GDN(self.main_dim),
            conv(self.main_dim, self.main_dim),
            GDN(self.main_dim),
            conv(self.main_dim, self.main_dim),
        )
        self.g_s = nn.Sequential(
            deconv(self.main_dim, self.main_dim),
            GDN(self.main_dim, inverse=True),
            deconv(self.main_dim, self.main_dim),
            GDN(self.main_dim, inverse=True),
            deconv(self.main_dim, self.main_dim),
            GDN(self.main_dim, inverse=True),
            deconv(self.main_dim, 3),
        )
        self.h_a = nn.Sequential(
            conv(self.main_dim, self.hyper_dim, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.hyper_dim, self.hyper_dim),
            nn.ReLU(inplace=True),
            conv(self.hyper_dim, self.hyper_dim),
        )
        self.h_s = nn.Sequential(
            deconv(self.hyper_dim, self.hyper_dim),
            nn.ReLU(inplace=True),
            deconv(self.hyper_dim, self.hyper_dim),
            nn.ReLU(inplace=True),
            conv(self.hyper_dim, self.main_dim * 2, stride=1, kernel_size=3),
        )

    @property
    def downsampling_factor(self) -> int:
        return 64

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "TBTCConvHyperprior":
        main_dim = int(state_dict["g_a.0.weight"].size(0))
        hyper_dim = int(state_dict["h_a.0.weight"].size(0))
        net = cls(main_dim=main_dim, hyper_dim=hyper_dim)
        net.load_state_dict(state_dict)
        return net

    def _hyper_params(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        return z_likelihoods, scales_hat, means_hat

    def forward(self, x: Tensor) -> Dict[str, Any]:
        y = self.g_a(x)
        z_likelihoods, scales_hat, means_hat = self._hyper_params(y)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(_ste_quantize_with_mean(y, means_hat))
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x: Tensor) -> Dict[str, Any]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat, means_hat = self.h_s(z_hat).chunk(2, dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings: Sequence[Sequence[bytes]], shape: Tuple[int, int]) -> Dict[str, Tensor]:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat, means_hat = self.h_s(z_hat).chunk(2, dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        return {"x_hat": self.g_s(y_hat).clamp_(0, 1)}


@register_model("zyc2022-conv-charm")
class TBTCConvChARM(TBTCConvHyperprior):
    """Conv-ChARM from Transformer-Based Transform Coding (ICLR 2022)."""

    def __init__(
        self,
        main_dim: int = 320,
        hyper_dim: int = 192,
        num_slices: int = 10,
        slice_channels: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(main_dim=main_dim, hyper_dim=hyper_dim, **kwargs)
        self.num_slices = int(num_slices)
        if slice_channels is None and main_dim % self.num_slices != 0 and main_dim % 32 == 0:
            self.num_slices = main_dim // 32
            slice_channels = 32
        self.slice_channels = int(slice_channels or main_dim // self.num_slices)
        if self.slice_channels * self.num_slices != self.main_dim:
            raise ValueError("main_dim must equal num_slices * slice_channels")

        self.charm_mean_transforms = nn.ModuleList(
            TBTCChARMBlockHalf(self.slice_channels * (index + 1), self.slice_channels)
            for index in range(self.num_slices)
        )
        self.charm_scale_transforms = nn.ModuleList(
            TBTCChARMBlockHalf(self.slice_channels * (index + 1), self.slice_channels)
            for index in range(self.num_slices)
        )

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "TBTCConvChARM":
        main_dim = int(state_dict["g_a.0.weight"].size(0))
        hyper_dim = int(state_dict["h_a.0.weight"].size(0))
        indices = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("charm_mean_transforms.") and key.endswith("layers.4.weight")
        }
        num_slices = len(indices) or 10
        key = "charm_mean_transforms.0.layers.4.weight"
        slice_channels = int(state_dict[key].size(0)) if key in state_dict else main_dim // num_slices
        net = cls(main_dim=main_dim, hyper_dim=hyper_dim, num_slices=num_slices, slice_channels=slice_channels)
        net.load_state_dict(state_dict)
        return net

    def _charm_params(
        self,
        slice_index: int,
        means_hat_slices: Sequence[Tensor],
        scales_hat_slices: Sequence[Tensor],
        y_hat_slices: Sequence[Tensor],
        spatial_shape: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        mean_support = torch.cat([means_hat_slices[slice_index], *y_hat_slices], dim=1)
        scale_support = torch.cat([scales_hat_slices[slice_index], *y_hat_slices], dim=1)
        mu = self.charm_mean_transforms[slice_index](mean_support)
        scale = self.charm_scale_transforms[slice_index](scale_support)
        return mu[:, :, : spatial_shape[0], : spatial_shape[1]], scale[:, :, : spatial_shape[0], : spatial_shape[1]]

    def forward(self, x: Tensor) -> Dict[str, Any]:
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset
        scales_hat, means_hat = self.h_s(z_hat).chunk(2, dim=1)
        means_hat_slices = means_hat.chunk(self.num_slices, dim=1)
        scales_hat_slices = scales_hat.chunk(self.num_slices, dim=1)
        y_hat_slices: List[Tensor] = []
        y_likelihood_slices: List[Tensor] = []
        spatial_shape = (y.shape[2], y.shape[3])

        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, dim=1)):
            mu, scale = self._charm_params(
                slice_index, means_hat_slices, scales_hat_slices, y_hat_slices, spatial_shape
            )
            _, likelihood = self.gaussian_conditional(y_slice, scale, means=mu)
            y_hat_slices.append(_ste_quantize_with_mean(y_slice, mu))
            y_likelihood_slices.append(likelihood)

        y_hat = torch.cat(y_hat_slices, dim=1)
        likelihoods = {"y": torch.cat(y_likelihood_slices, dim=1), "z": z_likelihoods}
        return {"x_hat": self.g_s(y_hat), "likelihoods": likelihoods}

    def compress(self, x: Tensor) -> Dict[str, Any]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat, means_hat = self.h_s(z_hat).chunk(2, dim=1)
        means_hat_slices = means_hat.chunk(self.num_slices, dim=1)
        scales_hat_slices = scales_hat.chunk(self.num_slices, dim=1)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        symbols: List[int] = []
        indexes: List[int] = []
        y_hat_slices: List[Tensor] = []
        spatial_shape = (y.shape[2], y.shape[3])

        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, dim=1)):
            mu, scale = self._charm_params(
                slice_index, means_hat_slices, scales_hat_slices, y_hat_slices, spatial_shape
            )
            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            symbols.extend(y_q_slice.reshape(-1).tolist())
            indexes.extend(index.reshape(-1).tolist())
            y_hat_slices.append(y_q_slice + mu)

        encoder = BufferedRansEncoder()
        encoder.encode_with_indexes(symbols, indexes, cdf, cdf_lengths, offsets)
        return {"strings": [[encoder.flush()], z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings: Sequence[Sequence[bytes]], shape: Tuple[int, int]) -> Dict[str, Tensor]:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat, means_hat = self.h_s(z_hat).chunk(2, dim=1)
        means_hat_slices = means_hat.chunk(self.num_slices, dim=1)
        scales_hat_slices = scales_hat.chunk(self.num_slices, dim=1)
        y_shape = (z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])
        y_hat_slices: List[Tensor] = []

        for slice_index in range(self.num_slices):
            mu, scale = self._charm_params(
                slice_index, means_hat_slices, scales_hat_slices, y_hat_slices, y_shape
            )
            index = self.gaussian_conditional.build_indexes(scale)
            values = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            y_q_slice = torch.tensor(values, device=mu.device, dtype=mu.dtype).reshape(mu.shape)
            y_hat_slices.append(self.gaussian_conditional.dequantize(y_q_slice, mu))

        return {"x_hat": self.g_s(torch.cat(y_hat_slices, dim=1)).clamp_(0, 1)}


@register_model("zyc2022-swint-hyperprior")
class TBTCSwinTHyperprior(TBTCConvHyperprior):
    """SwinT-Hyperprior from Transformer-Based Transform Coding."""

    def __init__(self, **kwargs: Any) -> None:
        cfg = deepcopy(_default_swint_config())
        for key in ("g_a", "g_s", "h_a", "h_s"):
            if key in kwargs:
                cfg[key] = kwargs.pop(key)
        super().__init__(main_dim=cfg["g_a"]["embed_dim"][-1], hyper_dim=cfg["h_a"]["embed_dim"][-1], **kwargs)
        self.g_a = TBTCAnalysisTransform(**cfg["g_a"])
        self.g_s = TBTCSynthesisTransform(**cfg["g_s"])
        self.h_a = TBTCHyperAnalysisTransform(**cfg["h_a"])
        self.h_s = TBTCHyperSynthesisTransform(**cfg["h_s"])

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "TBTCSwinTHyperprior":
        net = cls()
        net.load_state_dict(state_dict)
        return net


@register_model("zyc2022-swint-charm")
class TBTCSwinTChARM(TBTCConvChARM):
    """SwinT-ChARM from Transformer-Based Transform Coding."""

    def __init__(self, num_slices: int = 10, slice_channels: Optional[int] = None, **kwargs: Any) -> None:
        cfg = deepcopy(_default_swint_config())
        for key in ("g_a", "g_s", "h_a", "h_s"):
            if key in kwargs:
                cfg[key] = kwargs.pop(key)
        super().__init__(
            main_dim=cfg["g_a"]["embed_dim"][-1],
            hyper_dim=cfg["h_a"]["embed_dim"][-1],
            num_slices=num_slices,
            slice_channels=slice_channels,
            **kwargs,
        )
        self.g_a = TBTCAnalysisTransform(**cfg["g_a"])
        self.g_s = TBTCSynthesisTransform(**cfg["g_s"])
        self.h_a = TBTCHyperAnalysisTransform(**cfg["h_a"])
        self.h_s = TBTCHyperSynthesisTransform(**cfg["h_s"])

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "TBTCSwinTChARM":
        net = cls()
        net.load_state_dict(state_dict)
        return net
