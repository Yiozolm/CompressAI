from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple, TypeVar

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.latent_codecs import WeChARMLatentCodec
from compressai.layers.attn import SWAtten
from compressai.layers.wave import is_pytorch_wavelets_available
from compressai.registry import register_model

from .base import CompressionModel
from .weconvene_support import (
    WeConveneAnalysisTransform,
    WeConveneHyperAnalysisTransform,
    WeConveneHyperSynthesisTransform,
    WeConveneSynthesisTransform,
    infer_max_support_slices,
    infer_num_slices,
    infer_support_attention,
    lrp_channels,
    make_entropy_transform,
    support_channels,
)

__all__ = ["WeConvene"]

_ModelType = TypeVar("_ModelType", bound=type[nn.Module])


def _identity_decorator(cls: _ModelType) -> _ModelType:
    return cls


def _maybe_register_model(name: str) -> Callable[[_ModelType], _ModelType]:
    if is_pytorch_wavelets_available():
        return register_model(name)
    return _identity_decorator


def _require_wavelets() -> None:
    if is_pytorch_wavelets_available():
        return
    raise ModuleNotFoundError(
        "WeConvene requires the optional dependency `pytorch_wavelets`. "
        "Install `compressai[lic]` to enable this model."
    )


@_maybe_register_model("weconvene")
class WeConvene(CompressionModel):
    def __init__(
        self,
        N: int = 128,
        M: int = 320,
        hyper_channels: int = 192,
        num_slices: int = 5,
        max_support_slices: int = 5,
        residual_blocks: int = 3,
        wavelet: str = "haar",
        support_window_size: int = 8,
        support_head_dim: int = 16,
        support_attention_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        _require_wavelets()
        super().__init__(**kwargs)
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")
        if support_attention_dim % support_head_dim != 0:
            raise ValueError("support_attention_dim must be divisible by support_head_dim")

        self.N = int(N)
        self.M = int(M)
        self.hyper_channels = int(hyper_channels)
        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)
        self.residual_blocks = int(residual_blocks)
        self.wavelet = wavelet
        self.support_window_size = int(support_window_size)
        self.support_head_dim = int(support_head_dim)
        self.support_attention_dim = int(support_attention_dim)

        self.g_a = WeConveneAnalysisTransform(
            N=self.N,
            M=self.M,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )
        self.g_s = WeConveneSynthesisTransform(
            N=self.N,
            M=self.M,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )
        self.h_a = WeConveneHyperAnalysisTransform(
            N=self.N,
            M=self.M,
            hyper_channels=self.hyper_channels,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )
        self.h_mean_s = WeConveneHyperSynthesisTransform(
            N=self.N,
            M=self.M,
            hyper_channels=self.hyper_channels,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )
        self.h_scale_s = WeConveneHyperSynthesisTransform(
            N=self.N,
            M=self.M,
            hyper_channels=self.hyper_channels,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )

        low_slice_channels = self.M // self.num_slices
        high_slice_channels = 3 * low_slice_channels

        mean_support_transforms_low = nn.ModuleList(
            SWAtten(
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                self.support_head_dim,
                self.support_window_size,
                0.0,
                inter_dim=self.support_attention_dim,
            )
            for index in range(self.num_slices)
        )
        scale_support_transforms_low = nn.ModuleList(
            SWAtten(
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                self.support_head_dim,
                self.support_window_size,
                0.0,
                inter_dim=self.support_attention_dim,
            )
            for index in range(self.num_slices)
        )
        mean_support_transforms_high = nn.ModuleList(
            SWAtten(
                support_channels(2 * self.M, high_slice_channels, index, self.max_support_slices),
                support_channels(self.M, high_slice_channels, index, self.max_support_slices),
                self.support_head_dim,
                self.support_window_size,
                0.0,
                inter_dim=self.support_attention_dim,
            )
            for index in range(self.num_slices)
        )
        scale_support_transforms_high = nn.ModuleList(
            SWAtten(
                support_channels(2 * self.M, high_slice_channels, index, self.max_support_slices),
                support_channels(self.M, high_slice_channels, index, self.max_support_slices),
                self.support_head_dim,
                self.support_window_size,
                0.0,
                inter_dim=self.support_attention_dim,
            )
            for index in range(self.num_slices)
        )
        cc_mean_transforms_low = nn.ModuleList(
            make_entropy_transform(
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                low_slice_channels,
            )
            for index in range(self.num_slices)
        )
        cc_scale_transforms_low = nn.ModuleList(
            make_entropy_transform(
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                low_slice_channels,
            )
            for index in range(self.num_slices)
        )
        cc_mean_transforms_high = nn.ModuleList(
            make_entropy_transform(
                support_channels(self.M, high_slice_channels, index, self.max_support_slices),
                high_slice_channels,
            )
            for index in range(self.num_slices)
        )
        cc_scale_transforms_high = nn.ModuleList(
            make_entropy_transform(
                support_channels(self.M, high_slice_channels, index, self.max_support_slices),
                high_slice_channels,
            )
            for index in range(self.num_slices)
        )
        lrp_transforms_low = nn.ModuleList(
            make_entropy_transform(
                lrp_channels(self.M, low_slice_channels, index, self.max_support_slices),
                low_slice_channels,
            )
            for index in range(self.num_slices)
        )
        lrp_transforms_high = nn.ModuleList(
            make_entropy_transform(
                lrp_channels(2 * self.M, high_slice_channels, index, self.max_support_slices),
                high_slice_channels,
            )
            for index in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(self.hyper_channels)
        self.latent_codec = WeChARMLatentCodec(
            M=self.M,
            cc_mean_transforms_low=cc_mean_transforms_low,
            cc_scale_transforms_low=cc_scale_transforms_low,
            cc_mean_transforms_high=cc_mean_transforms_high,
            cc_scale_transforms_high=cc_scale_transforms_high,
            lrp_transforms_low=lrp_transforms_low,
            lrp_transforms_high=lrp_transforms_high,
            gaussian_conditional_low=GaussianConditional(None),
            gaussian_conditional_high=GaussianConditional(None),
            mean_support_transforms_low=mean_support_transforms_low,
            scale_support_transforms_low=scale_support_transforms_low,
            mean_support_transforms_high=mean_support_transforms_high,
            scale_support_transforms_high=scale_support_transforms_high,
            num_slices=self.num_slices,
            max_support_slices=self.max_support_slices,
            wavelet=self.wavelet,
        )

    @property
    def gaussian_conditional_low(self) -> GaussianConditional:
        return self.latent_codec.gaussian_conditional_low

    @property
    def gaussian_conditional_high(self) -> GaussianConditional:
        return self.latent_codec.gaussian_conditional_high

    @property
    def atten_mean_low(self) -> nn.ModuleList:
        return self.latent_codec.mean_support_transforms_low

    @property
    def atten_scale_low(self) -> nn.ModuleList:
        return self.latent_codec.scale_support_transforms_low

    @property
    def atten_mean_high(self) -> nn.ModuleList:
        return self.latent_codec.mean_support_transforms_high

    @property
    def atten_scale_high(self) -> nn.ModuleList:
        return self.latent_codec.scale_support_transforms_high

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        y_wavelet = self.latent_codec.to_wavelet(y)
        z = self.h_a(y_wavelet)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_out = self.latent_codec(
            y,
            latent_means,
            latent_scales,
            wavelet_output=y_wavelet,
        )
        return {
            "x_hat": self.g_s(y_out["y_hat"]),
            "likelihoods": {
                "y_low": y_out["likelihoods"]["y_low"],
                "y_high": y_out["likelihoods"]["y_high"],
                "z": z_likelihoods,
            },
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        y_wavelet = self.latent_codec.to_wavelet(y)
        z = self.h_a(y_wavelet)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        y_out = self.latent_codec.compress(
            y,
            self.h_mean_s(z_hat),
            self.h_scale_s(z_hat),
            wavelet_output=y_wavelet,
        )
        return {
            "strings": [*y_out["strings"], z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        if len(strings) != 3:
            raise ValueError("strings must contain [low_strings, high_strings, z_strings]")

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        y_shape = (z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        y_out = self.latent_codec.decompress(
            strings[:2],
            y_shape,
            self.h_mean_s(z_hat),
            self.h_scale_s(z_hat),
        )
        return {"x_hat": self.g_s(y_out["y_hat"]).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "WeConvene":
        N = state_dict["g_a.input_block.conv1.weight"].size(0)
        num_slices = infer_num_slices(state_dict)
        M = state_dict["latent_codec.cc_mean_transforms_low.0.4.weight"].size(0) * num_slices
        hyper_channels = state_dict["entropy_bottleneck.quantiles"].size(0)
        max_support_slices = infer_max_support_slices(state_dict, M, num_slices)
        support_window_size, support_head_dim, support_attention_dim = infer_support_attention(
            state_dict
        )

        net = cls(
            N=N,
            M=M,
            hyper_channels=hyper_channels,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            support_window_size=support_window_size,
            support_head_dim=support_head_dim,
            support_attention_dim=support_attention_dim,
        )
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            key for key in net.state_dict() if key.endswith("relative_position_index")
        }
        missing_keys = set(incompatible_keys.missing_keys) - allowed_missing
        if missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                "Unexpected incompatibility while loading WeConvene state_dict: "
                f"missing={sorted(missing_keys)}, "
                f"unexpected={sorted(incompatible_keys.unexpected_keys)}"
            )
        return net
