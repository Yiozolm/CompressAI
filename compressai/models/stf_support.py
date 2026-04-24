from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import ChannelSliceLatentCodec
from compressai.models.utils import conv

from .base import CompressionModel

__all__ = [
    "SliceEntropyCompressionModel",
    "infer_max_support_slices",
    "infer_num_slices",
]


_CC_TRANSFORM_KEY_PREFIX = "latent_codec.cc_mean_transforms."
_CC_TRANSFORM_KEY_SUFFIX = ".0.weight"


def infer_num_slices(state_dict: Dict[str, Tensor]) -> int:
    slice_indices = {
        int(key[len(_CC_TRANSFORM_KEY_PREFIX) :].split(".", 1)[0])
        for key in state_dict
        if key.startswith(_CC_TRANSFORM_KEY_PREFIX)
        and key.endswith(_CC_TRANSFORM_KEY_SUFFIX)
    }
    return len(slice_indices)


def infer_max_support_slices(
    state_dict: Dict[str, Tensor],
    latent_channels: int,
    num_slices: int,
) -> int:
    slice_channels = latent_channels // num_slices
    max_input_channels = max(
        tensor.size(1)
        for key, tensor in state_dict.items()
        if key.startswith(_CC_TRANSFORM_KEY_PREFIX)
        and key.endswith(_CC_TRANSFORM_KEY_SUFFIX)
    )
    return max(0, (max_input_channels - latent_channels) // slice_channels)


def _make_cc_transform(
    in_channels: int, out_channels: int
) -> nn.Sequential:
    return nn.Sequential(
        conv(in_channels, 224, stride=1, kernel_size=3),
        nn.GELU(),
        conv(224, 176, stride=1, kernel_size=3),
        nn.GELU(),
        conv(176, 128, stride=1, kernel_size=3),
        nn.GELU(),
        conv(128, 64, stride=1, kernel_size=3),
        nn.GELU(),
        conv(64, out_channels, stride=1, kernel_size=3),
    )


class SliceEntropyCompressionModel(CompressionModel):
    """Channel-conditional entropy backbone shared by WACNN and SymmetricalTransFormer.

    Subclasses must populate ``g_a``, ``g_s``, ``h_a``, ``h_mean_s`` and
    ``h_scale_s``, then call :meth:`_init_slice_entropy` to wire up the
    entropy bottleneck for ``z`` and the :class:`ChannelSliceLatentCodec`
    for ``y``.
    """

    h_a: nn.Module
    h_mean_s: nn.Module
    h_scale_s: nn.Module
    entropy_bottleneck: EntropyBottleneck
    latent_codec: ChannelSliceLatentCodec

    def _init_slice_entropy(
        self,
        latent_channels: int,
        entropy_bottleneck_channels: int,
        num_slices: int,
        max_support_slices: int,
    ) -> None:
        if latent_channels % num_slices != 0:
            raise ValueError("latent_channels must be divisible by num_slices")

        slice_channels = latent_channels // num_slices
        cc_mean_transforms = nn.ModuleList(
            _make_cc_transform(
                latent_channels + slice_channels * min(index, max_support_slices),
                slice_channels,
            )
            for index in range(num_slices)
        )
        cc_scale_transforms = nn.ModuleList(
            _make_cc_transform(
                latent_channels + slice_channels * min(index, max_support_slices),
                slice_channels,
            )
            for index in range(num_slices)
        )
        lrp_transforms = nn.ModuleList(
            _make_cc_transform(
                latent_channels
                + slice_channels * min(index + 1, max_support_slices + 1),
                slice_channels,
            )
            for index in range(num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        self.latent_codec = ChannelSliceLatentCodec(
            cc_mean_transforms=cc_mean_transforms,
            cc_scale_transforms=cc_scale_transforms,
            lrp_transforms=lrp_transforms,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            quantizer="ste",
        )

    @property
    def num_slices(self) -> int:
        return self.latent_codec.num_slices

    @property
    def max_support_slices(self) -> int:
        return self.latent_codec.max_support_slices

    def _hyper_priors(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        return z, z_likelihoods, latent_means, latent_scales

    def _forward_latent(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        _, z_likelihoods, latent_means, latent_scales = self._hyper_priors(y)
        y_out = self.latent_codec(y, latent_means, latent_scales)
        return y_out["y_hat"], y_out["likelihoods"]["y"], z_likelihoods

    def _compress_latent(self, y: Tensor) -> Dict[str, object]:
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_out = self.latent_codec.compress(y, latent_means, latent_scales)
        return {
            "strings": [[y_out["strings"][0]], z_strings],
            "shape": z.size()[-2:],
        }

    def _decompress_latent(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Tensor:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_shape = (z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        y_out = self.latent_codec.decompress(
            strings[0], y_shape, latent_means, latent_scales
        )
        return y_out["y_hat"]
