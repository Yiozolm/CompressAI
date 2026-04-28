from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import MambaICLatentCodec
from compressai.layers import CheckerboardMaskedConv2d
from compressai.layers.attn import (
    SWAtten,
    infer_swatten_attention_dim,
    infer_swatten_head_dim,
    infer_swatten_window_size,
)
from compressai.layers.ssm import (
    build_vss_backbone,
    build_vss_context_stage,
    infer_vss_block_kwargs,
    infer_vss_depths,
)
from compressai.models._bases import (
    infer_max_support_slices,
    infer_num_slices,
    lrp_support_channels,
    make_entropy_transform,
    slice_support_channels,
)
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["MambaIC"]


def _infer_context_depths(
    state_dict: Dict[str, Tensor],
    num_slices: int,
) -> Tuple[int, ...]:
    depths = []
    for index in range(num_slices):
        depth = 0
        while f"latent_codec.context_vss.{index}.{depth}.norm.weight" in state_dict:
            depth += 1
        depths.append(depth or 2)
    return tuple(depths)


@register_model("mambaic")
class MambaIC(CompressionModel):
    r"""MambaIC model from F. Zeng, H. Tang, Y. Shao, S. Chen, L. Shao, Y. Wang:
    `"MambaIC: State Space Models for High-Performance Learned Image
    Compression" <https://arxiv.org/abs/2503.12461>`_, IEEE/CVF Conf. on
    Computer Vision and Pattern Recognition (CVPR), 2025.

    Builds analysis/synthesis transforms with VSS (visual state-space) blocks
    and combines a window-based local attention spatial context with a
    channel-wise autoregressive entropy model.

    Args:
        N (int): Number of channels in the hyperprior backbone.
        M (int): Number of channels in the latent representation.
        num_slices (int): Number of channel slices for the entropy model.
    """

    def __init__(
        self,
        depths: Sequence[int] = (2, 2, 9, 2),
        drop_path_rate: float = 0.1,
        N: int = 128,
        M: int = 320,
        hyper_channels: int = 192,
        num_slices: int = 5,
        max_support_slices: int = 5,
        context_depths: Optional[Sequence[int]] = None,
        window_size: int = 8,
        support_head_dim: int = 16,
        context_head_dim: int = 16,
        support_attention_dim: int = 128,
        context_attention_dim: int = 128,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_conv: int = 3,
        scan_backend: str = "auto",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")

        self.depths = tuple(int(depth) for depth in depths)
        self.N = int(N)
        self.M = int(M)
        self.hyper_channels = int(hyper_channels)
        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)
        self.window_size = int(window_size)
        self.support_head_dim = int(support_head_dim)
        self.context_head_dim = int(context_head_dim)
        self.support_attention_dim = int(support_attention_dim)
        self.context_attention_dim = int(context_attention_dim)

        (
            self.g_a,
            self.g_s,
            self.h_a,
            self.h_mean_s,
            self.h_scale_s,
        ) = build_vss_backbone(
            self.depths,
            drop_path_rate,
            N,
            M,
            hyper_channels=hyper_channels,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_conv=ssm_conv,
            scan_backend=scan_backend,
        )

        if support_attention_dim % support_head_dim != 0:
            raise ValueError("support_head_dim must divide support_attention_dim")
        if context_attention_dim % context_head_dim != 0:
            raise ValueError("context_head_dim must divide context_attention_dim")

        slice_channels = M // num_slices
        context_depths = tuple(int(depth) for depth in (context_depths or (2,) * num_slices))
        if len(context_depths) != num_slices:
            raise ValueError("context_depths must have num_slices entries")

        mean_support_transforms = nn.ModuleList()
        scale_support_transforms = nn.ModuleList()
        cc_mean_transforms = nn.ModuleList()
        cc_scale_transforms = nn.ModuleList()
        lrp_transforms = nn.ModuleList()
        context_prediction = nn.ModuleList()
        context_mean_transforms = nn.ModuleList()
        context_scale_transforms = nn.ModuleList()

        for index in range(num_slices):
            base_support_channels = slice_support_channels(
                M,
                slice_channels,
                index,
                max_support_slices,
            )
            mean_support_transforms.append(
                SWAtten(
                    base_support_channels,
                    base_support_channels,
                    support_head_dim,
                    window_size,
                    0.0,
                    inter_dim=support_attention_dim,
                )
            )
            scale_support_transforms.append(
                SWAtten(
                    base_support_channels,
                    base_support_channels,
                    support_head_dim,
                    window_size,
                    0.0,
                    inter_dim=support_attention_dim,
                )
            )
            cc_mean_transforms.append(
                make_entropy_transform(base_support_channels, slice_channels)
            )
            cc_scale_transforms.append(
                make_entropy_transform(base_support_channels, slice_channels)
            )
            lrp_transforms.append(
                make_entropy_transform(
                    lrp_support_channels(M, slice_channels, index, max_support_slices),
                    slice_channels,
                )
            )
            context_prediction.append(
                CheckerboardMaskedConv2d(
                    slice_channels,
                    2 * slice_channels,
                    kernel_size=5,
                    padding=2,
                    stride=1,
                )
            )
            context_mean_transforms.append(
                SWAtten(
                    slice_channels,
                    slice_channels,
                    context_head_dim,
                    window_size,
                    0.0,
                    inter_dim=context_attention_dim,
                )
            )
            context_scale_transforms.append(
                SWAtten(
                    slice_channels,
                    slice_channels,
                    context_head_dim,
                    window_size,
                    0.0,
                    inter_dim=context_attention_dim,
                )
            )

        context_drop_paths = torch.linspace(0, drop_path_rate, sum(context_depths)).tolist()
        offset = 0
        context_vss = nn.ModuleList()
        for index, depth in enumerate(context_depths):
            stage_drop_paths = [
                float(value) for value in context_drop_paths[offset : offset + depth]
            ]
            offset += depth
            context_channels = 2 * M + (2 if index == 0 else 4) * slice_channels
            context_vss.append(
                build_vss_context_stage(
                    context_channels,
                    2 * slice_channels,
                    depth,
                    stage_drop_paths,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_conv=ssm_conv,
                    scan_backend=scan_backend,
                )
            )

        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)
        self.latent_codec = MambaICLatentCodec(
            mean_support_transforms=mean_support_transforms,
            scale_support_transforms=scale_support_transforms,
            cc_mean_transforms=cc_mean_transforms,
            cc_scale_transforms=cc_scale_transforms,
            context_prediction=context_prediction,
            context_vss=context_vss,
            context_mean_transforms=context_mean_transforms,
            context_scale_transforms=context_scale_transforms,
            lrp_transforms=lrp_transforms,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
        )

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_out = self.latent_codec(y, latent_means, latent_scales)
        return {
            "x_hat": self.g_s(y_out["y_hat"]),
            "likelihoods": {"y": y_out["likelihoods"]["y"], "z": z_likelihoods},
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_out = self.latent_codec.compress(y, latent_means, latent_scales)
        return {"strings": [y_out["strings"], z_strings], "shape": z.size()[-2:]}

    def decompress(
        self,
        strings: Sequence[object],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_shape = (z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        y_out = self.latent_codec.decompress(strings[0], y_shape, latent_means, latent_scales)
        return {"x_hat": self.g_s(y_out["y_hat"]).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "MambaIC":
        depths = infer_vss_depths(state_dict)
        vss_kwargs = infer_vss_block_kwargs(state_dict)
        N = state_dict["g_a.0.weight"].size(0) // 2
        M = state_dict["h_a.0.weight"].size(1)
        hyper_channels = state_dict["entropy_bottleneck.quantiles"].size(0)
        num_slices = infer_num_slices(state_dict) or 5
        max_support_slices = infer_max_support_slices(state_dict, M, num_slices)
        slice_channels = M // num_slices
        window_size = infer_swatten_window_size(
            state_dict,
            "latent_codec.mean_support_transforms.0.",
        )
        support_attention_dim = infer_swatten_attention_dim(
            state_dict,
            "latent_codec.mean_support_transforms.0",
        )
        support_head_dim = infer_swatten_head_dim(
            state_dict,
            "latent_codec.mean_support_transforms.0.",
            support_attention_dim,
        )
        context_attention_dim = infer_swatten_attention_dim(
            state_dict,
            "latent_codec.context_mean_transforms.0",
        )
        context_head_dim = infer_swatten_head_dim(
            state_dict,
            "latent_codec.context_mean_transforms.0.",
            context_attention_dim,
        )
        net = cls(
            depths=depths,
            N=N,
            M=M,
            hyper_channels=hyper_channels,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            context_depths=_infer_context_depths(state_dict, num_slices),
            window_size=window_size,
            support_head_dim=support_head_dim,
            context_head_dim=context_head_dim,
            support_attention_dim=support_attention_dim,
            context_attention_dim=context_attention_dim,
            **vss_kwargs,
        )
        net.load_state_dict(state_dict)
        return net
