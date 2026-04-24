from __future__ import annotations

import math

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.layers import GDN, conv3x3, subpel_conv3x3
from compressai.layers.lic.swin import PatchMerging, PatchSplit
from compressai.layers.lic.stf import PatchEmbed, STFBasicLayer, STFWinNoShiftAttention
from compressai.models.utils import conv, deconv
from compressai.registry import register_model

from .stf_support import (
    SliceEntropyCompressionModel,
    infer_max_support_slices,
    infer_num_slices,
)

__all__ = ["SymmetricalTransFormer", "WACNN"]


@register_model("stf-wacnn")
class WACNN(SliceEntropyCompressionModel):
    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        num_slices: int = 10,
        max_support_slices: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            STFWinNoShiftAttention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            STFWinNoShiftAttention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s = nn.Sequential(
            STFWinNoShiftAttention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            STFWinNoShiftAttention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )
        self.h_a = nn.Sequential(
            conv3x3(M, M),
            nn.GELU(),
            conv3x3(M, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )
        self.h_mean_s = nn.Sequential(
            conv3x3(N, N),
            nn.GELU(),
            subpel_conv3x3(N, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M),
        )
        self.h_scale_s = nn.Sequential(
            conv3x3(N, N),
            nn.GELU(),
            subpel_conv3x3(N, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M),
        )
        self._init_slice_entropy(M, N, num_slices, max_support_slices)

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        y_hat, y_likelihoods, z_likelihoods = self._forward_latent(y)
        return {"x_hat": self.g_s(y_hat), "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x: Tensor) -> Dict[str, object]:
        return self._compress_latent(self.g_a(x))

    def decompress(self, strings: Sequence[Sequence[bytes]], shape: Tuple[int, int]) -> Dict[str, Tensor]:
        return {"x_hat": self.g_s(self._decompress_latent(strings, shape)).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "WACNN":
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.7.weight"].size(0)
        num_slices = infer_num_slices(state_dict) or 10
        max_support_slices = infer_max_support_slices(state_dict, M, num_slices)
        net = cls(N=N, M=M, num_slices=num_slices, max_support_slices=max_support_slices)
        net.load_state_dict(state_dict)
        return net


@register_model("stf")
class SymmetricalTransFormer(SliceEntropyCompressionModel):
    def __init__(
        self,
        pretrain_img_size: int = 256,
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 48,
        depths: Optional[Sequence[int]] = None,
        num_heads: Optional[Sequence[int]] = None,
        window_size: int = 4,
        num_slices: int = 12,
        max_support_slices: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        patch_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        depths = list(depths or [2, 2, 6, 2])
        num_heads = list(num_heads or [3, 6, 12, 24])
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for layer_index in range(self.num_layers):
            self.layers.append(
                STFBasicLayer(
                    dim=int(embed_dim * 2**layer_index),
                    depth=depths[layer_index],
                    num_heads=num_heads[layer_index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:layer_index]) : sum(depths[: layer_index + 1])],
                    norm_layer=norm_layer,
                    downsample=None if layer_index == self.num_layers - 1 else PatchMerging,
                )
            )

        reversed_depths = list(reversed(depths))
        reversed_heads = list(reversed(num_heads))
        self.syn_layers = nn.ModuleList()
        for layer_index in range(self.num_layers):
            self.syn_layers.append(
                STFBasicLayer(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - layer_index)),
                    depth=reversed_depths[layer_index],
                    num_heads=reversed_heads[layer_index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(reversed_depths[:layer_index]) : sum(reversed_depths[: layer_index + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=None if layer_index == self.num_layers - 1 else PatchSplit,
                )
            )

        self.end_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * patch_size**2, kernel_size=5, stride=1, padding=2),
            nn.PixelShuffle(patch_size),
            nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
        )

        latent_channels = int(embed_dim * 2 ** (self.num_layers - 1))
        bottleneck_channels = latent_channels // 2
        self.h_a = nn.Sequential(
            conv3x3(latent_channels, latent_channels),
            nn.GELU(),
            conv3x3(latent_channels, latent_channels - embed_dim),
            nn.GELU(),
            conv3x3(latent_channels - embed_dim, latent_channels - 2 * embed_dim, stride=2),
            nn.GELU(),
            conv3x3(latent_channels - 2 * embed_dim, latent_channels - 3 * embed_dim),
            nn.GELU(),
            conv3x3(latent_channels - 3 * embed_dim, bottleneck_channels, stride=2),
        )
        self.h_mean_s = nn.Sequential(
            conv3x3(bottleneck_channels, latent_channels - 3 * embed_dim),
            nn.GELU(),
            subpel_conv3x3(latent_channels - 3 * embed_dim, latent_channels - 2 * embed_dim, 2),
            nn.GELU(),
            conv3x3(latent_channels - 2 * embed_dim, latent_channels - embed_dim),
            nn.GELU(),
            subpel_conv3x3(latent_channels - embed_dim, latent_channels, 2),
            nn.GELU(),
            conv3x3(latent_channels, latent_channels),
        )
        self.h_scale_s = nn.Sequential(
            conv3x3(bottleneck_channels, latent_channels - 3 * embed_dim),
            nn.GELU(),
            subpel_conv3x3(latent_channels - 3 * embed_dim, latent_channels - 2 * embed_dim, 2),
            nn.GELU(),
            conv3x3(latent_channels - 2 * embed_dim, latent_channels - embed_dim),
            nn.GELU(),
            subpel_conv3x3(latent_channels - embed_dim, latent_channels, 2),
            nn.GELU(),
            conv3x3(latent_channels, latent_channels),
        )
        self._init_slice_entropy(
            latent_channels,
            bottleneck_channels,
            num_slices,
            num_slices // 2 if max_support_slices is None else max_support_slices,
        )

    def _analysis_transform(self, x: Tensor) -> Tuple[Tensor, int, int]:
        output = self.patch_embed(x)
        height, width = output.size(2), output.size(3)
        output = self.pos_drop(output.flatten(2).transpose(1, 2))
        for layer in self.layers:
            output, height, width = layer(output, height, width)
        channels = self.embed_dim * 2 ** (self.num_layers - 1)
        output = output.view(-1, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return output, height, width

    def _synthesis_transform(self, y_hat: Tensor, height: int, width: int) -> Tensor:
        channels = self.embed_dim * 2 ** (self.num_layers - 1)
        output = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, height * width, channels)
        for layer in self.syn_layers:
            output, height, width = layer(output, height, width)
        output = output.view(-1, height, width, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        return self.end_conv(output)

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y, height, width = self._analysis_transform(x)
        y_hat, y_likelihoods, z_likelihoods = self._forward_latent(y)
        return {
            "x_hat": self._synthesis_transform(y_hat, height, width),
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y, _, _ = self._analysis_transform(x)
        return self._compress_latent(y)

    def decompress(self, strings: Sequence[Sequence[bytes]], shape: Tuple[int, int]) -> Dict[str, Tensor]:
        y_hat = self._decompress_latent(strings, shape)
        height, width = y_hat.shape[2:]
        return {"x_hat": self._synthesis_transform(y_hat, height, width).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "SymmetricalTransFormer":
        patch_size = state_dict["patch_embed.proj.weight"].size(2)
        embed_dim = state_dict["patch_embed.proj.weight"].size(0)
        layer_indices = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith("layers.") and ".blocks." in key
            }
        )
        depths = [
            len(
                {
                    int(key.split(".")[3])
                    for key in state_dict
                    if key.startswith(f"layers.{layer_index}.blocks.")
                }
            )
            for layer_index in layer_indices
        ]
        num_heads = [
            state_dict[f"layers.{layer_index}.blocks.0.attn.relative_position_bias_table"].size(1)
            for layer_index in layer_indices
        ]
        table_size = state_dict["layers.0.blocks.0.attn.relative_position_bias_table"].size(0)
        window_size = (math.isqrt(table_size) + 1) // 2
        num_slices = infer_num_slices(state_dict) or 12
        latent_channels = embed_dim * 2 ** (len(depths) - 1)
        max_support_slices = infer_max_support_slices(state_dict, latent_channels, num_slices)

        net = cls(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
        )
        net.load_state_dict(state_dict)
        return net
