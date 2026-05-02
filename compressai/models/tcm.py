# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# This file adapts code from https://github.com/jmliu206/LIC_TCM
# (originally distributed under the MIT License). The upstream copyright
# notice is preserved in that repository; modifications by InterDigital
# Communications, Inc. are released under the BSD 3-Clause Clear License
# terms below.

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

import re

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import (
    CausalContextAdjustmentEntropyModel,
    EntropyBottleneck,
    GaussianConditional,
)
from compressai.entropy_models.cca import (
    has_cca_aux_state,
    infer_cca_hidden_channels,
    infer_cca_num_layers,
)
from compressai.latent_codecs import ChannelSliceLatentCodec
from compressai.layers import (
    OLP,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    WLS,
    conv3x3,
    iWLS,
    subpel_conv3x3,
)
from compressai.layers.attn import ConvTransBlock, SWAtten
from compressai.registry import register_model

from .base import CompressionModel
from ._bases import infer_max_support_slices, infer_num_slices
from .utils import conv

__all__ = ["TCM"]

_LEGACY_LATENT_PREFIX_MAP = {
    "gaussian_conditional.": "latent_codec.gaussian_conditional.",
    "atten_mean.": "latent_codec.mean_support_transforms.",
    "atten_scale.": "latent_codec.scale_support_transforms.",
    "mean_support_transforms.": "latent_codec.mean_support_transforms.",
    "scale_support_transforms.": "latent_codec.scale_support_transforms.",
    "cc_mean_transforms.": "latent_codec.cc_mean_transforms.",
    "cc_scale_transforms.": "latent_codec.cc_scale_transforms.",
    "lrp_transforms.": "latent_codec.lrp_transforms.",
}

_UPSTREAM_SWATTEN_WRAPPER = re.compile(r"^(atten_mean|atten_scale)\.(\d+)\.0\.")


def _group_consecutive(indices: Iterable[int]) -> List[List[int]]:
    grouped: List[List[int]] = []
    for index in sorted(indices):
        if not grouped or index != grouped[-1][-1] + 1:
            grouped.append([index])
            continue
        grouped[-1].append(index)
    return grouped


def _infer_stage_groups(state_dict: Dict[str, Tensor], prefix: str) -> List[List[int]]:
    indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith(f"{prefix}.") and ".conv1_1.weight" in key
    }
    return _group_consecutive(indices)


def _infer_stage_depths(state_dict: Dict[str, Tensor]) -> Optional[List[int]]:
    g_a_groups = _infer_stage_groups(state_dict, "g_a")
    g_s_groups = _infer_stage_groups(state_dict, "g_s")
    if len(g_a_groups) != 3 or len(g_s_groups) != 3:
        return None
    return [len(group) for group in g_a_groups + g_s_groups]


def _infer_head_dims(state_dict: Dict[str, Tensor], N: int) -> Optional[List[int]]:
    head_dims: List[int] = []
    for prefix in ("g_a", "g_s"):
        for group in _infer_stage_groups(state_dict, prefix):
            if not group:
                continue
            table_key = (
                f"{prefix}.{group[0]}.trans_block.msa.attn.relative_position_bias_table"
            )
            if table_key not in state_dict:
                return None
            num_heads = state_dict[table_key].size(1)
            head_dims.append(N // num_heads)
    return head_dims if len(head_dims) == 6 else None


def _infer_hyper_head_dim(
    state_dict: Dict[str, Tensor], N: int, default: int
) -> int:
    for key in (
        "h_a.1.trans_block.msa.attn.relative_position_bias_table",
        "h_mean_s.1.trans_block.msa.attn.relative_position_bias_table",
    ):
        if key in state_dict:
            return N // state_dict[key].size(1)
    return default


def _slice_support_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * index
    return latent_channels + slice_channels * min(index, max_support_slices)


def _lrp_support_channels(
    latent_channels: int,
    slice_channels: int,
    index: int,
    max_support_slices: int,
) -> int:
    if max_support_slices < 0:
        return latent_channels + slice_channels * (index + 1)
    return latent_channels + slice_channels * min(index + 1, max_support_slices + 1)


def _analysis_aux_positions(config: Sequence[int]) -> Tuple[int, int, int, int]:
    return (
        0,
        config[0] + 1,
        config[0] + config[1] + 2,
        config[0] + config[1] + config[2] + 3,
    )


def _synthesis_aux_positions(config: Sequence[int]) -> Tuple[int, int, int, int]:
    return (
        config[3],
        config[3] + config[4] + 1,
        config[3] + config[4] + config[5] + 2,
        config[3] + config[4] + config[5] + 3,
    )


def _has_auxt_state(state_dict: Dict[str, Tensor]) -> bool:
    return any(
        key.startswith("AuxT_enc.") or key.startswith("AuxT_dec.")
        for key in state_dict
    )


def _is_auxt_wavelet_buffer_key(key: str) -> bool:
    if not (key.startswith("AuxT_enc.") or key.startswith("AuxT_dec.")):
        return False
    return ".dwt.transform." in key or ".idwt.inverse." in key


def _is_auxt_upstream_wavelet_buffer_key(key: str) -> bool:
    if key.startswith("AuxT_enc.") and ".dwt." in key:
        suffix = key.rsplit(".", 1)[-1]
        return suffix in {"w_ll", "w_lh", "w_hl", "w_hh"}
    if key.startswith("AuxT_dec.") and ".idwt." in key:
        return key.rsplit(".", 1)[-1] == "filters"
    return False


def _make_entropy_transform(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        conv(in_channels, 224, stride=1, kernel_size=3),
        nn.GELU(),
        conv(224, 128, stride=1, kernel_size=3),
        nn.GELU(),
        conv(128, out_channels, stride=1, kernel_size=3),
    )


def _make_mixed_stage(
    depth: int,
    branch_channels: int,
    head_dim: int,
    window_size: int,
    drop_paths: Sequence[float],
    tail: nn.Module,
) -> List[nn.Module]:
    if len(drop_paths) != depth:
        raise ValueError("drop_paths must match stage depth")
    blocks = [
        ConvTransBlock(
            branch_channels,
            branch_channels,
            head_dim,
            window_size,
            drop_paths[index],
            type="W" if index % 2 == 0 else "SW",
        )
        for index in range(depth)
    ]
    return [*blocks, tail]


@register_model("lic-tcm")
@register_model("tcm")
class TCM(CompressionModel):
    r"""TCM model from J. Liu, H. Sun, J. Katto: `"Learned Image Compression
    with Mixed Transformer-CNN Architectures"
    <https://arxiv.org/abs/2303.14978>`_, IEEE/CVF Conf. on Computer Vision
    and Pattern Recognition (CVPR), 2023 (Highlight).

    Stacks parallel Transformer-CNN Mixture (TCM) blocks for the
    analysis/synthesis transforms and uses a channel-wise autoregressive
    entropy model with parameter-efficient swin-transformer attention
    (``SWAtten``).

    Optionally enables AuxT disentangled training auxiliary transforms
    (``use_auxt=True``, see Li et al., ICLR 2025) and the Causal Context
    Adjustment loss entropy model (``use_cca=True``, see Han et al.,
    NeurIPS 2024).

    Args:
        N (int): Number of channels in the hyperprior backbone.
        M (int): Number of channels in the latent representation.
        num_slices (int): Number of channel slices for the entropy model.
    """

    def __init__(
        self,
        config: Optional[Sequence[int]] = None,
        head_dim: Optional[Sequence[int]] = None,
        drop_path_rate: float = 0.0,
        N: int = 128,
        M: int = 320,
        hyper_channels: int = 192,
        num_slices: int = 5,
        max_support_slices: int = 5,
        window_size: int = 8,
        hyper_window_size: int = 4,
        hyper_head_dim: int = 32,
        use_auxt: bool = False,
        use_cca: bool = False,
        cca_hidden_channels: int = 224,
        cca_num_layers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        config = tuple(int(value) for value in (config or (2, 2, 2, 2, 2, 2)))
        head_dim = tuple(int(value) for value in (head_dim or (8, 16, 32, 32, 16, 8)))
        if len(config) != 6:
            raise ValueError("config must provide six stage depths")
        if len(head_dim) != 6:
            raise ValueError("head_dim must provide six stage head dimensions")
        if any(value < 0 for value in config):
            raise ValueError("config values must be non-negative")
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")
        if any(N % value != 0 for value in head_dim):
            raise ValueError("Each head_dim must divide N")
        if N % hyper_head_dim != 0:
            raise ValueError("hyper_head_dim must divide N")

        self.config = config
        self.head_dim = head_dim
        self.window_size = int(window_size)
        self.hyper_window_size = int(hyper_window_size)
        self.hyper_head_dim = int(hyper_head_dim)
        self.N = int(N)
        self.M = int(M)
        self.hyper_channels = int(hyper_channels)
        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)
        self._analysis_aux_positions = _analysis_aux_positions(config)
        self._synthesis_aux_positions = _synthesis_aux_positions(config)

        drop_paths = torch.linspace(0, drop_path_rate, sum(config)).tolist()
        offset = 0

        def stage_drop_paths(depth: int) -> List[float]:
            nonlocal offset
            values = [float(value) for value in drop_paths[offset : offset + depth]]
            offset += depth
            return values

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, 2 * N, stride=2),
            *_make_mixed_stage(
                config[0],
                N,
                head_dim[0],
                self.window_size,
                stage_drop_paths(config[0]),
                ResidualBlockWithStride(2 * N, 2 * N, stride=2),
            ),
            *_make_mixed_stage(
                config[1],
                N,
                head_dim[1],
                self.window_size,
                stage_drop_paths(config[1]),
                ResidualBlockWithStride(2 * N, 2 * N, stride=2),
            ),
            *_make_mixed_stage(
                config[2],
                N,
                head_dim[2],
                self.window_size,
                stage_drop_paths(config[2]),
                conv3x3(2 * N, M, stride=2),
            ),
        )
        self.g_s = nn.Sequential(
            ResidualBlockUpsample(M, 2 * N, 2),
            *_make_mixed_stage(
                config[3],
                N,
                head_dim[3],
                self.window_size,
                stage_drop_paths(config[3]),
                ResidualBlockUpsample(2 * N, 2 * N, 2),
            ),
            *_make_mixed_stage(
                config[4],
                N,
                head_dim[4],
                self.window_size,
                stage_drop_paths(config[4]),
                ResidualBlockUpsample(2 * N, 2 * N, 2),
            ),
            *_make_mixed_stage(
                config[5],
                N,
                head_dim[5],
                self.window_size,
                stage_drop_paths(config[5]),
                subpel_conv3x3(2 * N, 3, 2),
            ),
        )

        self.h_a = nn.Sequential(
            ResidualBlockWithStride(M, 2 * N, 2),
            *_make_mixed_stage(
                config[0],
                N,
                self.hyper_head_dim,
                self.hyper_window_size,
                [0.0] * config[0],
                conv3x3(2 * N, hyper_channels, stride=2),
            ),
        )
        self.h_mean_s = nn.Sequential(
            ResidualBlockUpsample(hyper_channels, 2 * N, 2),
            *_make_mixed_stage(
                config[3],
                N,
                self.hyper_head_dim,
                self.hyper_window_size,
                [0.0] * config[3],
                subpel_conv3x3(2 * N, M, 2),
            ),
        )
        self.h_scale_s = nn.Sequential(
            ResidualBlockUpsample(hyper_channels, 2 * N, 2),
            *_make_mixed_stage(
                config[3],
                N,
                self.hyper_head_dim,
                self.hyper_window_size,
                [0.0] * config[3],
                subpel_conv3x3(2 * N, M, 2),
            ),
        )

        slice_channels = M // num_slices
        mean_support_transforms = nn.ModuleList(
            SWAtten(
                _slice_support_channels(M, slice_channels, index, max_support_slices),
                _slice_support_channels(M, slice_channels, index, max_support_slices),
                16,
                self.window_size,
                0.0,
                inter_dim=128,
            )
            for index in range(num_slices)
        )
        scale_support_transforms = nn.ModuleList(
            SWAtten(
                _slice_support_channels(M, slice_channels, index, max_support_slices),
                _slice_support_channels(M, slice_channels, index, max_support_slices),
                16,
                self.window_size,
                0.0,
                inter_dim=128,
            )
            for index in range(num_slices)
        )
        cc_mean_transforms = nn.ModuleList(
            _make_entropy_transform(
                _slice_support_channels(M, slice_channels, index, max_support_slices),
                slice_channels,
            )
            for index in range(num_slices)
        )
        cc_scale_transforms = nn.ModuleList(
            _make_entropy_transform(
                _slice_support_channels(M, slice_channels, index, max_support_slices),
                slice_channels,
            )
            for index in range(num_slices)
        )
        lrp_transforms = nn.ModuleList(
            _make_entropy_transform(
                _lrp_support_channels(M, slice_channels, index, max_support_slices),
                slice_channels,
            )
            for index in range(num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)
        self.latent_codec = ChannelSliceLatentCodec(
            cc_mean_transforms=cc_mean_transforms,
            cc_scale_transforms=cc_scale_transforms,
            lrp_transforms=lrp_transforms,
            gaussian_conditional=GaussianConditional(None),
            mean_support_transforms=mean_support_transforms,
            scale_support_transforms=scale_support_transforms,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            quantizer="ste",
        )
        self.AuxT_enc = (
            nn.ModuleList(
                [
                    WLS(3, 2 * N),
                    WLS(2 * N, 2 * N),
                    WLS(2 * N, 2 * N),
                    WLS(2 * N, M),
                ]
            )
            if use_auxt
            else None
        )
        self.AuxT_dec = (
            nn.ModuleList(
                [
                    iWLS(M, 2 * N),
                    iWLS(2 * N, 2 * N),
                    iWLS(2 * N, 2 * N),
                    iWLS(2 * N, 3),
                ]
            )
            if use_auxt
            else None
        )
        self.cca_aux_entropy_model = (
            CausalContextAdjustmentEntropyModel(
                latent_channels=M,
                num_slices=num_slices,
                hidden_channels=cca_hidden_channels,
                num_layers=cca_num_layers,
            )
            if use_cca
            else None
        )

    @property
    def gaussian_conditional(self) -> GaussianConditional:
        return self.latent_codec.gaussian_conditional

    @property
    def atten_mean(self) -> nn.ModuleList:
        return self.latent_codec.mean_support_transforms

    @property
    def atten_scale(self) -> nn.ModuleList:
        return self.latent_codec.scale_support_transforms

    @property
    def cc_mean_transforms(self) -> nn.ModuleList:
        return self.latent_codec.cc_mean_transforms

    @property
    def cc_scale_transforms(self) -> nn.ModuleList:
        return self.latent_codec.cc_scale_transforms

    @property
    def lrp_transforms(self) -> nn.ModuleList:
        return self.latent_codec.lrp_transforms

    @property
    def use_cca(self) -> bool:
        return self.cca_aux_entropy_model is not None

    @property
    def use_auxt(self) -> bool:
        return self.AuxT_enc is not None and self.AuxT_dec is not None

    def _forward_with_auxt(
        self,
        transform: nn.Sequential,
        auxiliary_layers: Optional[nn.ModuleList],
        merge_positions: Sequence[int],
        input_tensor: Tensor,
    ) -> Tensor:
        if auxiliary_layers is None:
            return transform(input_tensor)

        output = input_tensor
        auxiliary = input_tensor
        aux_index = 0
        for layer_index, layer in enumerate(transform):
            output = layer(output)
            if layer_index in merge_positions:
                auxiliary = auxiliary_layers[aux_index](auxiliary)
                output = output + auxiliary
                aux_index += 1

        if aux_index != len(auxiliary_layers):
            raise RuntimeError("AuxT merge positions do not match auxiliary depth.")
        return output

    def _analysis_transform(self, x: Tensor) -> Tensor:
        return self._forward_with_auxt(
            self.g_a,
            self.AuxT_enc,
            self._analysis_aux_positions,
            x,
        )

    def _synthesis_transform(self, y_hat: Tensor) -> Tensor:
        return self._forward_with_auxt(
            self.g_s,
            self.AuxT_dec,
            self._synthesis_aux_positions,
            y_hat,
        )

    def ortho_loss(self) -> Tensor:
        losses = [module.loss() for module in self.modules() if isinstance(module, OLP)]
        if losses:
            return torch.stack(losses).sum()
        parameter = next(self.parameters())
        return torch.zeros((), device=parameter.device, dtype=parameter.dtype)

    def forward(self, x: Tensor) -> Dict[str, Union[Dict[str, Tensor], Tensor]]:
        y = self._analysis_transform(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_out = self.latent_codec(y, latent_means, latent_scales)
        output: Dict[str, Union[Dict[str, Tensor], Tensor]] = {
            "x_hat": self._synthesis_transform(y_out["y_hat"]),
            "likelihoods": {"y": y_out["likelihoods"]["y"], "z": z_likelihoods},
        }
        if self.cca_aux_entropy_model is not None:
            output["aux_likelihoods"] = self.cca_aux_entropy_model(
                y,
                latent_means,
                latent_scales,
            )
        return output

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self._analysis_transform(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        y_out = self.latent_codec.compress(
            y,
            self.h_mean_s(z_hat),
            self.h_scale_s(z_hat),
        )
        return {
            "strings": [[y_out["strings"][0]], z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        y_shape = (z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        y_out = self.latent_codec.decompress(
            strings[0],
            y_shape,
            self.h_mean_s(z_hat),
            self.h_scale_s(z_hat),
        )
        return {"x_hat": self._synthesis_transform(y_out["y_hat"]).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "TCM":
        state_dict = cls._migrate_state_dict(state_dict)
        N = state_dict["g_a.0.conv1.weight"].size(0) // 2
        M = state_dict["h_a.0.conv1.weight"].size(1)
        config = _infer_stage_depths(state_dict) or [2, 2, 2, 2, 2, 2]
        head_dim = _infer_head_dims(state_dict, N) or [8, 16, 32, 32, 16, 8]
        hyper_channels = state_dict["entropy_bottleneck.quantiles"].size(0)
        num_slices = infer_num_slices(state_dict) or 5
        max_support_slices = infer_max_support_slices(state_dict, M, num_slices)
        net = cls(
            config=config,
            head_dim=head_dim,
            N=N,
            M=M,
            hyper_channels=hyper_channels,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            hyper_head_dim=_infer_hyper_head_dim(state_dict, N, 32),
            use_auxt=_has_auxt_state(state_dict),
            use_cca=has_cca_aux_state(state_dict),
            cca_hidden_channels=infer_cca_hidden_channels(state_dict),
            cca_num_layers=infer_cca_num_layers(state_dict),
        )
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            key
            for key in net.state_dict()
            if key.endswith("relative_position_index")
            or _is_auxt_wavelet_buffer_key(key)
        }
        missing_keys = set(incompatible_keys.missing_keys) - allowed_missing
        if missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                "Unexpected incompatibility while loading TCM state_dict: "
                f"missing={sorted(missing_keys)}, "
                f"unexpected={sorted(incompatible_keys.unexpected_keys)}"
            )
        return net

    @classmethod
    def _migrate_state_dict(cls, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        migrated: Dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if _is_auxt_upstream_wavelet_buffer_key(key):
                continue
            new_key = key
            if new_key.startswith("AuxT_enc.") or new_key.startswith("AuxT_dec."):
                new_key = new_key.replace(".OLP.", ".olp.")
            wrapper = _UPSTREAM_SWATTEN_WRAPPER.match(new_key)
            if wrapper:
                new_key = (
                    f"{wrapper.group(1)}.{wrapper.group(2)}." + new_key[wrapper.end():]
                )
            for legacy_prefix, target_prefix in _LEGACY_LATENT_PREFIX_MAP.items():
                if new_key.startswith(legacy_prefix):
                    new_key = f"{target_prefix}{new_key.removeprefix(legacy_prefix)}"
                    break

            if ".msa.relative_position_params" in new_key:
                new_key = new_key.replace(
                    ".msa.relative_position_params",
                    ".msa.attn.relative_position_bias_table",
                )
                value = value.permute(1, 2, 0).reshape(-1, value.size(0)).contiguous()
            elif ".msa.embedding_layer." in new_key:
                new_key = new_key.replace(".msa.embedding_layer.", ".msa.attn.qkv.")
            elif ".msa.linear." in new_key:
                new_key = new_key.replace(".msa.linear.", ".msa.output_proj.")
                cls._ensure_identity_attention_projection(migrated, new_key, value)

            new_key = new_key.replace(".ln1.", ".norm1.")
            new_key = new_key.replace(".ln2.", ".norm2.")
            new_key = new_key.replace(".mlp.0.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.2.", ".mlp.fc2.")

            migrated[new_key] = value
        return migrated

    @staticmethod
    def _ensure_identity_attention_projection(
        state_dict: Dict[str, Tensor],
        output_proj_key: str,
        output_proj_value: Tensor,
    ) -> None:
        prefix, suffix = output_proj_key.rsplit(".msa.output_proj.", 1)
        attn_proj_key = f"{prefix}.msa.attn.proj.{suffix}"
        if attn_proj_key in state_dict:
            return
        if suffix == "weight":
            dimension = output_proj_value.size(0)
            state_dict[attn_proj_key] = torch.eye(
                dimension,
                dtype=output_proj_value.dtype,
                device=output_proj_value.device,
            )
            return
        if suffix == "bias":
            state_dict[attn_proj_key] = torch.zeros_like(output_proj_value)
