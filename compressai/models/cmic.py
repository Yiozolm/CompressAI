from __future__ import annotations

import math
import re

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    EntropyBottleneckLatentCodec,
    GaussianConditionalLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import sequential_channel_ramp, subpel_conv3x3
from compressai.layers.lic import (
    CMICAnalysisTransform,
    CMICChannelContextBlock,
    CMICSpatialContextBlock,
    CMICSynthesisTransform,
    GatedTransformCNN,
    OLP,
)
from compressai.layers.wave import is_pytorch_wavelets_available
from compressai.models.utils import conv
from compressai.registry import register_model

from .base import SimpleVAECompressionModel

__all__ = [
    "CMIC",
    "CMICAnalysisTransform",
    "CMICSynthesisTransform",
    "convert_upstream_state_dict",
]

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
        "CMIC requires the optional dependency `pytorch_wavelets`. "
        "Install `compressai[lic]` to enable this model."
    )


def _default_groups(M: int) -> List[int]:
    if M > 128:
        return [16, 16, 32, 64, M - 128]
    num_groups = min(4, M)
    base, remainder = divmod(M, num_groups)
    groups = [base] * num_groups
    for index in range(remainder):
        groups[-(index + 1)] += 1
    return [group for group in groups if group > 0]


_CMIC_STAGE_PREFIXES = ("g_a.g2.", "g_a.g3.", "g_s.g1.", "g_s.g2.")


def _is_upstream_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    return any(
        ".residual_group.layers." in key
        or key.startswith("latent_codec.hyper.")
        for key in state_dict
    )


def convert_upstream_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream ``CMIC_AuxT`` checkpoint to compressai layout.

    Handles all known structural and naming differences:

    * drops candidate-side DWT/IDWT haar-coefficient buffers (compressai
      reinitialises them through ``pytorch_wavelets``);
    * renames ``OLP`` to ``olp`` inside ``WLS``/``iWLS`` modules;
    * inside ``CMIC_stage`` blocks (``g_a.g2``/``g_a.g3``/``g_s.g1``/``g_s.g2``):
      ``residual_group.layers.{j}`` -> ``blocks.{j}``,
      ``wqkv``/``win_mhsa`` -> ``window_attention.qkv``/``window_attention``,
      ``convffn{i}.conv`` -> ``feed_forward{i}.depthwise``,
      ``convffn{i}`` -> ``feed_forward{i}``,
      ``assm`` -> ``content_model`` with ``selectiveScan`` flattened,
      ``in_proj.0``/``CPE.0`` -> ``in_proj``/``cpe``,
      ``cal_embedding`` -> ``prompt_proj``;
    * candidate-side ``win_mhsa.relative_position_bias_table`` is zeroed since
      the upstream forward never adds it; compressai's forward does;
    * hyperprior layout: ``latent_codec.hyper.{h_a,h_s}`` is hoisted to
      ``latent_codec.{h_a,h_s}`` and ``latent_codec.hyper.entropy_bottleneck``
      becomes ``latent_codec.z.entropy_bottleneck``;
    * spatial context: ``ly{1,2}`` -> ``layer{1,2}``, ``mixer.depth_conv`` ->
      ``mixer.masked_conv``;
    * ``norm2`` -> ``norm`` for ``GatedTransformCNN``/``Param_Gated``/
      ``Param_Agg_Block`` (CMIC block ``norm{1..4}`` are preserved);
    * ``mixer.adaptor`` -> ``mixer.skip`` for residual depthwise mixers.
    """
    new_state_dict: Dict[str, Tensor] = {}

    for key, value in state_dict.items():
        if re.search(r"AuxT_(?:enc|dec)\.\d+\.dwt\.w_(?:ll|lh|hl|hh)$", key):
            continue
        if re.search(r"AuxT_(?:enc|dec)\.\d+\.(?:dwt|idwt)\.filters$", key):
            continue

        new_key = key
        new_value = value

        new_key = re.sub(
            r"(AuxT_(?:enc|dec)\.\d+)\.OLP\.", r"\1.olp.", new_key
        )

        is_cmic_stage = any(prefix in key for prefix in _CMIC_STAGE_PREFIXES)
        if is_cmic_stage:
            new_key = re.sub(
                r"\.residual_group\.layers\.(\d+)\.",
                r".blocks.\1.",
                new_key,
            )
            new_key = re.sub(
                r"\.blocks\.(\d+)\.wqkv\.",
                r".blocks.\1.window_attention.qkv.",
                new_key,
            )
            new_key = re.sub(
                r"\.blocks\.(\d+)\.win_mhsa\.",
                r".blocks.\1.window_attention.",
                new_key,
            )
            new_key = re.sub(
                r"\.convffn(\d)\.conv\.",
                r".feed_forward\1.depthwise.",
                new_key,
            )
            new_key = re.sub(
                r"\.convffn(\d)\.", r".feed_forward\1.", new_key
            )
            if ".assm." in new_key:
                new_key = new_key.replace(".assm.", ".content_model.")
                new_key = new_key.replace(
                    ".content_model.selectiveScan.", ".content_model."
                )
                new_key = re.sub(
                    r"\.content_model\.in_proj\.0\.",
                    r".content_model.in_proj.",
                    new_key,
                )
                new_key = re.sub(
                    r"\.content_model\.CPE\.0\.",
                    r".content_model.cpe.",
                    new_key,
                )
                new_key = new_key.replace(
                    ".content_model.cal_embedding.",
                    ".content_model.prompt_proj.",
                )

        new_key = new_key.replace(
            "latent_codec.hyper.entropy_bottleneck.",
            "latent_codec.z.entropy_bottleneck.",
        )
        new_key = new_key.replace(
            "latent_codec.hyper.h_a.", "latent_codec.h_a."
        )
        new_key = new_key.replace(
            "latent_codec.hyper.h_s.", "latent_codec.h_s."
        )

        new_key = re.sub(
            r"context_prediction\.ly(\d)\.",
            r"context_prediction.layer\1.",
            new_key,
        )
        if "context_prediction.layer" in new_key:
            new_key = new_key.replace(
                "mixer.depth_conv.", "mixer.masked_conv."
            )

        if not re.match(r"g_[as]\.g\d+\.blocks\.\d+\.norm[1234]\.", new_key):
            new_key = new_key.replace(".norm2.", ".norm.")

        new_key = new_key.replace(".mixer.adaptor.", ".mixer.skip.")

        if new_key.endswith(".window_attention.relative_position_bias_table"):
            new_value = torch.zeros_like(value)

        new_state_dict[new_key] = new_value

    return new_state_dict


@_maybe_register_model("cmic")
class CMIC(SimpleVAECompressionModel):
    r"""Content-Aware Mamba Image Compression model from Y. Chen, Z. Hu,
    et al.: `"Content-Aware Mamba for Learned Image Compression"
    <https://openreview.net/forum?id=WwDNiisZQm>`_, Int. Conf. on Learning
    Representations (ICLR), 2026.

    Combines wavelet/graph auxiliary branches with content-adaptive Mamba
    state-space blocks and a checkerboard / channel-group hyperprior.

    Args:
        N (int): Number of channels in the hyperprior.
        M (int): Number of channels in the latent representation.
    """

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        groups: Optional[List[int]] = None,
        stage_dims: Tuple[int, int, int] = (128, 192, 256),
        stage_depths: Tuple[int, int] = (2, 2),
        num_heads: Tuple[int, int] = (8, 8),
        d_state: int = 8,
        window_size: int = 8,
        inner_rank: int = 32,
        cluster_num: int = 64,
        stage_mlp_ratio: float = 3.0,
        **kwargs: Any,
    ) -> None:
        _require_wavelets()
        super().__init__(**kwargs)

        if len(stage_dims) != 3:
            raise ValueError("`stage_dims` must contain three feature dimensions.")
        if len(stage_depths) != 2:
            raise ValueError("`stage_depths` must contain two stage depths.")
        if len(num_heads) != 2:
            raise ValueError("`num_heads` must contain two head counts.")

        self.N = int(N)
        self.M = int(M)
        self.stage_dims = tuple(int(dim) for dim in stage_dims)
        self.stage_depths = tuple(int(depth) for depth in stage_depths)
        self.num_heads = tuple(int(head) for head in num_heads)
        self.d_state = int(d_state)
        self.window_size = int(window_size)
        self.inner_rank = int(inner_rank)
        self.cluster_num = int(cluster_num)
        self.stage_mlp_ratio = float(stage_mlp_ratio)
        self.groups = list(groups) if groups is not None else _default_groups(M)
        if sum(self.groups) != M:
            raise ValueError("Channel groups must sum to M.")

        self.g_a = CMICAnalysisTransform(
            M=M,
            stage_dims=self.stage_dims,
            stage_depths=self.stage_depths,
            num_heads=self.num_heads,
            d_state=self.d_state,
            window_size=self.window_size,
            inner_rank=self.inner_rank,
            cluster_num=self.cluster_num,
            stage_mlp_ratio=self.stage_mlp_ratio,
        )
        self.g_s = CMICSynthesisTransform(
            M=M,
            stage_dims=self.stage_dims,
            stage_depths=self.stage_depths,
            num_heads=self.num_heads,
            d_state=self.d_state,
            window_size=self.window_size,
            inner_rank=self.inner_rank,
            cluster_num=self.cluster_num,
            stage_mlp_ratio=self.stage_mlp_ratio,
        )

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N, kernel_size=3, stride=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N, kernel_size=3, stride=2),
        )
        h_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            subpel_conv3x3(N, N, 2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N * 2, kernel_size=3, stride=1),
        )

        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=GatedTransformCNN,
                make_act=lambda: nn.Identity(),
                expansion_factor=4,
            )
            for k in range(1, len(self.groups))
        }
        spatial_context = [
            CMICSpatialContextBlock(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]
        param_aggregation = [
            sequential_channel_ramp(
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=CMICChannelContextBlock,
                make_act=lambda: nn.Identity(),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={"y": GaussianConditionalLatentCodec(quantizer="ste")},
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        self.latent_codec = HyperpriorLatentCodec(
            h_a=h_a,
            h_s=h_s,
            latent_codec={
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                "z": EntropyBottleneckLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    quantizer="ste",
                ),
            },
        )

    def ortho_loss(self) -> Tensor:
        loss = sum(module.loss() for module in self.modules() if isinstance(module, OLP))
        return cast(Tensor, loss)

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "CMIC":
        if _is_upstream_state_dict(state_dict):
            state_dict = convert_upstream_state_dict(state_dict)

        N = state_dict["latent_codec.z.entropy_bottleneck.quantiles"].size(0)
        M = state_dict["g_a.down3.weight"].size(0)
        stage_dims = (
            state_dict["g_a.down0.weight"].size(0),
            state_dict["g_a.down1.weight"].size(0),
            state_dict["g_a.down2.weight"].size(0),
        )
        stage_depths = (
            cls._infer_depth(state_dict, "g_a.g2"),
            cls._infer_depth(state_dict, "g_a.g3"),
        )
        num_heads = (
            state_dict["g_a.g2.blocks.0.window_attention.relative_position_bias_table"].size(1),
            state_dict["g_a.g3.blocks.0.window_attention.relative_position_bias_table"].size(1),
        )
        table_size = state_dict[
            "g_a.g2.blocks.0.window_attention.relative_position_bias_table"
        ].size(0)
        groups = cls._infer_groups(state_dict)

        net = cls(
            N=N,
            M=M,
            groups=groups or None,
            stage_dims=stage_dims,
            stage_depths=stage_depths,
            num_heads=num_heads,
            d_state=state_dict["g_a.g2.blocks.0.content_model.A_logs"].size(1),
            window_size=(math.isqrt(table_size) + 1) // 2,
            cluster_num=state_dict["g_a.g2.blocks.0.content_model.means"].size(0),
        )

        # Compressai's WindowAttention registers a `relative_position_index`
        # buffer, and `pytorch_wavelets`'s `DWTForward`/`DWTInverse` register
        # their haar coefficient buffers, that the upstream checkpoint does not
        # ship. Backfill from the freshly-initialised model so we can keep
        # `strict=True`.
        init_state_dict = net.state_dict()
        for key, value in init_state_dict.items():
            if key in state_dict:
                continue
            if (
                key.endswith(".window_attention.relative_position_index")
                or ".dwt.transform." in key
                or ".idwt.inverse." in key
            ):
                state_dict[key] = value

        net.load_state_dict(state_dict)
        return net

    @staticmethod
    def _infer_depth(state_dict: Dict[str, Tensor], prefix: str) -> int:
        depth = 0
        while f"{prefix}.blocks.{depth}.norm1.weight" in state_dict:
            depth += 1
        return depth

    @staticmethod
    def _infer_groups(state_dict: Dict[str, Tensor]) -> List[int]:
        groups = []
        index = 0
        while True:
            key = (
                "latent_codec.y.latent_codec."
                f"y{index}.context_prediction.layer1.mixer.conv1.0.weight"
            )
            if key not in state_dict:
                key = f"latent_codec.y.latent_codec.y{index}.context_prediction.weight"
            if key not in state_dict:
                break
            groups.append(state_dict[key].size(1))
            index += 1
        return groups
