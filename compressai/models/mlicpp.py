from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.latent_codecs import MLICPlusPlusLatentCodec
from compressai.layers.lic.mlic import AnalysisTransform, SynthesisTransform
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["MLICPlusPlus"]

_LEGACY_LATENT_PREFIXES = (
    "h_a.",
    "h_s.",
    "entropy_bottleneck.",
    "gaussian_conditional.",
    "local_context.",
    "channel_context.",
    "global_inter_context.",
    "global_intra_context.",
    "entropy_parameters_anchor.",
    "entropy_parameters_nonanchor.",
    "lrp_anchor.",
    "lrp_nonanchor.",
)


@register_model("mlicpp")
class MLICPlusPlus(CompressionModel):
    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        slice_num: int = 10,
        context_window: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if slice_num <= 0:
            raise ValueError("slice_num must be positive")
        if context_window % 2 == 0:
            raise ValueError("context_window must be odd")
        if M % slice_num != 0:
            raise ValueError("M must be divisible by slice_num")

        self.N = int(N)
        self.M = int(M)
        self.context_window = int(context_window)
        self.slice_num = int(slice_num)
        self.slice_ch = int(M // slice_num)

        self.g_a = AnalysisTransform(N=N, M=M)
        self.g_s = SynthesisTransform(N=N, M=M)
        self.latent_codec = MLICPlusPlusLatentCodec(
            N=N,
            M=M,
            slice_num=slice_num,
            context_window=context_window,
        )

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    @property
    def h_a(self) -> nn.Module:
        return self.latent_codec.h_a

    @property
    def h_s(self) -> nn.Module:
        return self.latent_codec.h_s

    @property
    def entropy_bottleneck(self) -> EntropyBottleneck:
        return self.latent_codec.entropy_bottleneck

    @property
    def gaussian_conditional(self) -> GaussianConditional:
        return self.latent_codec.gaussian_conditional

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        return {
            "x_hat": self.g_s(y_out["y_hat"]),
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        y_out = self.latent_codec.compress(y)
        return {"strings": y_out["strings"], "shape": y_out["shape"]}

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        y_out = self.latent_codec.decompress(strings, shape)
        return {"x_hat": self.g_s(y_out["y_hat"]).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "MLICPlusPlus":
        state_dict = cls._migrate_state_dict(state_dict)
        N = state_dict["g_a.analysis_transform.0.conv1.weight"].size(0)
        M = state_dict["g_a.analysis_transform.6.weight"].size(0)
        slice_indices = {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("latent_codec.local_context.")
            and key.endswith(".relative_position_table")
        }
        slice_num = len(slice_indices) or 10
        context_tokens = state_dict[
            "latent_codec.local_context.0.relative_position_index"
        ].size(0)
        context_window = int(round(context_tokens**0.5))
        net = cls(N=N, M=M, slice_num=slice_num, context_window=context_window)
        net.load_state_dict(state_dict)
        return net

    @staticmethod
    def _migrate_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            (
                f"latent_codec.{key}"
                if not key.startswith("latent_codec.")
                and key.startswith(_LEGACY_LATENT_PREFIXES)
                else key
            ): value
            for key, value in state_dict.items()
        }
