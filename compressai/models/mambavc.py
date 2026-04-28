from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models.cca import (
    has_cca_aux_state,
    infer_cca_hidden_channels,
    infer_cca_num_layers,
)
from compressai.layers.attn import (
    SWAtten,
    infer_swatten_attention_dim,
    infer_swatten_head_dim,
    infer_swatten_window_size,
)
from compressai.layers.ssm import (
    build_vss_backbone,
    infer_vss_block_kwargs,
    infer_vss_depths,
)
from compressai.models._bases import (
    SliceEntropyCompressionModel,
    infer_max_support_slices,
    infer_num_slices,
    slice_support_channels,
)
from compressai.registry import register_model

__all__ = ["MambaVC"]


@register_model("mambavc")
class MambaVC(SliceEntropyCompressionModel):
    r"""MambaVC model from S. Qin, J. Wang, Y. Zhou, B. Chen, T. Luo, B. An,
    T. Dai, S. Xia, Y. Wang: `"MambaVC: Learned Visual Compression with
    Selective State Spaces" <https://arxiv.org/abs/2405.15413>`_,
    arXiv:2405.15413, 2024.

    Replaces the nonlinear activation after each downsampling stage with a
    visual state-space (VSS) block built on a 2D selective scanning (2DSS)
    module, paired with a Minnen2020-style channel-wise autoregressive
    entropy model.

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
        window_size: int = 8,
        support_head_dim: int = 16,
        support_attention_dim: int = 128,
        use_cca: bool = False,
        cca_hidden_channels: int = 224,
        cca_num_layers: int = 4,
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
        self.window_size = int(window_size)
        self.support_head_dim = int(support_head_dim)
        self.support_attention_dim = int(support_attention_dim)

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

        slice_channels = M // num_slices
        mean_support_transforms = nn.ModuleList()
        scale_support_transforms = nn.ModuleList()
        for index in range(num_slices):
            in_channels = slice_support_channels(
                M,
                slice_channels,
                index,
                max_support_slices,
            )
            mean_support_transforms.append(
                SWAtten(
                    in_channels,
                    in_channels,
                    support_head_dim,
                    window_size,
                    0.0,
                    inter_dim=support_attention_dim,
                )
            )
            scale_support_transforms.append(
                SWAtten(
                    in_channels,
                    in_channels,
                    support_head_dim,
                    window_size,
                    0.0,
                    inter_dim=support_attention_dim,
                )
            )

        self._init_slice_entropy(
            M,
            hyper_channels,
            num_slices,
            max_support_slices,
            use_cca=use_cca,
            cca_hidden_channels=cca_hidden_channels,
            cca_num_layers=cca_num_layers,
            mean_support_transforms=mean_support_transforms,
            scale_support_transforms=scale_support_transforms,
        )

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        latent_output = self._forward_latent_output(y)
        output: Dict[str, Dict[str, Tensor] | Tensor] = {
            "x_hat": self.g_s(latent_output["y_hat"]),
            "likelihoods": latent_output["likelihoods"],
        }
        if "aux_likelihoods" in latent_output:
            output["aux_likelihoods"] = latent_output["aux_likelihoods"]
        return output

    def compress(self, x: Tensor) -> Dict[str, object]:
        return self._compress_latent(self.g_a(x))

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        return {"x_hat": self.g_s(self._decompress_latent(strings, shape)).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "MambaVC":
        depths = infer_vss_depths(state_dict)
        vss_kwargs = infer_vss_block_kwargs(state_dict)
        N = state_dict["g_a.0.weight"].size(0) // 2
        M = state_dict["h_a.0.weight"].size(1)
        hyper_channels = state_dict["entropy_bottleneck.quantiles"].size(0)
        num_slices = infer_num_slices(state_dict) or 5
        max_support_slices = infer_max_support_slices(state_dict, M, num_slices)
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
        net = cls(
            depths=depths,
            N=N,
            M=M,
            hyper_channels=hyper_channels,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            window_size=window_size,
            support_head_dim=support_head_dim,
            support_attention_dim=support_attention_dim,
            use_cca=has_cca_aux_state(state_dict),
            cca_hidden_channels=infer_cca_hidden_channels(state_dict),
            cca_num_layers=infer_cca_num_layers(state_dict),
            **vss_kwargs,
        )
        net.load_state_dict(state_dict)
        return net
