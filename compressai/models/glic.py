from typing import Any, Dict, List, Optional, Tuple, cast

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import (
    CheckerboardMaskedConv2d,
    sequential_channel_ramp,
    subpel_conv3x3,
)
from compressai.layers.lic import (
    GFA,
    GatedFFN,
    GatedTransformCNN,
    LayerNorm2d,
    OLP,
    WLS,
    iWLS,
)
from compressai.models.utils import conv
from compressai.registry import register_model

from .base import SimpleVAECompressionModel

__all__ = [
    "GLIC",
    "GLICAnalysisTransform",
    "GLICParameterAggregationBlock",
    "GLICSynthesisTransform",
]


class GLICParameterAggregationBlock(nn.Module):
    """Pointwise mixing plus gated FFN for GLIC parameter aggregation."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        expansion_factor: float = 4.0,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        del layer_kwargs
        self.mixer = nn.Conv2d(dim, dim_out, kernel_size=1, stride=1)
        self.norm = LayerNorm2d(dim_out)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.mixer(input_tensor)
        return output + self.mlp(self.norm(output))


class GLICAnalysisTransform(nn.Module):
    """GLIC analysis transform with auxiliary wavelet and graph branches."""

    def __init__(self, N: int = 192, M: int = 320) -> None:
        super().__init__()
        del N
        embed_dim0 = 128
        embed_dim1 = 192
        embed_dim2 = 192

        self.AuxT_enc = nn.Sequential(
            WLS(3, embed_dim0),
            WLS(embed_dim0, embed_dim1),
            WLS(embed_dim1, embed_dim2),
            WLS(embed_dim2, M),
        )
        self.g1 = nn.Sequential(
            GatedTransformCNN(embed_dim0, embed_dim0),
            GatedTransformCNN(embed_dim0, embed_dim0),
            GatedTransformCNN(embed_dim0, embed_dim0),
        )
        self.g2 = GFA(
            dim=embed_dim1,
            depth=5,
            num_heads=8,
            window_size=8,
            sample_size=16,
            graph_flags=True,
            top_k=64,
            diff_scales=1.5,
            stages=["GN", "GS", "GN", "GS", "GN", "GS"],
        )
        self.g3 = GFA(
            dim=embed_dim2,
            depth=5,
            num_heads=8,
            window_size=8,
            sample_size=16,
            graph_flags=True,
            top_k=64,
            diff_scales=1.5,
            stages=["GN", "GS", "GN", "GS", "GN", "GS"],
            mlp_ratio=2,
        )
        self.down0 = nn.Conv2d(3, embed_dim0, 3, stride=2, padding=1)
        self.down1 = nn.Conv2d(embed_dim0, embed_dim1, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(embed_dim1, embed_dim2, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(embed_dim2, M, 3, stride=2, padding=1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output, *_ = self.forward_energy(input_tensor)
        return output

    def forward_energy(self, input_tensor: Tensor) -> Tuple[Tensor, ...]:
        aux_output = input_tensor
        energies = []

        output = self.down0(input_tensor)
        output = self.g1(output)
        aux_output = self.AuxT_enc[0](aux_output)
        output = output + aux_output
        energies.append(output)

        output = self.down1(output)
        _, _, height, width = output.shape
        output = self.g2(output, (height, width))
        aux_output = self.AuxT_enc[1](aux_output)
        output = output + aux_output
        energies.append(output)

        output = self.down2(output)
        _, _, height, width = output.shape
        output = self.g3(output, (height, width))
        aux_output = self.AuxT_enc[2](aux_output)
        output = output + aux_output
        energies.append(output)

        output = self.down3(output)
        aux_output = self.AuxT_enc[3](aux_output)
        output = output + aux_output
        return (output, *energies)


class GLICSynthesisTransform(nn.Module):
    """GLIC synthesis transform mirroring the graph-auxiliary encoder."""

    def __init__(self, N: int = 192, M: int = 320) -> None:
        super().__init__()
        del N
        embed_dim1 = 128
        embed_dim2 = 192
        embed_dim3 = 192

        self.AuxT_dec = nn.Sequential(
            iWLS(M, embed_dim3),
            iWLS(embed_dim3, embed_dim2),
            iWLS(embed_dim2, embed_dim1),
            iWLS(embed_dim1, 3),
        )
        self.g1 = GFA(
            dim=embed_dim3,
            depth=5,
            num_heads=8,
            window_size=8,
            sample_size=16,
            graph_flags=True,
            top_k=64,
            diff_scales=1.5,
            stages=["GN", "GS", "GN", "GS", "GN", "GS"],
            mlp_ratio=2,
        )
        self.g2 = GFA(
            dim=embed_dim2,
            depth=5,
            num_heads=8,
            window_size=8,
            sample_size=16,
            graph_flags=True,
            top_k=64,
            diff_scales=1.5,
            stages=["GN", "GS", "GN", "GS", "GN", "GS"],
        )
        self.g3 = nn.Sequential(
            GatedTransformCNN(embed_dim1, embed_dim1),
            GatedTransformCNN(embed_dim1, embed_dim1),
            GatedTransformCNN(embed_dim1, embed_dim1),
        )
        self.up0 = subpel_conv3x3(M, embed_dim3, 2)
        self.up1 = subpel_conv3x3(embed_dim3, embed_dim2, 2)
        self.up2 = subpel_conv3x3(embed_dim2, embed_dim1, 2)
        self.up3 = subpel_conv3x3(embed_dim1, 3, 2)

    def forward(self, input_tensor: Tensor) -> Tensor:
        aux_output = input_tensor

        output = self.up0(input_tensor)
        _, _, height, width = output.shape
        output = self.g1(output, (height, width))
        aux_output = self.AuxT_dec[0](aux_output)
        output = output + aux_output

        output = self.up1(output)
        _, _, height, width = output.shape
        output = self.g2(output, (height, width))
        aux_output = self.AuxT_dec[1](aux_output)
        output = output + aux_output

        output = self.up2(output)
        output = self.g3(output)
        aux_output = self.AuxT_dec[2](aux_output)
        output = output + aux_output

        output = self.up3(output)
        aux_output = self.AuxT_dec[3](aux_output)
        return output + aux_output


@register_model("glic")
class GLIC(SimpleVAECompressionModel):
    """Graph-based learned image compression with auxiliary transforms."""

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        groups: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        if sum(self.groups) != M:
            raise ValueError("Channel groups must sum to M")

        self.g_a = GLICAnalysisTransform(N, M)
        self.g_s = GLICSynthesisTransform(N, M)

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
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }
        spatial_context = [
            CheckerboardMaskedConv2d(
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
                make_layer=GLICParameterAggregationBlock,
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
            latent_codec={
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            }
        )

    def ortho_loss(self) -> Tensor:
        loss = sum(module.loss() for module in self.modules() if isinstance(module, OLP))
        return cast(Tensor, loss)

    def energy(self, input_tensor: Tensor) -> Tuple[Tensor, ...]:
        return self.g_a.forward_energy(input_tensor)

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "GLIC":
        N = state_dict["latent_codec.z.entropy_bottleneck.quantiles"].size(0)
        M = state_dict["g_a.down3.weight"].size(0)
        groups = []
        index = 0
        while True:
            key = (
                f"latent_codec.y.latent_codec.y{index}.context_prediction.weight"
            )
            if key not in state_dict:
                break
            groups.append(state_dict[key].size(1))
            index += 1
        net = cls(N=N, M=M, groups=groups or None)
        net.load_state_dict(state_dict)
        return net
