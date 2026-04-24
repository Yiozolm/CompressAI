from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from torch import Tensor

from ..layers import deconv, subpel_conv3x3
from ..wave.wavelet import WLS, iWLS
from .blocks import GatedTransformCNN
from .cmic_context import CMICChannelContextBlock, CMICSpatialContextBlock
from .cmic_stage import CMICStage

__all__ = [
    "CMICAnalysisTransform",
    "CMICChannelContextBlock",
    "CMICSpatialContextBlock",
    "CMICStage",
    "CMICSynthesisTransform",
]


class CMICAnalysisTransform(nn.Module):
    def __init__(
        self,
        M: int,
        stage_dims: Tuple[int, int, int] = (128, 192, 256),
        stage_depths: Tuple[int, int] = (2, 2),
        num_heads: Tuple[int, int] = (8, 8),
        d_state: int = 8,
        window_size: int = 8,
        inner_rank: int = 32,
        cluster_num: int = 64,
        stage_mlp_ratio: float = 3.0,
    ) -> None:
        super().__init__()
        embed_dim0, embed_dim1, embed_dim2 = stage_dims
        depth1, depth2 = stage_depths
        heads1, heads2 = num_heads

        self.AuxT_enc = nn.Sequential(
            WLS(3, embed_dim0),
            WLS(embed_dim0, embed_dim1),
            WLS(embed_dim1, embed_dim2),
            WLS(embed_dim2, M),
        )
        self.g1 = nn.Sequential(
            GatedTransformCNN(embed_dim0, embed_dim0, expansion_factor=stage_mlp_ratio),
            GatedTransformCNN(embed_dim0, embed_dim0, expansion_factor=stage_mlp_ratio),
            GatedTransformCNN(embed_dim0, embed_dim0, expansion_factor=stage_mlp_ratio),
        )
        self.g2 = CMICStage(
            dim=embed_dim1,
            d_state=d_state,
            depth=depth1,
            num_heads=heads1,
            window_size=window_size,
            inner_rank=inner_rank,
            cluster_num=cluster_num,
            mlp_ratio=stage_mlp_ratio,
        )
        self.g3 = CMICStage(
            dim=embed_dim2,
            d_state=d_state,
            depth=depth2,
            num_heads=heads2,
            window_size=window_size,
            inner_rank=inner_rank,
            cluster_num=cluster_num,
            mlp_ratio=stage_mlp_ratio,
        )
        self.down0 = nn.Conv2d(3, embed_dim0, kernel_size=3, stride=2, padding=1)
        self.down1 = nn.Conv2d(embed_dim0, embed_dim1, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(embed_dim1, embed_dim2, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(embed_dim2, M, kernel_size=3, stride=2, padding=1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        aux_output = input_tensor

        output = self.down0(input_tensor)
        output = self.g1(output)
        aux_output = self.AuxT_enc[0](aux_output)
        output = output + aux_output

        output = self.down1(output)
        output = self.g2(output, output.shape[-2:])
        aux_output = self.AuxT_enc[1](aux_output)
        output = output + aux_output

        output = self.down2(output)
        output = self.g3(output, output.shape[-2:])
        aux_output = self.AuxT_enc[2](aux_output)
        output = output + aux_output

        output = self.down3(output)
        aux_output = self.AuxT_enc[3](aux_output)
        return output + aux_output


class CMICSynthesisTransform(nn.Module):
    def __init__(
        self,
        M: int,
        stage_dims: Tuple[int, int, int] = (128, 192, 256),
        stage_depths: Tuple[int, int] = (2, 2),
        num_heads: Tuple[int, int] = (8, 8),
        d_state: int = 8,
        window_size: int = 8,
        inner_rank: int = 32,
        cluster_num: int = 64,
        stage_mlp_ratio: float = 3.0,
    ) -> None:
        super().__init__()
        embed_dim1, embed_dim2, embed_dim3 = stage_dims
        depth1, depth2 = stage_depths
        heads1, heads2 = num_heads

        self.AuxT_dec = nn.Sequential(
            iWLS(M, embed_dim3),
            iWLS(embed_dim3, embed_dim2),
            iWLS(embed_dim2, embed_dim1),
            iWLS(embed_dim1, 3),
        )
        self.g1 = CMICStage(
            dim=embed_dim3,
            d_state=d_state,
            depth=depth2,
            num_heads=heads2,
            window_size=window_size,
            inner_rank=inner_rank,
            cluster_num=cluster_num,
            mlp_ratio=stage_mlp_ratio,
        )
        self.g2 = CMICStage(
            dim=embed_dim2,
            d_state=d_state,
            depth=depth1,
            num_heads=heads1,
            window_size=window_size,
            inner_rank=inner_rank,
            cluster_num=cluster_num,
            mlp_ratio=stage_mlp_ratio,
        )
        self.g3 = nn.Sequential(
            GatedTransformCNN(embed_dim1, embed_dim1, expansion_factor=stage_mlp_ratio),
            GatedTransformCNN(embed_dim1, embed_dim1, expansion_factor=stage_mlp_ratio),
            GatedTransformCNN(embed_dim1, embed_dim1, expansion_factor=stage_mlp_ratio),
        )
        self.up0 = deconv(M, embed_dim3, kernel_size=3)
        self.up1 = deconv(embed_dim3, embed_dim2, kernel_size=3)
        self.up2 = deconv(embed_dim2, embed_dim1, kernel_size=3)
        self.up3 = subpel_conv3x3(embed_dim1, 3, 2)

    def forward(self, input_tensor: Tensor) -> Tensor:
        aux_output = input_tensor

        output = self.up0(input_tensor)
        output = self.g1(output, output.shape[-2:])
        aux_output = self.AuxT_dec[0](aux_output)
        output = output + aux_output

        output = self.up1(output)
        output = self.g2(output, output.shape[-2:])
        aux_output = self.AuxT_dec[1](aux_output)
        output = output + aux_output

        output = self.up2(output)
        output = self.g3(output)
        aux_output = self.AuxT_dec[2](aux_output)
        output = output + aux_output

        output = self.up3(output)
        aux_output = self.AuxT_dec[3](aux_output)
        return output + aux_output
