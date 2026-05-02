from __future__ import annotations

import math

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import DropPath
from torch import Tensor

from ..attn.swin_attention import pad_to_window_multiple
from ..layers import conv, conv1x1, conv3x3, deconv
from .blocks import ResidualBottleneckBlock

__all__ = [
    "ConvolutionalGLU",
    "ConvWithDW",
    "DenseBlock",
    "DWConv",
    "MultiScaleAggregation",
    "MutiScaleDictionaryCrossAttentionGLU",
    "ResidualBottleneckBlockWithStride",
    "ResidualBottleneckBlockWithUpsample",
    "ResScaleConvolutionGateBlock",
    "Scale",
    "SpatialAttentionModule",
    "SwinBlockWithConvMulti",
]


class ResidualBottleneckBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = conv(in_ch, out_ch, kernel_size=5, stride=2)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.conv(input_tensor)
        output = self.res1(output)
        output = self.res2(output)
        return self.res3(output)


class ResidualBottleneckBlockWithUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = deconv(in_ch, out_ch, kernel_size=5, stride=2)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.res1(input_tensor)
        output = self.res2(output)
        output = self.res3(output)
        return self.conv(output)


class WMSA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        type: str,
    ) -> None:
        super().__init__()
        if type not in {"W", "SW"}:
            raise ValueError(f"Unsupported attention type: {type}")
        if input_dim % head_dim != 0:
            raise ValueError("input_dim must be divisible by head_dim")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        relative_position = torch.zeros(
            self.n_heads,
            2 * window_size - 1,
            2 * window_size - 1,
        )
        nn.init.trunc_normal_(relative_position, std=0.02)
        self.relative_position_params = nn.Parameter(relative_position)
        self.linear = nn.Linear(input_dim, output_dim)

    def generate_mask(
        self,
        height_windows: int,
        width_windows: int,
        window_size: int,
        shift: int,
    ) -> Tensor:
        attention_mask = torch.zeros(
            height_windows,
            width_windows,
            window_size,
            window_size,
            window_size,
            window_size,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attention_mask

        split = window_size - shift
        attention_mask[-1, :, :split, :, split:, :] = True
        attention_mask[-1, :, split:, :, :split, :] = True
        attention_mask[:, -1, :, :split, :, split:] = True
        attention_mask[:, -1, :, split:, :, :split] = True
        return rearrange(
            attention_mask,
            "h w p1 p2 p3 p4 -> 1 1 (h w) (p1 p2) (p3 p4)",
        )

    def relative_embedding(self) -> Tensor:
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(self.window_size, device=self.relative_position_params.device),
                torch.arange(self.window_size, device=self.relative_position_params.device),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        relation = coords[:, None, :] - coords[None, :, :] + self.window_size - 1
        return self.relative_position_params[
            :,
            relation[:, :, 0].long(),
            relation[:, :, 1].long(),
        ]

    def forward(self, input_tensor: Tensor) -> Tensor:
        if self.type != "W":
            input_tensor = torch.roll(
                input_tensor,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )

        output = rearrange(
            input_tensor,
            "b (h p1) (w p2) c -> b h w p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        height_windows = output.size(1)
        width_windows = output.size(2)
        output = rearrange(
            output,
            "b h w p1 p2 c -> b (h w) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )

        qkv = self.embedding_layer(output)
        qkv = rearrange(
            qkv,
            "b nw np (three heads dim) -> three b heads nw np dim",
            three=3,
            heads=self.n_heads,
            dim=self.head_dim,
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        similarity = torch.einsum("bhwnc,bhwmc->bhwnm", query, key) * self.scale
        similarity = similarity + rearrange(self.relative_embedding(), "h p q -> 1 h 1 p q")
        if self.type != "W":
            attention_mask = self.generate_mask(
                height_windows,
                width_windows,
                self.window_size,
                shift=self.window_size // 2,
            )
            similarity = similarity.masked_fill(attention_mask, float("-inf"))

        probabilities = similarity.softmax(dim=-1)
        output = torch.einsum("bhwij,bhwjc->bhwic", probabilities, value)
        output = rearrange(output, "b h w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (h w) (p1 p2) c -> b (h p1) (w p2) c",
            h=height_windows,
            p1=self.window_size,
            p2=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output


class DWConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=dim,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = rearrange(input_tensor, "b h w c -> b c h w")
        output = self.dwconv(output)
        return rearrange(output, "b c h w -> b h w c")


class ConvolutionalGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = (hidden_features or in_features) // 2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output, gate = self.fc1(input_tensor).chunk(2, dim=-1)
        output = self.act(self.dwconv(output)) * gate
        return self.fc2(output)


class Scale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1.0, trainable: bool = True) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(dim),
            requires_grad=trainable,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * self.scale


class ResScaleConvolutionGateBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        type: str = "W",
        input_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        del output_dim, input_resolution
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, type)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = ConvolutionalGLU(input_dim, input_dim * 4)
        self.res_scale_1 = Scale(input_dim, init_value=1.0)
        self.res_scale_2 = Scale(input_dim, init_value=1.0)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.res_scale_1(input_tensor) + self.drop_path(self.msa(self.ln1(input_tensor)))
        return self.res_scale_2(output) + self.drop_path(self.mlp(self.ln2(output)))


class SwinBlockWithConvMulti(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        block: type[nn.Module] = ResScaleConvolutionGateBlock,
        block_num: int = 2,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        self.layers = nn.ModuleList(
            block(
                input_dim,
                input_dim,
                head_dim,
                window_size,
                drop_path,
                type="W" if index % 2 == 0 else "SW",
            )
            for index in range(block_num)
        )
        self.block_num = block_num
        self.conv = conv(input_dim, output_dim, 3, 1)
        self.window_size = window_size

    def forward(self, input_tensor: Tensor) -> Tensor:
        output, pad_height, pad_width = pad_to_window_multiple(input_tensor, self.window_size)
        output = rearrange(output, "b c h w -> b h w c")
        for layer in self.layers:
            output = layer(output)
        output = rearrange(output, "b h w c -> b c h w")
        output = self.conv(output) + F.pad(input_tensor, (0, pad_width, 0, pad_height))
        if pad_height > 0 or pad_width > 0:
            output = output[:, :, : input_tensor.size(2), : input_tensor.size(3)]
        return output.contiguous()


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: Tensor) -> Tensor:
        average = input_tensor.mean(dim=1, keepdim=True)
        maximum, _ = input_tensor.max(dim=1, keepdim=True)
        output = torch.cat([average, maximum], dim=1)
        return self.sigmoid(self.conv1(output))


class ConvWithDW(nn.Module):
    def __init__(self, input_dim: int = 320, output_dim: int = 320) -> None:
        super().__init__()
        self.in_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=True)
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(
            output_dim,
            output_dim,
            kernel_size=3,
            padding=1,
            groups=output_dim,
            bias=True,
        )
        self.act2 = nn.GELU()
        self.out_trans = nn.Conv2d(output_dim, output_dim, kernel_size=1, bias=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.in_trans(input_tensor)
        output = self.act1(output)
        output = self.dw_conv(output)
        output = self.act2(output)
        return self.out_trans(output)


class DenseBlock(nn.Module):
    def __init__(self, dim: int = 320, layer_num: int = 3) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.conv_layers = nn.ModuleList(
            nn.Sequential(nn.GELU(), ConvWithDW(dim, dim))
            for _ in range(layer_num)
        )
        self.proj = nn.Conv2d(dim * (layer_num + 1), dim, kernel_size=1, bias=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        outputs = [input_tensor]
        for layer in self.conv_layers:
            outputs.append(layer(outputs[-1]))
        return self.proj(torch.cat(outputs, dim=1))


class MultiScaleAggregation(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.s = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = DenseBlock(dim)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = rearrange(input_tensor, "b h w c -> b c h w")
        output = self.s(output)
        output = self.dense(output)
        output = output * self.spatial_atte(output)
        return rearrange(output, "b c h w -> b h w c")


class MutiScaleDictionaryCrossAttentionGLU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_rate: int = 4,
        head_num: int = 20,
        qkv_bias: bool = True,
        dictionary_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        dict_dim = dictionary_dim or 32 * head_num
        if dict_dim % head_num != 0:
            raise ValueError("dictionary_dim must be divisible by head_num")

        self.head_num = head_num
        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)
        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)
        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)
        self.mlp = ConvolutionalGLU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Sequential(nn.Linear(dict_dim, output_dim))
        self.softmax = nn.Softmax(dim=-1)
        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

    def forward(self, input_tensor: Tensor, dictionary: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.size()
        output = rearrange(input_tensor, "b c h w -> b h w c")
        output = self.x_trans(output)
        output = self.msa(self.ln_scale(output)) + self.res_scale_1(output)

        shortcut = output
        output = rearrange(self.q_trans(self.lnx(output)), "b h w c -> b c h w")
        query = rearrange(output, "b (e c) h w -> b e (h w) c", e=self.head_num)

        dictionary = self.dict_ln(dictionary)
        key = rearrange(self.k(dictionary), "b n (e c) -> b e n c", e=self.head_num)
        dictionary_value = rearrange(dictionary, "b n (e c) -> b e n c", e=self.head_num)

        scale = self.scale.to(device=query.device, dtype=query.dtype)
        similarity = torch.einsum("benc,bedc->bend", query, key) * scale
        probabilities = self.softmax(similarity)
        output = torch.einsum("bend,bedc->benc", probabilities, dictionary_value)
        output = rearrange(output, "b e (h w) c -> b h w (e c)", h=height, w=width)
        output = self.linear(output) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)
        output = self.output_trans(output)
        return rearrange(output, "b h w c -> b c h w", b=batch_size)
