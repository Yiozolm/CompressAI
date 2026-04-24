from __future__ import annotations

import math

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from ..attn.swin_attention import (
    WindowAttention,
    _pad_to_window_size,
    window_partition,
    window_reverse,
)
from ..ssm import selective_scan
from .cmic_context import _apply_permutation, _batched_bincount, _inverse_permutation

__all__ = ["CMICStage"]


class _ConvTokenGatedFFN(nn.Module):
    def __init__(self, channels: int, expansion_factor: float) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.channels = channels
        self.project_in = nn.Conv2d(
            channels,
            hidden_channels * 2,
            kernel_size=1,
            bias=False,
        )
        self.depthwise = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        height, width = x_size
        output = input_tensor.transpose(1, 2).reshape(
            input_tensor.size(0),
            self.channels,
            height,
            width,
        )
        gate_tensor, value_tensor = self.depthwise(self.project_in(output)).chunk(
            2,
            dim=1,
        )
        output = self.project_out(F.gelu(gate_tensor) * value_tensor)
        return output.flatten(2).transpose(1, 2).contiguous()


class _ContentAwareMamba(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        cluster_num: int = 64,
        inner_rank: int = 128,
        mlp_ratio: float = 2.0,
        n_iter: int = 5,
        ema_decay: float = 0.999,
    ) -> None:
        super().__init__()
        del inner_rank
        self.dim = dim
        self.d_state = d_state
        self.cluster_num = cluster_num
        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.hidden_dim = int(dim * mlp_ratio)
        self.dt_rank = math.ceil(self.hidden_dim / 16)

        self.in_proj = nn.Conv2d(dim, self.hidden_dim, kernel_size=1)
        self.cpe = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            groups=self.hidden_dim,
        )
        self.prompt_proj = nn.Linear(dim, d_state)
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, dim)

        x_proj = nn.Linear(self.hidden_dim, self.dt_rank + d_state * 2, bias=False)
        self.x_proj_weight = nn.Parameter(x_proj.weight.unsqueeze(0))
        dt_proj = self._make_dt_proj()
        self.dt_projs_weight = nn.Parameter(dt_proj.weight.unsqueeze(0))
        self.dt_projs_bias = nn.Parameter(dt_proj.bias.unsqueeze(0))
        self.A_logs = self._make_a_logs()
        self.Ds = self._make_ds()

        self.register_buffer("means", torch.randn(cluster_num, dim))
        self.register_buffer("initted", torch.tensor(False))

    def _make_dt_proj(self) -> nn.Linear:
        layer = nn.Linear(self.dt_rank, self.hidden_dim, bias=True)
        std = self.dt_rank**-0.5
        nn.init.uniform_(layer.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.hidden_dim) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            layer.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        return layer

    def _make_a_logs(self) -> nn.Parameter:
        values = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        values = values.repeat(self.hidden_dim, 1)
        param = nn.Parameter(torch.log(values))
        param._no_weight_decay = True
        return param

    def _make_ds(self) -> nn.Parameter:
        param = nn.Parameter(torch.ones(self.hidden_dim))
        param._no_weight_decay = True
        return param

    def _center_iter(self, tokens: Tensor, means: Tensor) -> Tensor:
        scores = torch.einsum(
            "bnc,kc->bnk",
            F.normalize(tokens, dim=-1),
            F.normalize(means, dim=-1),
        )
        buckets = scores.argmax(dim=-1)
        counts = _batched_bincount(buckets, self.cluster_num, dtype=tokens.dtype)
        zero_mask = counts.sum(dim=0) == 0

        sums = torch.zeros(
            tokens.size(0),
            self.cluster_num,
            tokens.size(-1),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        expanded_index = buckets.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        sums.scatter_add_(1, expanded_index, tokens)
        means_new = sums.sum(dim=0) / counts.sum(dim=0).clamp(min=1).unsqueeze(-1)
        means_new = F.normalize(means_new, dim=-1).type_as(tokens)
        return torch.where(zero_mask.unsqueeze(-1), means, means_new)

    def _update_centroids(self, tokens: Tensor) -> Tensor:
        tokens = tokens.contiguous()
        if not bool(self.initted.item()):
            pad_tokens = (self.cluster_num - tokens.size(1) % self.cluster_num) % self.cluster_num
            if pad_tokens > 0:
                tokens = F.pad(tokens, (0, 0, 0, pad_tokens))
            grouped = tokens.view(tokens.size(0), self.cluster_num, -1, tokens.size(-1))
            means = grouped.permute(1, 0, 2, 3).reshape(self.cluster_num, -1, self.dim)
            centroids = means.mean(dim=1).detach()
        else:
            centroids = self.means.detach()

        if not self.training:
            return centroids

        with torch.no_grad():
            updated = centroids
            for _ in range(max(self.n_iter - 1, 0)):
                updated = self._center_iter(tokens, updated)
            if not bool(self.initted.item()):
                self.means.copy_(updated)
                self.initted.copy_(torch.tensor(True, device=self.initted.device))
            else:
                self.means.mul_(self.ema_decay).add_(updated, alpha=1 - self.ema_decay)
        return updated

    def _forward_scan(self, input_tensor: Tensor, prompt: Tensor) -> Tensor:
        batch_size, length, _ = input_tensor.shape
        xs = input_tensor.transpose(1, 2).reshape(batch_size, 1, self.hidden_dim, length)
        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, b_param, c_param = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)

        output = selective_scan(
            xs.float().view(batch_size, -1, length),
            dts.float().view(batch_size, -1, length),
            -torch.exp(self.A_logs.float()).view(-1, self.d_state),
            b_param.float().view(batch_size, 1, -1, length),
            c_param.float().view(batch_size, 1, -1, length) + prompt,
            self.Ds.float().view(-1),
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
        )
        return output.view(batch_size, self.hidden_dim, length).transpose(1, 2)

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        height, width = x_size
        batch_size, _, channels = input_tensor.shape
        tokens = input_tensor
        centroids = self._update_centroids(tokens.detach())
        scores = torch.einsum(
            "bnc,kc->bnk",
            F.normalize(tokens.detach(), dim=-1),
            F.normalize(centroids, dim=-1),
        )
        cluster_index = scores.argmax(dim=-1)
        prompt = torch.matmul(
            F.one_hot(cluster_index, num_classes=self.cluster_num).float(),
            self.prompt_proj(centroids),
        )

        sort_index = torch.sort(cluster_index, dim=-1, stable=True).indices
        inverse_index = _inverse_permutation(sort_index)

        output = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        output = self.in_proj(output)
        output = output * torch.sigmoid(self.cpe(output))
        output = output.flatten(2).transpose(1, 2).contiguous()
        output = _apply_permutation(output, sort_index)
        output = self._forward_scan(output, prompt.transpose(1, 2).unsqueeze(1))
        output = self.out_proj(self.out_norm(output))
        return _apply_permutation(output, inverse_index)


class _CMICBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        num_heads: int,
        window_size: int,
        inner_rank: int,
        cluster_num: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.window_attention = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
        )
        self.feed_forward1 = _ConvTokenGatedFFN(dim, expansion_factor=mlp_ratio)
        self.content_model = _ContentAwareMamba(
            dim=dim,
            d_state=d_state,
            cluster_num=cluster_num,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
        )
        self.feed_forward2 = _ConvTokenGatedFFN(dim, expansion_factor=mlp_ratio)

    def _window_forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        height, width = x_size
        output = input_tensor.reshape(input_tensor.size(0), height, width, self.dim)
        output, pad_height, pad_width = _pad_to_window_size(output, self.window_size)
        padded_height, padded_width = output.shape[1], output.shape[2]
        windows = window_partition(output, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, self.dim)
        windows = self.window_attention(windows)
        windows = windows.view(-1, self.window_size, self.window_size, self.dim)
        output = window_reverse(
            windows,
            self.window_size,
            padded_height,
            padded_width,
        )
        if pad_height > 0 or pad_width > 0:
            output = output[:, :height, :width, :].contiguous()
        return output.reshape(input_tensor.size(0), height * width, self.dim)

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        output = input_tensor + self._window_forward(self.norm1(input_tensor), x_size)
        output = output + self.feed_forward1(self.norm2(output), x_size)
        output = output + self.content_model(self.norm3(output), x_size)
        return output + self.feed_forward2(self.norm4(output), x_size)


class CMICStage(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        depth: int,
        num_heads: int,
        window_size: int,
        inner_rank: int,
        cluster_num: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _CMICBlock(
                    dim=dim,
                    d_state=d_state,
                    num_heads=num_heads,
                    window_size=window_size,
                    inner_rank=inner_rank,
                    cluster_num=cluster_num,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, input_tensor: Tensor, x_size: Tuple[int, int]) -> Tensor:
        output = input_tensor.flatten(2).transpose(1, 2).contiguous()
        for block in self.blocks:
            output = block(output, x_size)
        height, width = x_size
        return output.transpose(1, 2).reshape(input_tensor.size(0), -1, height, width)
