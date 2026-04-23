import math
from functools import partial
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from timm.layers import DropPath

from .ssm_ops import (
    cross_merge,
    cross_scan,
    cross_selective_scan,
    get_selective_scan_backend,
    is_mamba_ssm_available,
    is_selective_scan_cuda_available,
    selective_scan,
    selective_scan_ref,
)


__all__ = [
    "SS2D",
    "VSSBlock",
    "cross_merge",
    "cross_scan",
    "cross_selective_scan",
    "get_selective_scan_backend",
    "is_mamba_ssm_available",
    "is_selective_scan_cuda_available",
    "selective_scan",
    "selective_scan_ref",
]


class SS2D(nn.Module):
    def __init__(
        self,
        d_model: int = 96,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: Any = "auto",
        act_layer: Callable[..., nn.Module] = nn.SiLU,
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        initialize: str = "v0",
        forward_type: str = "v2",
        scan_backend: str = "auto",
    ) -> None:
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_conv = int(d_conv)
        self.scan_backend = scan_backend
        self.disable_force32, forward_type = self._strip_suffix(forward_type, "no32")
        self.disable_z, forward_type = self._strip_suffix(forward_type, "noz")
        self.disable_z_act, forward_type = self._strip_suffix(forward_type, "nozact")

        self.out_norm_shape = "v0"
        self.out_norm, forward_type = self._build_out_norm(forward_type, d_inner)
        output_channels = d_inner if self.disable_z else d_inner * 2
        self.in_proj = nn.Linear(d_model, output_channels, bias=bias)
        self.act = act_layer()
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                d_inner,
                d_inner,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                groups=d_inner,
                bias=conv_bias,
            )

        x_proj_layers = [
            nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([layer.weight for layer in x_proj_layers], dim=0)
        )
        self.dt_projs_weight, self.dt_projs_bias = self._init_dt_projs(
            dt_rank,
            d_inner,
            dt_scale,
            dt_init,
            dt_min,
            dt_max,
            dt_init_floor,
        )
        self.A_logs = self._a_log_init(d_state, d_inner, copies=4)
        self.Ds = self._d_init(d_inner, copies=4)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if forward_type not in ("v1", "v2", "v3", "v4"):
            raise ValueError(f"Unsupported SS2D forward_type: {forward_type}")
        if initialize not in ("v0",):
            raise ValueError(f"Unsupported SS2D initialize mode: {initialize}")

    def _build_out_norm(
        self,
        forward_type: str,
        channels: int,
    ) -> Tuple[nn.Module, str]:
        if forward_type.endswith("none"):
            return nn.Identity(), forward_type[: -len("none")]
        if forward_type.endswith("dwconv3"):
            self.out_norm_shape = "v1"
            return (
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=1,
                    groups=channels,
                    bias=False,
                ),
                forward_type[: -len("dwconv3")],
            )
        if forward_type.endswith("softmax"):
            return nn.Softmax(dim=1), forward_type[: -len("softmax")]
        if forward_type.endswith("sigmoid"):
            return nn.Sigmoid(), forward_type[: -len("sigmoid")]
        return nn.LayerNorm(channels), forward_type

    @staticmethod
    def _strip_suffix(value: str, suffix: str) -> Tuple[bool, str]:
        if value.endswith(suffix):
            return True, value[: -len(suffix)]
        return False, value

    @staticmethod
    def _init_dt_projs(
        dt_rank: int,
        d_inner: int,
        dt_scale: float,
        dt_init: str,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
    ) -> Tuple[nn.Parameter, nn.Parameter]:
        weights = []
        biases = []
        for _ in range(4):
            layer = nn.Linear(dt_rank, d_inner, bias=True)
            std = dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(layer.weight, std)
            elif dt_init == "random":
                nn.init.uniform_(layer.weight, -std, std)
            else:
                raise ValueError(f"Unsupported dt_init: {dt_init}")
            scale = math.log(dt_max) - math.log(dt_min)
            dt = torch.exp(torch.rand(d_inner) * scale + math.log(dt_min))
            dt = dt.clamp(min=dt_init_floor)
            with torch.no_grad():
                layer.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            weights.append(layer.weight)
            biases.append(layer.bias)
        return (
            nn.Parameter(torch.stack(weights, dim=0)),
            nn.Parameter(torch.stack(biases, dim=0)),
        )

    @staticmethod
    def _a_log_init(d_state: int, d_inner: int, copies: int) -> nn.Parameter:
        values = torch.arange(1, d_state + 1, dtype=torch.float32)
        values = values.repeat(d_inner, 1)
        values = torch.log(values).unsqueeze(0).repeat(copies, 1, 1).flatten(0, 1)
        param = nn.Parameter(values)
        param._no_weight_decay = True
        return param

    @staticmethod
    def _d_init(d_inner: int, copies: int) -> nn.Parameter:
        param = nn.Parameter(torch.ones(copies * d_inner))
        param._no_weight_decay = True
        return param

    def forward_core(self, x: Tensor) -> Tensor:
        return cross_selective_scan(
            x,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            out_norm=self.out_norm,
            out_norm_shape=self.out_norm_shape,
            force_fp32=not self.disable_force32,
            scan_backend=self.scan_backend,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        z = None
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1)
            if not self.disable_z_act:
                z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.d_conv > 1:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        if z is not None:
            y = y * z
        return self.dropout(self.out_proj(y))


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.post_norm = bool(post_norm)
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(d_model=hidden_dim, **kwargs)
        self.drop_path = DropPath(drop_path)

    def _forward(self, x: Tensor) -> Tensor:
        if self.post_norm:
            return x + self.drop_path(self.norm(self.op(x)))
        return x + self.drop_path(self.op(self.norm(x)))

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.use_checkpoint:
            from torch.utils import checkpoint

            x = checkpoint.checkpoint(self._forward, x)
        else:
            x = self._forward(x)
        return x.permute(0, 3, 1, 2).contiguous()
