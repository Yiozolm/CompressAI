import importlib
import importlib.util

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

__all__ = [
    "cross_merge",
    "cross_scan",
    "cross_selective_scan",
    "get_selective_scan_backend",
    "is_mamba_ssm_available",
    "is_selective_scan_cuda_available",
    "selective_scan",
    "selective_scan_ref",
]


def is_mamba_ssm_available() -> bool:
    return importlib.util.find_spec("mamba_ssm") is not None


def is_selective_scan_cuda_available() -> bool:
    return any(
        importlib.util.find_spec(name) is not None
        for name in (
            "selective_scan_cuda_oflex",
            "selective_scan_cuda_core",
            "selective_scan_cuda",
        )
    )


def get_selective_scan_backend() -> str:
    if is_selective_scan_cuda_available():
        return "selective_scan_cuda"
    if is_mamba_ssm_available():
        return "mamba_ssm"
    return "torch"


def _has_grad_inputs(*inputs: Optional[Tensor]) -> bool:
    return any(input is not None and input.requires_grad for input in inputs)


def _expand_group_param(param: Tensor, channels: int) -> Tensor:
    if param.dim() == 4:
        batch, groups, states, length = param.shape
        if channels % groups != 0:
            raise ValueError("Channels must be divisible by parameter groups")
        repeat = channels // groups
        return param[:, :, None].expand(batch, groups, repeat, states, length)
    if param.dim() == 3:
        batch, states, length = param.shape
        return param[:, None, None].expand(batch, 1, channels, states, length)
    if param.dim() == 2:
        states = param.size(1)
        return param[None, None, :, :, None].expand(1, 1, channels, states, 1)
    raise ValueError("Expected B/C parameter with 2, 3, or 4 dimensions")


def selective_scan_ref(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    delta_bias: Optional[Tensor] = None,
    delta_softplus: bool = True,
) -> Tensor:
    batch, channels, length = u.shape
    states = A.size(1)
    groups = B.size(1) if B.dim() == 4 else 1
    if channels % groups != 0:
        raise ValueError("Channels must be divisible by selective-scan groups")

    width = channels // groups
    dtype = u.dtype
    fp_types = (torch.float16, torch.bfloat16)
    scan_dtype = torch.float32 if u.dtype in fp_types else u.dtype
    u_grouped = u.to(scan_dtype).view(batch, groups, width, length)
    delta_grouped = delta.to(scan_dtype).view(batch, groups, width, length)
    A_grouped = A.to(scan_dtype).view(groups, width, states)
    B_grouped = _expand_group_param(B.to(scan_dtype), channels)
    C_grouped = _expand_group_param(C.to(scan_dtype), channels)
    if B_grouped.size(0) == 1:
        B_grouped = B_grouped.expand(batch, -1, -1, -1, -1)
    if C_grouped.size(0) == 1:
        C_grouped = C_grouped.expand(batch, -1, -1, -1, -1)

    if delta_bias is not None:
        delta_bias = delta_bias.to(scan_dtype).view(groups, width)
    if D is not None:
        D = D.to(scan_dtype).view(groups, width)

    state = u.new_zeros((batch, groups, width, states), dtype=scan_dtype)
    outputs = []
    for index in range(length):
        delta_t = delta_grouped[..., index]
        if delta_bias is not None:
            delta_t = delta_t + delta_bias[None, :, :]
        if delta_softplus:
            delta_t = F.softplus(delta_t)

        delta_a = torch.exp(delta_t[..., None] * A_grouped[None, :, :, :])
        delta_bu = (
            delta_t[..., None] * B_grouped[..., index] * u_grouped[..., index, None]
        )
        state = state * delta_a + delta_bu
        output_t = (state * C_grouped[..., index]).sum(dim=-1)
        if D is not None:
            output_t = output_t + u_grouped[..., index] * D[None, :, :]
        outputs.append(output_t)

    return torch.stack(outputs, dim=-1).view(batch, channels, length).to(dtype)


def _selective_scan_mamba(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor],
    delta_bias: Optional[Tensor],
    delta_softplus: bool,
) -> Tensor:
    module = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    selective_scan_fn = getattr(module, "selective_scan_fn")
    return selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D,
        z=None,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=False,
    )


def _selective_scan_cuda_forward(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor],
    delta_bias: Optional[Tensor],
    delta_softplus: bool,
) -> Tensor:
    for name in (
        "selective_scan_cuda_oflex",
        "selective_scan_cuda_core",
        "selective_scan_cuda",
    ):
        if importlib.util.find_spec(name) is None:
            continue
        module = importlib.import_module(name)
        if name == "selective_scan_cuda_oflex":
            return module.fwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                delta_bias,
                delta_softplus,
                1,
                True,
            )[0]
        if name == "selective_scan_cuda_core":
            return module.fwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                delta_bias,
                delta_softplus,
                1,
            )[0]
        return module.fwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            None,
            delta_bias,
            delta_softplus,
        )[0]
    raise RuntimeError("No selective_scan_cuda backend is available")


def selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    delta_bias: Optional[Tensor] = None,
    delta_softplus: bool = True,
    backend: str = "auto",
) -> Tensor:
    if backend not in ("auto", "cuda", "mamba_ssm", "torch"):
        raise ValueError(f"Unknown selective-scan backend: {backend}")

    grad_inputs = _has_grad_inputs(u, delta, A, B, C, D, delta_bias)
    if backend in ("auto", "cuda") and not grad_inputs:
        if is_selective_scan_cuda_available():
            return _selective_scan_cuda_forward(
                u,
                delta,
                A,
                B,
                C,
                D,
                delta_bias,
                delta_softplus,
            )
    if backend in ("auto", "mamba_ssm") and is_mamba_ssm_available():
        return _selective_scan_mamba(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            delta_softplus,
        )
    return selective_scan_ref(u, delta, A, B, C, D, delta_bias, delta_softplus)


def cross_scan(inputs: Tensor) -> Tensor:
    batch, channels, height, width = inputs.shape
    horizontal = inputs.flatten(2, 3)
    vertical = inputs.transpose(2, 3).contiguous().flatten(2, 3)
    return torch.stack(
        [horizontal, vertical, horizontal.flip(-1), vertical.flip(-1)],
        dim=1,
    ).view(batch, 4, channels, height * width)


def cross_merge(inputs: Tensor) -> Tensor:
    batch, groups, channels, height, width = inputs.shape
    if groups != 4:
        raise ValueError("cross_merge expects four scan directions")
    length = height * width
    inputs = inputs.view(batch, groups, channels, length)
    merged = inputs[:, 0:2] + inputs[:, 2:4].flip(-1)
    vertical = merged[:, 1].view(batch, channels, width, height)
    vertical = vertical.transpose(2, 3).contiguous().view(batch, channels, length)
    return merged[:, 0] + vertical


def cross_selective_scan(
    x: Tensor,
    x_proj_weight: Tensor,
    x_proj_bias: Optional[Tensor],
    dt_projs_weight: Tensor,
    dt_projs_bias: Tensor,
    A_logs: Tensor,
    Ds: Tensor,
    delta_softplus: bool = True,
    out_norm: Optional[nn.Module] = None,
    out_norm_shape: str = "v0",
    to_dtype: bool = True,
    force_fp32: bool = False,
    scan_backend: str = "auto",
) -> Tensor:
    batch, channels, height, width = x.shape
    groups, _, rank = dt_projs_weight.shape
    states = A_logs.size(1)
    length = height * width
    xs = cross_scan(x)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, groups, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [rank, states, states], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(batch, -1, length)
    dts = dts.contiguous().view(batch, -1, length)
    Bs = Bs.contiguous().view(batch, groups, states, length)
    Cs = Cs.contiguous().view(batch, groups, states, length)
    As = -torch.exp(A_logs.float())
    Ds = Ds.float()
    delta_bias = dt_projs_bias.float().view(-1)
    if force_fp32:
        xs, dts, Bs, Cs = xs.float(), dts.float(), Bs.float(), Cs.float()

    ys = selective_scan(
        xs,
        dts,
        As,
        Bs,
        Cs,
        Ds,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        backend=scan_backend,
    ).view(batch, groups, channels, height, width)
    y = cross_merge(ys)
    out_norm = out_norm or nn.Identity()
    if out_norm_shape == "v1":
        y = out_norm(y.view(batch, -1, height, width)).permute(0, 2, 3, 1)
    else:
        y = y.transpose(1, 2).contiguous()
        y = out_norm(y).view(batch, height, width, -1)
    return y.to(x.dtype) if to_dtype else y
