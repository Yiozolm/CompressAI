from __future__ import annotations

from typing import Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F

from torch import Tensor

__all__ = [
    "compute_sobel_gradients",
    "cosine_similarity",
    "cossim",
    "gaussian_blur",
    "global_sampling",
    "local_sampling",
]


def cosine_similarity(
    query: Tensor,
    key: Tensor,
    graph: Optional[Tensor] = None,
) -> Tensor:
    similarity = torch.einsum(
        "a b m c, a b n c -> a b m n",
        F.normalize(query, dim=-1),
        F.normalize(key, dim=-1),
    )
    if graph is not None:
        similarity = similarity + (-100.0) * (~graph)
    return similarity


def _to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return value


def local_sampling(
    input_tensor: Tensor,
    group_size: Union[int, Tuple[int, int]],
    unfold_dict: Optional[dict],
    output: int = 0,
    tensor_format: str = "bhwc",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    group_height, group_width = _to_2tuple(group_size)

    if output != 1:
        if tensor_format == "bhwc":
            grouped = einops.rearrange(
                input_tensor,
                "b (nh gh) (nw gw) c -> (b nh nw) (gh gw) c",
                gh=group_height,
                gw=group_width,
            )
        elif tensor_format == "bchw":
            grouped = einops.rearrange(
                input_tensor,
                "b c (nh gh) (nw gw) -> (b nh nw) (gh gw) c",
                gh=group_height,
                gw=group_width,
            )
        else:
            raise ValueError(f"Unsupported tensor format: {tensor_format}")

        if output == 0:
            return grouped

    if unfold_dict is None:
        raise ValueError("`unfold_dict` is required for local sampled output.")

    if tensor_format == "bhwc":
        input_tensor = einops.rearrange(input_tensor, "b h w c -> b c h w")
    elif tensor_format != "bchw":
        raise ValueError(f"Unsupported tensor format: {tensor_format}")

    kernel_height, kernel_width = unfold_dict["kernel_size"]
    sampled = einops.rearrange(
        F.unfold(input_tensor, **unfold_dict),
        "b (c kh kw) l -> (b l) (kh kw) c",
        kh=kernel_height,
        kw=kernel_width,
    )

    if output == 1:
        return sampled

    if grouped.size(0) != sampled.size(0):
        raise ValueError("Grouped and sampled tensors have incompatible windows.")
    return grouped, sampled


def global_sampling(
    input_tensor: Tensor,
    group_size: Union[int, Tuple[int, int]],
    sample_size: Union[int, Tuple[int, int]],
    output: int = 0,
    tensor_format: str = "bhwc",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    group_height, group_width = _to_2tuple(group_size)
    sample_height, sample_width = _to_2tuple(sample_size)

    if output != 1:
        if tensor_format == "bchw":
            grouped = einops.rearrange(
                input_tensor,
                "b c (gh nh) (gw nw) -> (b nh nw) (gh gw) c",
                gh=group_height,
                gw=group_width,
            )
        elif tensor_format == "bhwc":
            grouped = einops.rearrange(
                input_tensor,
                "b (gh nh) (gw nw) c -> (b nh nw) (gh gw) c",
                gh=group_height,
                gw=group_width,
            )
        else:
            raise ValueError(f"Unsupported tensor format: {tensor_format}")

        if output == 0:
            return grouped

    if tensor_format == "bchw":
        sampled = einops.rearrange(
            input_tensor,
            "b c (sh eh nh) (sw ew nw) -> b eh nh ew nw c sh sw",
            sh=sample_height,
            sw=sample_width,
            eh=1,
            ew=1,
        )
    elif tensor_format == "bhwc":
        sampled = einops.rearrange(
            input_tensor,
            "b (sh eh nh) (sw ew nw) c -> b eh nh ew nw c sh sw",
            sh=sample_height,
            sw=sample_width,
            eh=1,
            ew=1,
        )
    else:
        raise ValueError(f"Unsupported tensor format: {tensor_format}")

    batch_size, _, num_h, _, num_w, channels, sample_h, sample_w = sampled.shape
    ratio_h = sample_height // group_height
    ratio_w = sample_width // group_width
    sampled = sampled.expand(
        batch_size,
        ratio_h,
        num_h,
        ratio_w,
        num_w,
        channels,
        sample_h,
        sample_w,
    )
    sampled = sampled.reshape(-1, channels, sample_h * sample_w).permute(0, 2, 1)

    if output == 1:
        return sampled

    if grouped.size(0) != sampled.size(0):
        raise ValueError("Grouped and sampled tensors have incompatible windows.")
    return grouped, sampled


def gaussian_blur(
    input_tensor: Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
    mode: str = "replicate",
) -> Tensor:
    squeeze_channel = input_tensor.dim() == 3
    if squeeze_channel:
        input_tensor = input_tensor.unsqueeze(1)

    _, channels, _, _ = input_tensor.shape
    coords = torch.arange(
        kernel_size,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    coords = coords - (kernel_size - 1) / 2
    gaussian = torch.exp(-0.5 * (coords / sigma) ** 2)
    gaussian = gaussian / gaussian.sum()

    kernel_x = gaussian.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    kernel_y = gaussian.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
    padding = kernel_size // 2
    output = F.pad(input_tensor, (padding, padding, 0, 0), mode=mode)
    output = F.conv2d(output, kernel_x, groups=channels)
    output = F.pad(output, (0, 0, padding, padding), mode=mode)
    output = F.conv2d(output, kernel_y, groups=channels)
    return output.squeeze(1) if squeeze_channel else output


def compute_sobel_gradients(
    input_tensor: Tensor,
    shape: Tuple[int, int],
    num_heads: int = 1,
) -> Tensor:
    batch_size, _, channels = input_tensor.shape
    height, width = shape
    output = input_tensor.view(
        batch_size,
        height,
        width,
        channels // num_heads,
        num_heads,
    )
    output = output.mean(-1).permute(0, 3, 1, 2)

    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    ).view(1, 1, 3, 3)
    sobel_x = sobel_x / 4
    sobel_y = sobel_y / 4

    gradient_channels = output.size(1)
    kernel_x = sobel_x.repeat(gradient_channels, 1, 1, 1)
    kernel_y = sobel_y.repeat(gradient_channels, 1, 1, 1)
    padded = F.pad(output, (1, 1, 1, 1), mode="replicate")
    gradient_x = F.conv2d(padded, kernel_x, groups=gradient_channels)
    gradient_y = F.conv2d(padded, kernel_y, groups=gradient_channels)

    gradient = torch.sqrt(gradient_x.pow(2) + gradient_y.pow(2) + 1e-9)
    gradient = gaussian_blur(gradient, kernel_size=3, sigma=0.8)
    return torch.sqrt(gradient.pow(2).mean(1))


cossim = cosine_similarity
