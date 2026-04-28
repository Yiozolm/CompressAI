from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple, TypeVar

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.layers import AttentionBlock
from compressai.layers.lic import (
    CouplingLayer,
    EnhBlock,
    InvertibleConv1x1,
    SqueezeLayer,
    is_freia_available,
)
from compressai.registry import register_model

from .waseda import Cheng2020Anchor

__all__ = ["InvCompress"]

_FREIA_DUMMY_SPATIAL = 16
_ModelType = TypeVar("_ModelType", bound=type[nn.Module])


def _identity_decorator(cls: _ModelType) -> _ModelType:
    return cls


def _maybe_register_model(name: str) -> Callable[[_ModelType], _ModelType]:
    if is_freia_available():
        return register_model(name)
    return _identity_decorator


def _require_freia() -> None:
    if not is_freia_available():
        raise ModuleNotFoundError(
            "InvCompress requires the optional dependency `FrEIA`. "
            "Install `compressai[invcompress]` to enable this model."
        )


def _initialize_conv(
    conv: nn.Conv2d,
    *,
    scale: float,
    use_xavier: bool,
) -> None:
    if use_xavier:
        nn.init.xavier_normal_(conv.weight)
    else:
        nn.init.kaiming_normal_(conv.weight, a=0, mode="fan_in")
    conv.weight.data *= scale
    if conv.bias is not None:
        conv.bias.data.zero_()


def _freia_dims(channels: int) -> list[tuple[int, int, int]]:
    return [(channels, _FREIA_DUMMY_SPATIAL, _FREIA_DUMMY_SPATIAL)]


@dataclass(frozen=True)
class _InvertibleStageConfig:
    kernel_size: int
    coupling_blocks: int = 3


class _BottleneckSubnet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        _initialize_conv(self.conv1, scale=0.1, use_xavier=True)
        _initialize_conv(self.conv2, scale=0.1, use_xavier=True)
        _initialize_conv(self.conv3, scale=0.0, use_xavier=False)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.activation(self.conv1(input_tensor))
        output = self.activation(self.conv2(output))
        return self.conv3(output)


def _build_coupling_layer(channels: int, kernel_size: int) -> CouplingLayer:
    dims_in = _freia_dims(channels)
    split_length = channels // 4

    def subnet_constructor(input_channels: int, output_channels: int) -> nn.Module:
        return _BottleneckSubnet(input_channels, output_channels, kernel_size)

    # FrEIA's coupling API is stable in spirit but not in details across
    # releases. In particular, the meaning and accepted type of `split_len`
    # differ between versions, so we probe the integer, ratio, and default
    # forms in that order and keep the first one that works.
    candidates = (
        {
            "subnet_constructor": subnet_constructor,
            "clamp": 1.0,
            "split_len": split_length,
        },
        {
            "subnet_constructor": subnet_constructor,
            "clamp": 1.0,
            "split_len": split_length / channels,
        },
        {"subnet_constructor": subnet_constructor, "clamp": 1.0},
    )
    errors: list[TypeError] = []
    for kwargs in candidates:
        try:
            return CouplingLayer.from_freia(dims_in, **kwargs)
        except TypeError as error:
            errors.append(error)
    raise TypeError("Unable to initialize FrEIA coupling layer.") from errors[-1]


def _build_invertible_conv(channels: int) -> InvertibleConv1x1:
    dims_in = _freia_dims(channels)
    errors: list[TypeError] = []
    # FrEIA 0.2 does not expose `InvertibleConv1x1`; its closest equivalent is
    # `Fixed1x1Conv`. The wrapper in `compressai.layers.lic.invertible`
    # normalizes that API difference, so the model keeps a single call site.
    for kwargs in ({"LU_decomposed": False}, {}):
        try:
            return InvertibleConv1x1.from_freia(dims_in, **kwargs)
        except TypeError as error:
            errors.append(error)
    raise TypeError("Unable to initialize FrEIA invertible 1x1 conv.") from errors[-1]


def _build_squeeze_layer(channels: int) -> SqueezeLayer:
    dims_in = _freia_dims(channels)
    errors: list[TypeError] = []
    # FrEIA's `IRevNetUpsampling` is initialized from the downsampled shape
    # rather than the pre-squeeze shape. `SqueezeLayer.from_freia` hides that
    # version-specific constructor requirement for InvCompress.
    for kwargs in ({"legacy_backend": False}, {}):
        try:
            return SqueezeLayer.from_freia(dims_in, **kwargs)
        except TypeError as error:
            errors.append(error)
    raise TypeError("Unable to initialize FrEIA squeeze layer.") from errors[-1]


class _AttentionModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.forw_att = AttentionBlock(channels)
        self.back_att = AttentionBlock(channels)

    def forward(self, input_tensor: Tensor, reverse: bool = False) -> Tensor:
        block = self.back_att if reverse else self.forw_att
        return block(input_tensor)


class _EnhancementModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.forw_enh = EnhBlock(channels)
        self.back_enh = EnhBlock(channels)

    def forward(self, input_tensor: Tensor, reverse: bool = False) -> Tensor:
        block = self.back_enh if reverse else self.forw_enh
        return block(input_tensor)


class _InvertibleCompressionTransform(nn.Module):
    _STAGES: Tuple[_InvertibleStageConfig, ...] = (
        _InvertibleStageConfig(kernel_size=5),
        _InvertibleStageConfig(kernel_size=5),
        _InvertibleStageConfig(kernel_size=3),
        _InvertibleStageConfig(kernel_size=3),
    )

    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        _require_freia()

        self.out_channels = int(latent_channels)
        current_channels = 3
        operations: list[nn.Module] = []

        for stage in self._STAGES:
            operations.append(_build_squeeze_layer(current_channels))
            current_channels *= 4
            operations.append(_build_invertible_conv(current_channels))
            for _ in range(stage.coupling_blocks):
                operations.append(
                    _build_coupling_layer(current_channels, stage.kernel_size)
                )

        if current_channels % self.out_channels != 0:
            raise ValueError(
                "InvCompress requires the expanded channel count to be divisible "
                "by the latent channel count."
            )

        self.expanded_channels = current_channels
        self.channel_repeat = current_channels // self.out_channels
        self.operations = nn.ModuleList(operations)

    def forward(self, input_tensor: Tensor, reverse: bool = False) -> Tensor:
        if not reverse:
            output = input_tensor
            for operation in self.operations:
                output = operation(output)
            batch_size, channels, height, width = output.shape
            if channels != self.expanded_channels:
                raise RuntimeError("Unexpected FrEIA channel count in InvCompress.")
            return output.reshape(
                batch_size,
                self.channel_repeat,
                self.out_channels,
                height,
                width,
            ).mean(dim=1)

        output = input_tensor.repeat(1, self.channel_repeat, 1, 1)
        for operation in reversed(self.operations):
            output = operation(output, reverse=True)
        return output


@_maybe_register_model("invcompress")
class InvCompress(Cheng2020Anchor):
    r"""InvCompress model from Y. Xie, K.L. Cheng, Q. Chen: `"Enhanced
    Invertible Encoding for Learned Image Compression"
    <https://arxiv.org/abs/2108.03690>`_, ACM Int. Conf. on Multimedia
    (ACMMM), 2021.

    Replaces the autoencoder transform of :class:`Cheng2020Anchor` with an
    invertible neural network plus an attentive channel-squeeze layer and a
    feature enhancement module to mitigate information loss.

    Args:
        N (int): Number of channels in the latent representation.
    """

    def __init__(self, N: int = 192, **kwargs: Any) -> None:
        _require_freia()
        super().__init__(N=N, **kwargs)
        self.g_a = None
        self.g_s = None
        self.enh = _EnhancementModule(64)
        self.inv = _InvertibleCompressionTransform(latent_channels=N)
        self.attention = _AttentionModule(N)

    def g_a_func(self, x: Tensor) -> Tensor:
        return self.attention(self.inv(self.enh(x)))

    def g_s_func(self, x: Tensor) -> Tensor:
        output = self.attention(x, reverse=True)
        output = self.inv(output, reverse=True)
        return self.enh(output, reverse=True)

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a_func(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s_func(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.g_a_func(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)

        scale = 4
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * scale
        y_width = z_hat.size(3) * scale
        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for index in range(y.size(0)):
            y_strings.append(
                self._compress_ar(
                    y_hat[index : index + 1],
                    params[index : index + 1],
                    y_height,
                    y_width,
                    kernel_size,
                    padding,
                )
            )

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        scale = 4
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * scale
        y_width = z_hat.size(3) * scale
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
            dtype=z_hat.dtype,
        )

        for index, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[index : index + 1],
                params[index : index + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        return {"x_hat": self.g_s_func(y_hat).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "InvCompress":
        N = state_dict["h_a.0.weight"].size(0)
        net = cls(N=N)
        net.load_state_dict(state_dict)
        return net
