from __future__ import annotations

from importlib import import_module, util
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init

from torch import Tensor

__all__ = [
    "CouplingLayer",
    "DenseBlock",
    "EnhBlock",
    "InvertibleConv1x1",
    "SqueezeLayer",
    "build_freia_coupling_layer",
    "build_freia_invertible_conv",
    "build_freia_squeeze_pair",
    "is_freia_available",
]

_FREIA_MODULES: Optional[Any] = None
_FREIA_IMPORT_ERROR: Optional[ModuleNotFoundError] = None


def is_freia_available() -> bool:
    return util.find_spec("FrEIA") is not None


def _load_freia_modules() -> Any:
    global _FREIA_IMPORT_ERROR, _FREIA_MODULES

    if _FREIA_MODULES is not None:
        return _FREIA_MODULES

    try:
        _FREIA_MODULES = import_module("FrEIA.modules")
    except ModuleNotFoundError as error:
        _FREIA_IMPORT_ERROR = error
        raise ModuleNotFoundError(
            "Invertible layers require the optional dependency `FrEIA`."
        ) from error
    return _FREIA_MODULES


def _resolve_freia_class(candidate_names: Sequence[str]) -> Any:
    modules = _load_freia_modules()
    for candidate_name in candidate_names:
        if hasattr(modules, candidate_name):
            return getattr(modules, candidate_name)
    raise AttributeError(
        "Unable to find any of "
        f"{tuple(candidate_names)} in `FrEIA.modules`."
    )


def _extract_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (list, tuple)):
        first = output[0]
        if isinstance(first, Tensor):
            return first
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], Tensor):
            return first[0]
    raise TypeError("Unsupported FrEIA output structure.")


def _call_backend(backend: nn.Module, input_tensor: Tensor, reverse: bool) -> Tensor:
    call_patterns = (
        lambda: backend([input_tensor], rev=reverse),
        lambda: backend(input_tensor, rev=reverse),
        lambda: backend([input_tensor], reverse=reverse),
        lambda: backend(input_tensor, reverse=reverse),
        lambda: backend([input_tensor]),
        lambda: backend(input_tensor),
    )
    type_errors = []
    for pattern in call_patterns:
        try:
            return _extract_tensor(pattern())
        except TypeError as error:
            type_errors.append(error)
    raise TypeError(
        "Could not dispatch to the wrapped FrEIA backend."
    ) from type_errors[-1]


def build_freia_coupling_layer(
    *args: Any,
    kind: str = "glow",
    **kwargs: Any,
) -> nn.Module:
    kind_to_candidates = {
        "glow": ("GLOWCouplingBlock",),
        "rnvp": ("RNVPCouplingBlock",),
        "nice": ("NICECouplingBlock",),
    }
    if kind not in kind_to_candidates:
        raise ValueError(f"Unsupported coupling kind: {kind}")
    coupling_class = _resolve_freia_class(kind_to_candidates[kind])
    return coupling_class(*args, **kwargs)


def build_freia_invertible_conv(*args: Any, **kwargs: Any) -> nn.Module:
    conv_class = _resolve_freia_class(
        ("InvertibleConv1x1", "Fixed1x1ConvOrthogonal")
    )
    return conv_class(*args, **kwargs)


def build_freia_squeeze_pair(
    *args: Any,
    downsample_name: str = "IRevNetDownsampling",
    upsample_name: str = "IRevNetUpsampling",
    **kwargs: Any,
) -> Tuple[nn.Module, nn.Module]:
    downsample_class = _resolve_freia_class((downsample_name,))
    upsample_class = _resolve_freia_class((upsample_name,))
    return downsample_class(*args, **kwargs), upsample_class(*args, **kwargs)


def _initialize_weights(modules: Sequence[nn.Module], scale: float = 1.0) -> None:
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)


def _initialize_weights_xavier(
    modules: Sequence[nn.Module],
    scale: float = 1.0,
) -> None:
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                init.xavier_normal_(layer.weight)
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)


class DenseBlock(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        init_method: str = "xavier",
        growth_channels: int = 32,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, growth_channels, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(
            channel_in + growth_channels,
            growth_channels,
            3,
            1,
            1,
            bias=bias,
        )
        self.conv3 = nn.Conv2d(
            channel_in + 2 * growth_channels,
            growth_channels,
            3,
            1,
            1,
            bias=bias,
        )
        self.conv4 = nn.Conv2d(
            channel_in + 3 * growth_channels,
            growth_channels,
            3,
            1,
            1,
            bias=bias,
        )
        self.conv5 = nn.Conv2d(
            channel_in + 4 * growth_channels,
            channel_out,
            3,
            1,
            1,
            bias=bias,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init_method == "xavier":
            _initialize_weights_xavier(
                [self.conv1, self.conv2, self.conv3, self.conv4],
                scale=0.1,
            )
        else:
            _initialize_weights(
                [self.conv1, self.conv2, self.conv3, self.conv4],
                scale=0.1,
            )
        _initialize_weights([self.conv5], scale=0.0)

    def forward(self, input_tensor: Tensor) -> Tensor:
        feat1 = self.activation(self.conv1(input_tensor))
        feat2 = self.activation(self.conv2(torch.cat((input_tensor, feat1), 1)))
        feat3 = self.activation(
            self.conv3(torch.cat((input_tensor, feat1, feat2), 1))
        )
        feat4 = self.activation(
            self.conv4(torch.cat((input_tensor, feat1, feat2, feat3), 1))
        )
        return self.conv5(
            torch.cat((input_tensor, feat1, feat2, feat3, feat4), 1)
        )


class EnhBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            DenseBlock(3, channels),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            DenseBlock(channels, 3),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor + self.layers(input_tensor) * 0.2


class CouplingLayer(nn.Module):
    def __init__(self, backend: nn.Module) -> None:
        super().__init__()
        self.backend = backend

    @classmethod
    def from_freia(cls, *args: Any, **kwargs: Any) -> "CouplingLayer":
        return cls(build_freia_coupling_layer(*args, **kwargs))

    def forward(
        self,
        input_tensor: Tensor,
        reverse: bool = False,
        rev: Optional[bool] = None,
    ) -> Tensor:
        use_reverse = rev if rev is not None else reverse
        return _call_backend(self.backend, input_tensor, use_reverse)


class InvertibleConv1x1(nn.Module):
    def __init__(self, backend: nn.Module) -> None:
        super().__init__()
        self.backend = backend

    @classmethod
    def from_freia(cls, *args: Any, **kwargs: Any) -> "InvertibleConv1x1":
        return cls(build_freia_invertible_conv(*args, **kwargs))

    def forward(
        self,
        input_tensor: Tensor,
        reverse: bool = False,
        rev: Optional[bool] = None,
    ) -> Tensor:
        use_reverse = rev if rev is not None else reverse
        return _call_backend(self.backend, input_tensor, use_reverse)


class SqueezeLayer(nn.Module):
    def __init__(
        self,
        downsample_backend: nn.Module,
        upsample_backend: nn.Module,
    ) -> None:
        super().__init__()
        self.downsample_backend = downsample_backend
        self.upsample_backend = upsample_backend

    @classmethod
    def from_freia(cls, *args: Any, **kwargs: Any) -> "SqueezeLayer":
        downsample_backend, upsample_backend = build_freia_squeeze_pair(
            *args,
            **kwargs,
        )
        return cls(downsample_backend, upsample_backend)

    def forward(
        self,
        input_tensor: Tensor,
        reverse: bool = False,
        rev: Optional[bool] = None,
    ) -> Tensor:
        use_reverse = rev if rev is not None else reverse
        backend = self.upsample_backend if use_reverse else self.downsample_backend
        return _call_backend(backend, input_tensor, reverse=False)
