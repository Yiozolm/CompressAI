from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ModuleNotFoundError as error:
    DWTForward = None  # type: ignore[assignment]
    DWTInverse = None  # type: ignore[assignment]
    _PYTORCH_WAVELETS_IMPORT_ERROR = error
else:
    _PYTORCH_WAVELETS_IMPORT_ERROR = None

from ..lic.blocks import OLP

__all__ = [
    "DWT2D",
    "IDWT2D",
    "DWT_2D",
    "IDWT_2D",
    "WLS",
    "iWLS",
    "is_pytorch_wavelets_available",
]


def is_pytorch_wavelets_available() -> bool:
    return DWTForward is not None and DWTInverse is not None


def _require_pytorch_wavelets() -> None:
    if is_pytorch_wavelets_available():
        return
    raise ModuleNotFoundError(
        "Wavelet layers require the optional dependency `pytorch_wavelets`."
    ) from _PYTORCH_WAVELETS_IMPORT_ERROR


def _make_scaling_factors(channels: int) -> Tensor:
    return torch.cat(
        (
            torch.full((1, 1, channels), 0.5),
            torch.full((1, 1, channels), 0.5),
            torch.full((1, 1, channels), 0.5),
            torch.zeros((1, 1, channels)),
        ),
        dim=2,
    )


class DWT2D(nn.Module):
    """Single-level DWT wrapper with channel-concatenated subbands."""

    def __init__(self, wave: str = "haar", mode: str = "zero") -> None:
        super().__init__()
        _require_pytorch_wavelets()
        self.transform = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, input_tensor: Tensor) -> Tensor:
        lowpass, highpass_pyramid = self.transform(input_tensor)
        [highpass] = highpass_pyramid
        subbands = (
            lowpass,
            highpass[:, :, 0, ...],
            highpass[:, :, 1, ...],
            highpass[:, :, 2, ...],
        )
        return torch.cat(subbands, dim=1)


class IDWT2D(nn.Module):
    """Inverse wrapper matching :class:`DWT2D` channel layout."""

    def __init__(self, wave: str = "haar", mode: str = "zero") -> None:
        super().__init__()
        _require_pytorch_wavelets()
        self.inverse = DWTInverse(wave=wave, mode=mode)

    def forward(self, input_tensor: Tensor) -> Tensor:
        lowpass, band_lh, band_hl, band_hh = input_tensor.chunk(4, dim=1)
        highpass = torch.stack((band_lh, band_hl, band_hh), dim=2)
        return self.inverse((lowpass, [highpass]))


class WLS(nn.Module):
    """Wavelet Linear Scaling analysis block."""

    def __init__(self, in_dim: int, out_dim: int, wave: str = "haar") -> None:
        super().__init__()
        self.dwt = DWT2D(wave=wave)
        self.olp = OLP(in_dim * 4, out_dim)
        self.scaling_factors = nn.Parameter(_make_scaling_factors(in_dim))

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.dwt(input_tensor)
        batch_size, _, height, width = output.shape
        output = output.view(batch_size, -1, height * width).permute(0, 2, 1)
        output = output * torch.exp(self.scaling_factors)
        output = self.olp(output)
        output = output.view(batch_size, height, width, -1)
        return output.permute(0, 3, 1, 2).contiguous()


class iWLS(nn.Module):
    """Inverse Wavelet Linear Scaling synthesis block."""

    def __init__(self, in_dim: int, out_dim: int, wave: str = "haar") -> None:
        super().__init__()
        self.idwt = IDWT2D(wave=wave)
        self.olp = OLP(in_dim, out_dim * 4)
        self.scaling_factors = nn.Parameter(_make_scaling_factors(out_dim))

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size, _, height, width = input_tensor.shape
        output = input_tensor.view(batch_size, -1, height * width).permute(0, 2, 1)
        output = self.olp(output)
        output = output / torch.exp(self.scaling_factors)
        output = output.view(batch_size, height, width, -1)
        output = output.permute(0, 3, 1, 2).contiguous()
        return self.idwt(output)


DWT_2D = DWT2D
IDWT_2D = IDWT2D
