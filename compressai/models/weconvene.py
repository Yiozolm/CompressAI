from __future__ import annotations

import math

from typing import Any, Callable, Dict, Sequence, Tuple, TypeVar

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.latent_codecs import WeChARMLatentCodec
from compressai.layers import ResidualBlock
from compressai.layers.attn import SWAtten
from compressai.layers.wave import (
    WeConveneAnalysisTransform,
    WeConveneHyperAnalysisTransform,
    WeConveneHyperSynthesisTransform,
    WeConveneSynthesisTransform,
    is_pytorch_wavelets_available,
)
from compressai.models._bases import (
    lrp_support_channels as lrp_channels,
    make_entropy_transform,
    slice_support_channels as support_channels,
)
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["WeConvene", "convert_upstream_state_dict"]


class _ResidualSwinBlock(nn.Module):
    """Drop-in replacement for ``SwinBlock`` matching the published WeConvene
    upstream implementation (``candidate/WeConvene/.../SwinBlock``), where the
    transformer-style ``Block``s are replaced by two stacked ``ResidualBlock``s
    operating directly in BCHW.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.block_1 = ResidualBlock(input_dim, output_dim)
        self.block_2 = ResidualBlock(output_dim, output_dim)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.block_2(self.block_1(input_tensor))


def _infer_num_slices(state_dict: Dict[str, Tensor]) -> int:
    indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("latent_codec.cc_mean_transforms_low.") and key.endswith(".0.weight")
    }
    if not indices:
        raise KeyError("Unable to infer num_slices from state_dict")
    return max(indices) + 1


def _is_upstream_state_dict(state_dict: Dict[str, Tensor]) -> bool:
    """Heuristic: upstream WeConvene checkpoints carry the wavelet-domain split
    under ``_real`` / ``_imag`` suffixes at the top level rather than under
    ``latent_codec.*_low`` / ``latent_codec.*_high``.
    """
    for key in state_dict:
        if key.startswith("cc_mean_transforms_real.") or key.startswith(
            "gaussian_conditional_real."
        ):
            return True
    return False


def _has_residual_attention_state(state_dict: Dict[str, Tensor]) -> bool:
    """Returns True when the support transforms in ``state_dict`` were saved
    from a model built with ``use_residual_attention=True`` — detected by the
    presence of ``non_local_block.block_*.conv*`` weights (vs. the transformer
    ``norm1`` / ``msa.attn.*`` weights produced by the default ``SwinBlock``).
    """
    for key in state_dict:
        if (
            ".non_local_block.block_1.conv1.weight" in key
            or ".non_local_block.block_2.conv1.weight" in key
        ):
            return True
    return False


# upstream `g_a` / `g_s` are flat ``nn.Sequential`` of 13 children:
#     0       -> input_block (ResidualBlockWithStride / ResidualBlockUpsample)
#     1..3    -> 3x ResidualBlock  (down1 / up1)
#     4       -> WaveletResidualBlockWithStride / Upsample (tail of down1 / up1)
#     5..7    -> 3x ResidualBlock  (down2 / up2)
#     8       -> Wavelet*          (tail of down2 / up2)
#     9..11   -> 3x ResidualBlock  (down3 / up3)
#     12      -> conv3x3 / subpel_conv3x3 (tail of down3 / up3)
_GA_INDEX_MAP: Dict[int, str] = {0: "input_block"}
_GA_INDEX_MAP.update({i: f"down1.{i - 1}" for i in range(1, 5)})
_GA_INDEX_MAP.update({i: f"down2.{i - 5}" for i in range(5, 9)})
_GA_INDEX_MAP.update({i: f"down3.{i - 9}" for i in range(9, 13)})

_GS_INDEX_MAP: Dict[int, str] = {0: "input_block"}
_GS_INDEX_MAP.update({i: f"up1.{i - 1}" for i in range(1, 5)})
_GS_INDEX_MAP.update({i: f"up2.{i - 5}" for i in range(5, 9)})
_GS_INDEX_MAP.update({i: f"up3.{i - 9}" for i in range(9, 13)})

# upstream `h_a` / `h_mean_s` / `h_scale_s` are flat ``nn.Sequential`` of 5
# children: ``[wavelet_block, ResidualBlock x 3, conv3x3 / subpel_conv3x3]``.
_HA_INDEX_MAP: Dict[int, str] = {0: "wavelet_block"}
_HA_INDEX_MAP.update({i: f"down.{i - 1}" for i in range(1, 5)})

_HS_INDEX_MAP: Dict[int, str] = {0: "wavelet_block"}
_HS_INDEX_MAP.update({i: f"up.{i - 1}" for i in range(1, 5)})

# Latent codec key prefix renames. Upstream stores the wavelet-domain
# low/high-pass branches as ``_real`` / ``_imag``.
_CODEC_PREFIX_RENAMES: Dict[str, str] = {
    "cc_mean_transforms_real.": "latent_codec.cc_mean_transforms_low.",
    "cc_mean_transforms_imag.": "latent_codec.cc_mean_transforms_high.",
    "cc_scale_transforms_real.": "latent_codec.cc_scale_transforms_low.",
    "cc_scale_transforms_imag.": "latent_codec.cc_scale_transforms_high.",
    "lrp_transforms_real.": "latent_codec.lrp_transforms_low.",
    "lrp_transforms_imag.": "latent_codec.lrp_transforms_high.",
    "gaussian_conditional_real.": "latent_codec.gaussian_conditional_low.",
    "gaussian_conditional_imag.": "latent_codec.gaussian_conditional_high.",
}

# Atten key renames. Upstream wraps each ``SWAtten`` in a one-element
# ``nn.Sequential`` (extra ``.0.`` segment) under ``atten_*_{real,imag}``.
_ATTEN_PREFIX_RENAMES: Dict[str, str] = {
    "atten_mean_real.": "latent_codec.mean_support_transforms_low.",
    "atten_mean_imag.": "latent_codec.mean_support_transforms_high.",
    "atten_scale_real.": "latent_codec.scale_support_transforms_low.",
    "atten_scale_imag.": "latent_codec.scale_support_transforms_high.",
}


def _migrate_transform_block(
    key: str,
    prefix: str,
    index_map: Dict[int, str],
    *,
    drop_unused_conv2_indices: Sequence[int] = (),
) -> str | None:
    """Translate one ``g_a/g_s/h_a/h_mean_s/h_scale_s`` key to compressai layout.

    Returns ``None`` when the key should be dropped (DWT/IDWT buffers, unused
    upstream ``conv2`` weights inside wavelet blocks).
    """
    rest = key[len(prefix):]
    head, _, tail = rest.partition(".")
    if not head.isdigit():
        return key
    idx = int(head)
    if idx not in index_map:
        return key
    if tail.startswith("dwt.") or tail.startswith("idwt."):
        return None  # buffers regenerated at construction (different layout)
    if idx in drop_unused_conv2_indices and tail.startswith("conv2."):
        return None  # upstream wavelet block has unused conv2
    return f"{prefix}{index_map[idx]}.{tail}"


def convert_upstream_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Translate an upstream WeConvene checkpoint to compressai key layout.

    The upstream model (``candidate/WeConvene``) stores parameters in a flat
    ``nn.Sequential`` for each transform, splits the wavelet-domain entropy
    branches as ``_real`` / ``_imag``, wraps each ``SWAtten`` in a
    one-element ``nn.Sequential``, and registers DWT/IDWT kernels as
    ``w_{ll,lh,hl,hh}`` / ``filters`` buffers. This helper rewrites those
    keys to the nested compressai layout (and drops buffers that are
    re-registered by ``pytorch_wavelets`` at construction time).
    """
    migrated: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        # Drop top-level DWT/IDWT (constructed inside the latent codec).
        if key.startswith("dwt.") or key.startswith("idwt."):
            continue

        # Codec prefix renames: cc_*/lrp_*/gaussian_conditional_* (no extra
        # wrapper to strip — match longest prefix).
        renamed = None
        for src_prefix, dst_prefix in _CODEC_PREFIX_RENAMES.items():
            if key.startswith(src_prefix):
                renamed = dst_prefix + key[len(src_prefix):]
                break
        if renamed is not None:
            migrated[renamed] = value
            continue

        # SWAtten renames: drop the `.0.` Sequential wrapper between the
        # ModuleList index and the SWAtten internals.
        for src_prefix, dst_prefix in _ATTEN_PREFIX_RENAMES.items():
            if key.startswith(src_prefix):
                rest = key[len(src_prefix):]
                slice_idx, _, after_slice = rest.partition(".")
                wrapper, _, after_wrapper = after_slice.partition(".")
                if wrapper != "0":
                    raise RuntimeError(
                        f"Unexpected upstream attention key (no `.0.` wrapper): {key}"
                    )
                renamed = f"{dst_prefix}{slice_idx}.{after_wrapper}"
                break
        if renamed is not None:
            migrated[renamed] = value
            continue

        # Transform blocks: g_a / g_s / h_a / h_mean_s / h_scale_s.
        if key.startswith("g_a."):
            renamed = _migrate_transform_block(
                key, "g_a.", _GA_INDEX_MAP, drop_unused_conv2_indices=(4, 8)
            )
        elif key.startswith("g_s."):
            renamed = _migrate_transform_block(key, "g_s.", _GS_INDEX_MAP)
        elif key.startswith("h_a."):
            renamed = _migrate_transform_block(
                key, "h_a.", _HA_INDEX_MAP, drop_unused_conv2_indices=(0,)
            )
        elif key.startswith("h_mean_s."):
            renamed = _migrate_transform_block(key, "h_mean_s.", _HS_INDEX_MAP)
        elif key.startswith("h_scale_s."):
            renamed = _migrate_transform_block(key, "h_scale_s.", _HS_INDEX_MAP)
        else:
            renamed = key

        if renamed is None:
            continue
        migrated[renamed] = value

    return migrated


def _infer_max_support_slices(
    state_dict: Dict[str, Tensor],
    M: int,
    num_slices: int,
) -> int:
    slice_channels = M // num_slices
    last_index = num_slices - 1
    key = f"latent_codec.mean_support_transforms_low.{last_index}.in_conv.weight"
    if key not in state_dict:
        return last_index
    input_channels = state_dict[key].size(1)
    return max(0, (input_channels - M) // slice_channels)


def _infer_support_attention(state_dict: Dict[str, Tensor]) -> Tuple[int, int, int]:
    in_conv_key = "latent_codec.mean_support_transforms_low.0.in_conv.weight"
    table_key = (
        "latent_codec.mean_support_transforms_low.0."
        "non_local_block.block_1.msa.attn.relative_position_bias_table"
    )
    if in_conv_key not in state_dict or table_key not in state_dict:
        return 8, 16, 128

    hidden_dim = state_dict[in_conv_key].size(0)
    table_size, num_heads = state_dict[table_key].shape
    window_size = (math.isqrt(table_size) + 1) // 2
    head_dim = hidden_dim // num_heads
    return window_size, head_dim, hidden_dim


def _is_pytorch_wavelets_buffer_key(key: str) -> bool:
    """Match buffer paths registered by ``pytorch_wavelets`` inside our DWT2D /
    IDWT2D wrappers — these are deterministic per-wavelet kernels rebuilt at
    ``__init__`` time, so they can safely be missing from a state dict.
    """
    return (
        ".dwt.transform." in key
        or ".idwt.inverse." in key
        or key.startswith("latent_codec.dwt.transform.")
        or key.startswith("latent_codec.idwt.inverse.")
    )

_ModelType = TypeVar("_ModelType", bound=type[nn.Module])


def _identity_decorator(cls: _ModelType) -> _ModelType:
    return cls


def _maybe_register_model(name: str) -> Callable[[_ModelType], _ModelType]:
    if is_pytorch_wavelets_available():
        return register_model(name)
    return _identity_decorator


def _require_wavelets() -> None:
    if is_pytorch_wavelets_available():
        return
    raise ModuleNotFoundError(
        "WeConvene requires the optional dependency `pytorch_wavelets`. "
        "Install `compressai[lic]` to enable this model."
    )


@_maybe_register_model("weconvene")
class WeConvene(CompressionModel):
    r"""WeConvene model from H. Fu, J. Liang, Z. Fang, J. Han, F. Liang,
    G. Zhang: `"WeConvene: Learned Image Compression with Wavelet-Domain
    Convolution and Entropy Model"
    <https://arxiv.org/abs/2407.09983>`_, European Conf. on Computer Vision
    (ECCV), 2024.

    Inserts wavelet-domain convolution (WeConv) modules in the
    analysis/synthesis transforms and performs entropy coding in the wavelet
    domain via a wavelet-domain channel-wise autoregressive entropy model
    (WeChARM) that codes low-frequency coefficients first and then uses them
    as priors for the high-frequency coefficients.

    Args:
        N (int): Number of channels in the hyperprior.
        M (int): Number of channels in the latent representation.
        wavelet (str): Wavelet basis used by the WeConv / WeChARM modules.
        num_slices (int): Number of channel slices for the entropy model.
    """

    def __init__(
        self,
        N: int = 128,
        M: int = 320,
        hyper_channels: int = 192,
        num_slices: int = 5,
        max_support_slices: int = 5,
        residual_blocks: int = 3,
        wavelet: str = "haar",
        support_window_size: int = 8,
        support_head_dim: int = 16,
        support_attention_dim: int = 128,
        use_residual_attention: bool = False,
        **kwargs: Any,
    ) -> None:
        _require_wavelets()
        super().__init__(**kwargs)
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")
        if support_attention_dim % support_head_dim != 0:
            raise ValueError("support_attention_dim must be divisible by support_head_dim")

        self.N = int(N)
        self.M = int(M)
        self.hyper_channels = int(hyper_channels)
        self.num_slices = int(num_slices)
        self.max_support_slices = int(max_support_slices)
        self.residual_blocks = int(residual_blocks)
        self.wavelet = wavelet
        self.support_window_size = int(support_window_size)
        self.support_head_dim = int(support_head_dim)
        self.support_attention_dim = int(support_attention_dim)
        self.use_residual_attention = bool(use_residual_attention)

        self.g_a = WeConveneAnalysisTransform(
            N=self.N,
            M=self.M,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )
        self.g_s = WeConveneSynthesisTransform(
            N=self.N,
            M=self.M,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )
        self.h_a = WeConveneHyperAnalysisTransform(
            N=self.N,
            M=self.M,
            hyper_channels=self.hyper_channels,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )
        self.h_mean_s = WeConveneHyperSynthesisTransform(
            N=self.N,
            M=self.M,
            hyper_channels=self.hyper_channels,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )
        self.h_scale_s = WeConveneHyperSynthesisTransform(
            N=self.N,
            M=self.M,
            hyper_channels=self.hyper_channels,
            residual_blocks=self.residual_blocks,
            wavelet=self.wavelet,
        )

        low_slice_channels = self.M // self.num_slices
        high_slice_channels = 3 * low_slice_channels

        mean_support_transforms_low = nn.ModuleList(
            SWAtten(
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                self.support_head_dim,
                self.support_window_size,
                0.0,
                inter_dim=self.support_attention_dim,
            )
            for index in range(self.num_slices)
        )
        scale_support_transforms_low = nn.ModuleList(
            SWAtten(
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                self.support_head_dim,
                self.support_window_size,
                0.0,
                inter_dim=self.support_attention_dim,
            )
            for index in range(self.num_slices)
        )
        mean_support_transforms_high = nn.ModuleList(
            SWAtten(
                support_channels(2 * self.M, high_slice_channels, index, self.max_support_slices),
                support_channels(self.M, high_slice_channels, index, self.max_support_slices),
                self.support_head_dim,
                self.support_window_size,
                0.0,
                inter_dim=self.support_attention_dim,
            )
            for index in range(self.num_slices)
        )
        scale_support_transforms_high = nn.ModuleList(
            SWAtten(
                support_channels(2 * self.M, high_slice_channels, index, self.max_support_slices),
                support_channels(self.M, high_slice_channels, index, self.max_support_slices),
                self.support_head_dim,
                self.support_window_size,
                0.0,
                inter_dim=self.support_attention_dim,
            )
            for index in range(self.num_slices)
        )
        cc_mean_transforms_low = nn.ModuleList(
            make_entropy_transform(
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                low_slice_channels,
            )
            for index in range(self.num_slices)
        )
        cc_scale_transforms_low = nn.ModuleList(
            make_entropy_transform(
                support_channels(self.M, low_slice_channels, index, self.max_support_slices),
                low_slice_channels,
            )
            for index in range(self.num_slices)
        )
        cc_mean_transforms_high = nn.ModuleList(
            make_entropy_transform(
                support_channels(self.M, high_slice_channels, index, self.max_support_slices),
                high_slice_channels,
            )
            for index in range(self.num_slices)
        )
        cc_scale_transforms_high = nn.ModuleList(
            make_entropy_transform(
                support_channels(self.M, high_slice_channels, index, self.max_support_slices),
                high_slice_channels,
            )
            for index in range(self.num_slices)
        )
        lrp_transforms_low = nn.ModuleList(
            make_entropy_transform(
                lrp_channels(self.M, low_slice_channels, index, self.max_support_slices),
                low_slice_channels,
            )
            for index in range(self.num_slices)
        )
        lrp_transforms_high = nn.ModuleList(
            make_entropy_transform(
                lrp_channels(2 * self.M, high_slice_channels, index, self.max_support_slices),
                high_slice_channels,
            )
            for index in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(self.hyper_channels)
        self.latent_codec = WeChARMLatentCodec(
            M=self.M,
            cc_mean_transforms_low=cc_mean_transforms_low,
            cc_scale_transforms_low=cc_scale_transforms_low,
            cc_mean_transforms_high=cc_mean_transforms_high,
            cc_scale_transforms_high=cc_scale_transforms_high,
            lrp_transforms_low=lrp_transforms_low,
            lrp_transforms_high=lrp_transforms_high,
            gaussian_conditional_low=GaussianConditional(None),
            gaussian_conditional_high=GaussianConditional(None),
            mean_support_transforms_low=mean_support_transforms_low,
            scale_support_transforms_low=scale_support_transforms_low,
            mean_support_transforms_high=mean_support_transforms_high,
            scale_support_transforms_high=scale_support_transforms_high,
            num_slices=self.num_slices,
            max_support_slices=self.max_support_slices,
            wavelet=self.wavelet,
        )

        if self.use_residual_attention:
            self._patch_residual_attention()

    def _patch_residual_attention(self) -> None:
        """Replace each ``SWAtten.non_local_block`` with ``_ResidualSwinBlock``.

        Used to match the published WeConvene checkpoints, where the upstream
        ``SwinBlock`` is two stacked ``ResidualBlock``s rather than the Swin
        transformer ``Block`` pair from ``compressai.layers.attn.swin``.
        """
        for transforms in (
            self.latent_codec.mean_support_transforms_low,
            self.latent_codec.mean_support_transforms_high,
            self.latent_codec.scale_support_transforms_low,
            self.latent_codec.scale_support_transforms_high,
        ):
            for module in transforms:
                module.non_local_block = _ResidualSwinBlock(
                    self.support_attention_dim, self.support_attention_dim
                )

    @property
    def gaussian_conditional_low(self) -> GaussianConditional:
        return self.latent_codec.gaussian_conditional_low

    @property
    def gaussian_conditional_high(self) -> GaussianConditional:
        return self.latent_codec.gaussian_conditional_high

    @property
    def atten_mean_low(self) -> nn.ModuleList:
        return self.latent_codec.mean_support_transforms_low

    @property
    def atten_scale_low(self) -> nn.ModuleList:
        return self.latent_codec.scale_support_transforms_low

    @property
    def atten_mean_high(self) -> nn.ModuleList:
        return self.latent_codec.mean_support_transforms_high

    @property
    def atten_scale_high(self) -> nn.ModuleList:
        return self.latent_codec.scale_support_transforms_high

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        y_wavelet = self.latent_codec.to_wavelet(y)
        z = self.h_a(y_wavelet)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        latent_means = self.h_mean_s(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        y_out = self.latent_codec(
            y,
            latent_means,
            latent_scales,
            wavelet_output=y_wavelet,
        )
        return {
            "x_hat": self.g_s(y_out["y_hat"]),
            "likelihoods": {
                "y_low": y_out["likelihoods"]["y_low"],
                "y_high": y_out["likelihoods"]["y_high"],
                "z": z_likelihoods,
            },
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        y_wavelet = self.latent_codec.to_wavelet(y)
        z = self.h_a(y_wavelet)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        y_out = self.latent_codec.compress(
            y,
            self.h_mean_s(z_hat),
            self.h_scale_s(z_hat),
            wavelet_output=y_wavelet,
        )
        return {
            "strings": [*y_out["strings"], z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(
        self,
        strings: Sequence[Sequence[bytes]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        if len(strings) != 3:
            raise ValueError("strings must contain [low_strings, high_strings, z_strings]")

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        y_shape = (z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        y_out = self.latent_codec.decompress(
            strings[:2],
            y_shape,
            self.h_mean_s(z_hat),
            self.h_scale_s(z_hat),
        )
        return {"x_hat": self.g_s(y_out["y_hat"]).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "WeConvene":
        if _is_upstream_state_dict(state_dict):
            state_dict = convert_upstream_state_dict(state_dict)
            use_residual_attention = True
        else:
            use_residual_attention = _has_residual_attention_state(state_dict)

        N = state_dict["g_a.input_block.conv1.weight"].size(0)
        num_slices = _infer_num_slices(state_dict)
        M = state_dict["latent_codec.cc_mean_transforms_low.0.4.weight"].size(0) * num_slices
        hyper_channels = state_dict["entropy_bottleneck.quantiles"].size(0)
        max_support_slices = _infer_max_support_slices(state_dict, M, num_slices)
        support_window_size, support_head_dim, support_attention_dim = _infer_support_attention(
            state_dict
        )

        net = cls(
            N=N,
            M=M,
            hyper_channels=hyper_channels,
            num_slices=num_slices,
            max_support_slices=max_support_slices,
            support_window_size=support_window_size,
            support_head_dim=support_head_dim,
            support_attention_dim=support_attention_dim,
            use_residual_attention=use_residual_attention,
        )
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            key
            for key in net.state_dict()
            if key.endswith("relative_position_index")
            or _is_pytorch_wavelets_buffer_key(key)
        }
        missing_keys = set(incompatible_keys.missing_keys) - allowed_missing
        if missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                "Unexpected incompatibility while loading WeConvene state_dict: "
                f"missing={sorted(missing_keys)}, "
                f"unexpected={sorted(incompatible_keys.unexpected_keys)}"
            )
        return net
