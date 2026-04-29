"""ShiftLIC — Shift-block based learned image compression.

Bao et al., "ShiftLIC: Lossy Image Compression with Lightweight Channel-wise
Shift", `arXiv:2503.23052`_ (TCSVT 2025). License is not declared upstream;
this fork ships per existing repo policy.

Three pre-defined configs share the same encoder backbone (PixelUnshuffle +
1×1 + ``ResidualBlockShift × 3`` per scale; ``CheapCS1`` injected after the
192- and 256-channel stacks for ``middle`` and ``large``) and the same
hyperencoder.

* ``small`` / ``middle``: scale-only ``ScaleHyperprior``-style entropy
  model (hyperdecoder outputs ``M`` channels of ``σ``).
* ``large``: hyperdecoder outputs ``2M`` channels (``μ`` + ``σ``) and the
  staged checkerboard
  :class:`compressai.latent_codecs.MultistageCheckerboardLatentCodec`
  (``gamma_mode="linear"``, ``make_cc_transform=ResidualShiftStack``)
  handles the latent.

.. _arXiv:2503.23052: https://arxiv.org/abs/2503.23052
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import init

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.latent_codecs import MultistageCheckerboardLatentCodec
from compressai.layers import (
    CheapCS1,
    ResidualBlockShift,
    ResidualShiftStack,
)
from compressai.models.base import CompressionModel
from compressai.ops import quantize_ste
from compressai.registry import register_model


__all__ = ["ShiftLIC"]


_VARIANTS = ("small", "middle", "large")
_VariantT = Literal["small", "middle", "large"]
_CODEC_PREFIXES = (
    "entropy_parameters_",
    "cc_transforms.",
    "sc_transform_",
    "gaussian_conditional.",
)


@register_model("shiftlic")
class ShiftLIC(CompressionModel):
    """ShiftLIC end-to-end image compression model (3 variants).

    Args:
        variant: ``"small"`` / ``"middle"`` / ``"large"``.
        N: Hyper-latent / hyper-tower channel width (default 192).
        M: Latent (``y``) channels (default 320).
    """

    def __init__(
        self,
        variant: _VariantT = "large",
        N: int = 192,
        M: int = 320,
    ) -> None:
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {_VARIANTS}; got {variant!r}"
            )
        self.variant = variant
        self.N = int(N)
        self.M = int(M)

        with_cs = variant in ("middle", "large")
        self.encoder = self._build_encoder(M, with_cs)
        self.decoder = self._build_decoder(M, with_cs)
        self.hyperencoder = self._build_hyperencoder(M, N)
        # ``large`` consumes both μ and σ via the staged codec, so its
        # hyperdecoder outputs ``2M`` channels; small/middle remain scale-only.
        hyper_out_ch = 2 * M if variant == "large" else M
        self.hyperdecoder = self._build_hyperdecoder(M, N, hyper_out_ch)

        self.entropy_bottleneck = EntropyBottleneck(N)

        if variant == "large":
            self.latent_codec = MultistageCheckerboardLatentCodec(
                channels=M,
                hyper_channels=2 * M,
                num_iters=4,
                gamma_mode="linear",
                make_cc_transform=ResidualShiftStack,
            )
            # Do NOT alias ``self.latent_codec.gaussian_conditional`` to a
            # top-level attribute: nn.Module would then emit duplicate
            # state_dict keys (one under ``gaussian_conditional.*`` and one
            # under ``latent_codec.gaussian_conditional.*``), which both must
            # be present at load time. ``CompressionModel.update`` /
            # ``aux_loss`` traverse ``named_modules()`` so they find the codec's
            # inner instance just fine.
        else:
            self.gaussian_conditional = GaussianConditional(None)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Encoder / decoder factories.
    # ------------------------------------------------------------------
    @staticmethod
    def _build_encoder(M: int, with_cs: bool) -> nn.Sequential:
        layers: List[nn.Module] = [
            nn.PixelUnshuffle(2),
            nn.Conv2d(12, 128, kernel_size=1, stride=1, padding=0),
            ResidualBlockShift(128, 128),
            ResidualBlockShift(128, 128),
            ResidualBlockShift(128, 128),
            nn.PixelUnshuffle(2),
            nn.Conv2d(128 * 4, 192, kernel_size=1, stride=1, padding=0),
            ResidualBlockShift(192, 192),
            ResidualBlockShift(192, 192),
            ResidualBlockShift(192, 192),
        ]
        if with_cs:
            layers.append(CheapCS1(192))
        layers.extend(
            [
                nn.PixelUnshuffle(2),
                nn.Conv2d(192 * 4, 256, kernel_size=1, stride=1, padding=0),
                ResidualBlockShift(256, 256),
                ResidualBlockShift(256, 256),
                ResidualBlockShift(256, 256),
            ]
        )
        if with_cs:
            layers.append(CheapCS1(256))
        layers.extend(
            [
                nn.PixelUnshuffle(2),
                nn.Conv2d(256 * 4, M, kernel_size=1, stride=1, padding=0),
            ]
        )
        return nn.Sequential(*layers)

    @staticmethod
    def _build_decoder(M: int, with_cs: bool) -> nn.Sequential:
        layers: List[nn.Module] = [
            nn.Conv2d(M, 256 * 4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
        ]
        if with_cs:
            layers.append(CheapCS1(256))
        layers.extend(
            [
                ResidualBlockShift(256, 256),
                ResidualBlockShift(256, 256),
                ResidualBlockShift(256, 256),
                nn.Conv2d(256, 192 * 4, kernel_size=1, stride=1, padding=0),
                nn.PixelShuffle(2),
            ]
        )
        if with_cs:
            layers.append(CheapCS1(192))
        layers.extend(
            [
                ResidualBlockShift(192, 192),
                ResidualBlockShift(192, 192),
                ResidualBlockShift(192, 192),
                nn.Conv2d(192, 128 * 4, kernel_size=1, stride=1, padding=0),
                nn.PixelShuffle(2),
                ResidualBlockShift(128, 128),
                ResidualBlockShift(128, 128),
                ResidualBlockShift(128, 128),
                nn.Conv2d(128, 12, kernel_size=1, stride=1, padding=0),
                nn.PixelShuffle(2),
            ]
        )
        return nn.Sequential(*layers)

    @staticmethod
    def _build_hyperencoder(M: int, N: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlockShift(M, N),
            nn.LeakyReLU(inplace=True),
            ResidualBlockShift(N, N),
            nn.LeakyReLU(inplace=True),
            nn.PixelUnshuffle(2),
            nn.Conv2d(N * 4, N, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            ResidualBlockShift(N, N),
            nn.LeakyReLU(inplace=True),
            nn.PixelUnshuffle(2),
            nn.Conv2d(N * 4, N, kernel_size=1, stride=1, padding=0),
        )

    @staticmethod
    def _build_hyperdecoder(M: int, N: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlockShift(N, N),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N * 4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True),
            ResidualBlockShift(N, N),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N * 4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True),
            ResidualBlockShift(N, out_ch),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Dict[str, Any]:
        y = self.encoder(x)
        # Upstream small/middle take ``abs(y)`` for the hyperencoder; large
        # uses raw ``y``. Match upstream so converted weights agree.
        z_input = torch.abs(y) if self.variant != "large" else y
        z = self.hyperencoder(z_input)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        if self.variant == "large":
            params = self.hyperdecoder(z_hat)
            codec_out = self.latent_codec(y, params)
            x_hat = self.decoder(codec_out["y_hat"]).clamp_(0.0, 1.0)
            return {
                "x_hat": x_hat,
                "likelihoods": {
                    "y": codec_out["likelihoods"]["y"],
                    "z": z_likelihoods,
                },
            }

        scales = self.hyperdecoder(z_hat)
        _, y_likelihoods = self.gaussian_conditional(y, scales)
        y_hat = quantize_ste(y)
        x_hat = self.decoder(y_hat).clamp_(0.0, 1.0)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: Tensor) -> Dict[str, Any]:
        y = self.encoder(x)
        z_input = torch.abs(y) if self.variant != "large" else y
        z = self.hyperencoder(z_input)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        if self.variant == "large":
            params = self.hyperdecoder(z_hat)
            codec_out = self.latent_codec.compress(y, params)
            return {
                "strings": [codec_out["strings"][0], z_strings],
                "shape": z.size()[-2:],
            }

        scales = self.hyperdecoder(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        if self.variant == "large":
            params = self.hyperdecoder(z_hat)
            codec_out = self.latent_codec.decompress(
                [strings[0]], shape, params
            )
            x_hat = self.decoder(codec_out["y_hat"]).clamp_(0.0, 1.0)
            return {"x_hat": x_hat}

        scales_hat = self.hyperdecoder(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.decoder(y_hat).clamp_(0.0, 1.0)
        return {"x_hat": x_hat}

    # ------------------------------------------------------------------
    # State-dict loading: rewrite upstream large-variant top-level entropy
    # keys to live under ``latent_codec.*``. For small/middle the codec is
    # not used so no remap is needed.
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict: bool = True):
        if self.variant == "large":
            remapped: Dict[str, Tensor] = {}
            for key, value in state_dict.items():
                if key.startswith("latent_codec."):
                    remapped[key] = value
                elif any(key.startswith(prefix) for prefix in _CODEC_PREFIXES):
                    remapped[f"latent_codec.{key}"] = value
                else:
                    remapped[key] = value
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "ShiftLIC":
        # Decide variant. Codec-only top-level prefixes (sc_transform_,
        # cc_transforms., entropy_parameters_) appear ONLY in ``large``;
        # gaussian_conditional is a top-level attr in small/middle so it
        # cannot be used as a discriminator. Otherwise scan for ``CheapCS1``
        # (its child ``CheapChannel`` shows up as ``encoder.<i>.CheapChannel*``)
        # → ``middle``; else ``small``.
        keys = list(state_dict.keys())
        large_only_prefixes = (
            "entropy_parameters_",
            "cc_transforms.",
            "sc_transform_",
        )
        has_codec = any(
            k.startswith("latent_codec.")
            or any(k.startswith(prefix) for prefix in large_only_prefixes)
            for k in keys
        )
        has_cheap = any("CheapChannel" in k for k in keys if k.startswith("encoder."))
        if has_codec:
            variant: _VariantT = "large"
        elif has_cheap:
            variant = "middle"
        else:
            variant = "small"

        # Hyperencoder's first ResidualBlockShift goes M→N; its ``conv2`` is a
        # ``(N, M, 1, 1)`` 1×1 conv. Read N from there.
        N = int(state_dict["hyperencoder.0.conv2.weight"].size(0))

        # Encoder's last conv is the ``256*4 -> M`` 1×1; its index depends on
        # whether CheapCS1 modules are present (small omits two indices).
        last_encoder_conv_idx = 16 if variant == "small" else 18
        M = int(
            state_dict[f"encoder.{last_encoder_conv_idx}.weight"].size(0)
        )

        net = cls(variant=variant, N=N, M=M)
        net.load_state_dict(state_dict)
        return net
