"""Informer: Joint Global and Local Hierarchical Priors for Learned Image Compression.

Kim, Heo & Lee, `"Joint Global and Local Hierarchical Priors for Learned Image
Compression"
<https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Joint_Global_and_Local_Hierarchical_Priors_for_Learned_Image_Compression_CVPR_2022_paper.html>`_
(CVPR 2022).

Architecture: standard 4-stage Conv+GDN encoder/decoder from
:class:`JointAutoregressiveHierarchicalPriors` paired with two parallel
hyperprior branches:

* a *local* 1×1-conv hyperprior with channel reduction ``M → M/4 → M/16``
  and synthesis ``M/16 → M/2 → 2M`` (per-spatial-location params);
* a *global* hyperprior built from ``num_global`` learnable tokens that
  cross-attend to the encoded latent, are quantized through a dedicated
  entropy bottleneck, then broadcast through a parameter-model
  cross-attention to mix with the masked-conv spatial context.

Module / parameter names are preserved verbatim from upstream so checkpoints
load directly via :meth:`from_state_dict`.
"""

from __future__ import annotations

import warnings

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.layers.attn import CrossAttentionBlock
from compressai.models.base import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.registry import register_model

__all__ = ["Informer"]


def _img2seq(x: Tensor) -> Tuple[Tensor, torch.Size]:
    """Flatten ``(B, C, H, W) → (B, H*W, C)`` while preserving the original shape."""
    return rearrange(x, "b c h w -> b (h w) c"), x.shape


def _seq2img(x: Tensor, x_size: torch.Size) -> Tensor:
    """Inverse of :func:`_img2seq`."""
    return rearrange(x, "b (h w) c -> b c h w", h=x_size[2], w=x_size[3])


@register_model("informer")
class Informer(CompressionModel):
    """Informer end-to-end image compression model.

    Args:
        N: Backbone channel width (default 192).
        M: Latent (``y``) channels (default 192). Must be divisible by both
            ``num_global`` and 64 (the cross-attention head dim).
        num_global: Number of global tokens used by the global hyperprior
            (default 8). ``M`` must be divisible by ``num_global``.
    """

    def __init__(self, N: int = 192, M: int = 192, num_global: int = 8) -> None:
        super().__init__()

        if M % num_global != 0:
            raise ValueError(
                f"M ({M}) must be divisible by num_global ({num_global})"
            )
        if M % 64 != 0:
            raise ValueError(
                f"M ({M}) must be divisible by 64 (cross-attention head dim)"
            )

        self.N = int(N)
        self.M = int(M)
        self.num_global = int(num_global)

        # ----- Analysis (g_a) / Synthesis (g_s) -----
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        # ----- Local hyperprior model -----
        self.local_h_a = nn.Sequential(
            nn.Conv2d(M, M // 4, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M // 4, M // 16, 1),
        )
        self.local_h_s = nn.Sequential(
            nn.Conv2d(M // 16, M // 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M // 2, M * 2, 1),
        )
        self.local_entropy_bottleneck = EntropyBottleneck(M // 16)

        # ----- Global hyperprior model -----
        self.global_tokens = nn.Parameter(torch.randn(num_global, M))
        self.ca_a = CrossAttentionBlock(dim=M, num_heads=M // 64, qkv_bias=True)
        self.global_h_a = nn.Linear(M, M // num_global)
        self.global_h_s = nn.Linear(M // num_global, M * 2)
        self.global_entropy_bottleneck = EntropyBottleneck(M)

        # ----- Parameter model (fuses spatial context + global params) -----
        self.ca_s = CrossAttentionBlock(
            dim=M * 2, num_heads=M // 64 * 2, qkv_bias=True
        )

        # ----- Spatial context + entropy parameters -----
        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.gaussian_conditional = GaussianConditional(None)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    # ------------------------------------------------------------------
    def _global_hyperprior_encode(self, y: Tensor) -> Tensor:
        """Encoder side: ``num_global`` tokens cross-attend to ``y``, then
        project to ``M / num_global`` channels per token. Returns the
        ``(B, M, 1, 1)`` global ``z`` tensor consumed by the entropy
        bottleneck."""
        global_seq = repeat(self.global_tokens, "n c -> b n c", b=y.shape[0])
        y_seq, _ = _img2seq(y)
        global_seq = self.ca_a(global_seq, y_seq)
        global_seq = self.global_h_a(global_seq)
        global_seq = rearrange(global_seq, "b n c -> b (n c)")
        return global_seq.unsqueeze(2).unsqueeze(3)

    def _global_params_seq(self, global_z_hat: Tensor) -> Tensor:
        """Decoder side: lift quantized global ``z`` back to a token sequence
        of ``2M``-dim params for the parameter cross-attention."""
        flat = global_z_hat.squeeze(3).squeeze(2)
        seq = rearrange(flat, "b (n c) -> b n c", n=self.num_global)
        return self.global_h_s(seq)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Dict[str, Any]:
        y = self.g_a(x)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        # Local hyperprior.
        local_z = self.local_h_a(y)
        local_z_hat, local_z_likelihoods = self.local_entropy_bottleneck(local_z)
        local_params = self.local_h_s(local_z_hat)

        # Global hyperprior.
        global_z = self._global_hyperprior_encode(y)
        global_z_hat, global_z_likelihoods = self.global_entropy_bottleneck(global_z)
        global_params_seq = self._global_params_seq(global_z_hat)

        # Spatial autoregressive context.
        ctx_params = self.context_prediction(y_hat)

        # Parameter model: cross-attend ctx tokens against global params.
        ctx_params_seq, ctx_params_size = _img2seq(ctx_params)
        ctx_params_seq = self.ca_s(ctx_params_seq, global_params_seq)
        ctx_params = _seq2img(ctx_params_seq, ctx_params_size)

        gaussian_params = self.entropy_parameters(
            torch.cat((local_params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(
            y, scales_hat, means=means_hat
        )

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "l_z": local_z_likelihoods,
                "g_z": global_z_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "Informer":
        N = int(state_dict["g_a.0.weight"].size(0))
        M = int(state_dict["g_a.6.weight"].size(0))
        num_global = int(state_dict["global_tokens"].size(0))
        net = cls(N=N, M=M, num_global=num_global)
        net.load_state_dict(state_dict)
        return net

    # ------------------------------------------------------------------
    # Sequential autoregressive bitstream coding (slow CPU path; same
    # structure as ``JointAutoregressiveHierarchicalPriors._compress_ar``
    # but per-pixel ``ctx_p`` is mixed with the global params via the
    # parameter cross-attention before feeding the entropy network).
    # ------------------------------------------------------------------
    def compress(self, x: Tensor) -> Dict[str, Any]:
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.g_a(x)

        # Local hyperprior round-trip.
        local_z = self.local_h_a(y)
        local_z_strings = self.local_entropy_bottleneck.compress(local_z)
        local_z_hat = self.local_entropy_bottleneck.decompress(
            local_z_strings, local_z.size()[-2:]
        )
        local_params = self.local_h_s(local_z_hat)

        # Global hyperprior round-trip.
        global_z = self._global_hyperprior_encode(y)
        global_z_strings = self.global_entropy_bottleneck.compress(global_z)
        global_z_hat = self.global_entropy_bottleneck.decompress(
            global_z_strings, global_z.size()[-2:]
        )

        kernel_size = 5
        padding = kernel_size // 2

        # local_z is at the same resolution as y (s=1 pooling between them).
        y_height = local_z_hat.size(2)
        y_width = local_z_hat.size(3)
        y_hat_padded = F.pad(y, (padding, padding, padding, padding))

        y_strings: List[bytes] = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat_padded[i : i + 1],
                local_params[i : i + 1],
                global_z_hat[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {
            "strings": [y_strings, local_z_strings, global_z_strings],
            "shape": local_z.size()[-2:],
        }

    def _compress_ar(
        self,
        y_hat: Tensor,
        local_params: Tensor,
        global_z_hat: Tensor,
        height: int,
        width: int,
        kernel_size: int,
        padding: int,
    ) -> bytes:
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list: List[int] = []
        indexes_list: List[int] = []

        global_params_seq = self._global_params_seq(global_z_hat)

        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                local_p = local_params[:, :, h : h + 1, w : w + 1]

                # Parameter model on the single-pixel ctx_p.
                ctx_p_seq, ctx_p_size = _img2seq(ctx_p)
                ctx_p_seq = self.ca_s(ctx_p_seq, global_params_seq)
                ctx_p = _seq2img(ctx_p_seq, ctx_p_size)

                gaussian_params = self.entropy_parameters(
                    torch.cat((local_p, ctx_p), dim=1)
                )
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(
                    y_crop, "symbols", means_hat
                )
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().int().tolist())
                indexes_list.extend(indexes.squeeze().int().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        return encoder.flush()

    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        assert isinstance(strings, list) and len(strings) == 3

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        local_z_hat = self.local_entropy_bottleneck.decompress(strings[1], shape)
        local_params = self.local_h_s(local_z_hat)

        global_z_hat = self.global_entropy_bottleneck.decompress(strings[2], (1, 1))

        kernel_size = 5
        padding = kernel_size // 2
        y_height = local_z_hat.size(2)
        y_width = local_z_hat.size(3)

        y_hat = torch.zeros(
            (
                local_z_hat.size(0),
                self.M,
                y_height + 2 * padding,
                y_width + 2 * padding,
            ),
            device=local_z_hat.device,
            dtype=local_z_hat.dtype,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                local_params[i : i + 1],
                global_z_hat[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self,
        y_string: bytes,
        y_hat: Tensor,
        local_params: Tensor,
        global_z_hat: Tensor,
        height: int,
        width: int,
        kernel_size: int,
        padding: int,
    ) -> None:
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        global_params_seq = self._global_params_seq(global_z_hat)

        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                local_p = local_params[:, :, h : h + 1, w : w + 1]

                ctx_p_seq, ctx_p_size = _img2seq(ctx_p)
                ctx_p_seq = self.ca_s(ctx_p_seq, global_params_seq)
                ctx_p = _seq2img(ctx_p_seq, ctx_p_size)

                gaussian_params = self.entropy_parameters(
                    torch.cat((local_p, ctx_p), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().int().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.tensor(rv, dtype=y_hat.dtype).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
