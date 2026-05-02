"""DAMO Academy image-compression models.

Two models from the same DAMO Academy team (Yichen Qian, Ming Lin, Xiuyu Sun,
Zhiyu Tan, Rong Jin, et al.) share this file. They both use a Ballé-style 4-stage
Conv encoder/decoder pair with different normalisation choices and entropy
models:

* :class:`Entroformer` -- ICLR 2022, ``arXiv:2202.05492``. Conv+GDN backbone with
  a Transformer-based hyperprior + auto-regressive entropy model.
* :class:`RefBasedAR` -- ICLR 2021, ``arXiv:2010.08321``. Conv+GSDN backbone with
  a 3-cascade auto-regressive entropy model that combines local masked-conv
  context, a global reference search and a hyperprior-derived feature into a
  K=3 Gaussian mixture likelihood.

The Ballé-style transforms (``_Balle2Encoder`` / ``_Balle2Decoder`` /
``_Balle2Upsample``) and the ref-AR hyperprior (``_RefHyperEncoder`` /
``_RefHyperDecoder``) are private to this file -- both models are the only
consumers. Module attribute names (``encoder`` / ``decoder``, ``transpose``,
``mu`` / ``beta`` / ...) are kept identical to upstream so converted
checkpoints load 1:1.
"""
from __future__ import annotations

import math

from typing import Any, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import (
    GaussianConditional,
    GaussianMixtureConditional,
    LearnedGaussianBottleneck,
)
from compressai.latent_codecs import (
    RefAutoregressiveLatentCodec,
    TransformerARLatentCodec,
)
from compressai.layers.attn import TransDecoder, TransHyperScale
from compressai.layers.gdn import GDN
from compressai.ops.parametrizers import NonNegativeParametrizer
from compressai.registry import register_model

from .base import CompressionModel

__all__ = ["Entroformer", "RefBasedAR"]


# ---------------------------------------------------------------------------
# Shared transforms (formerly compressai/layers/lic/balle2.py + ref_hyper.py)
# ---------------------------------------------------------------------------


class _Balle2Upsample(nn.Module):
    """``ConvTranspose2d`` wrapper preserving the upstream ``transpose`` attribute name."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        output_padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            padding_mode="zeros",
            groups=groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transpose(x)


class _Balle2Encoder(nn.Module):
    """4-stage 5x5 stride-2 Conv + GDN/GSDN analysis transform."""

    def __init__(
        self,
        channels: int = 192,
        last_channels: int = 384,
        norm_cls: Type[nn.Module] = GDN,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 5, stride=2, padding=2, padding_mode="zeros"),
            norm_cls(channels),
            nn.Conv2d(channels, channels, 5, stride=2, padding=2, padding_mode="zeros"),
            norm_cls(channels),
            nn.Conv2d(channels, channels, 5, stride=2, padding=2, padding_mode="zeros"),
            norm_cls(channels),
            nn.Conv2d(channels, last_channels, 5, stride=2, padding=2, padding_mode="zeros"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class _Balle2Decoder(nn.Module):
    """4-stage 5x5 stride-2 ConvTranspose + inverse-GDN/GSDN synthesis transform."""

    def __init__(
        self,
        channels: int = 192,
        last_channels: int = 384,
        norm_cls: Type[nn.Module] = GDN,
    ) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            _Balle2Upsample(last_channels, channels, 5, stride=2, padding=2, output_padding=1),
            norm_cls(channels, inverse=True),
            _Balle2Upsample(channels, channels, 5, stride=2, padding=2, output_padding=1),
            norm_cls(channels, inverse=True),
            _Balle2Upsample(channels, channels, 5, stride=2, padding=2, output_padding=1),
            norm_cls(channels, inverse=True),
            _Balle2Upsample(channels, 3, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class _RefHyperEncoder(nn.Module):
    """``y -> z`` analysis: 3x3 + 5x5 stride-2 + 5x5 stride-2 (3-stage)."""

    def __init__(
        self, in_channel: int = 384, out_channel: int = 192, channel: int = 192
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, stride=1, padding=1, padding_mode="zeros"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel, 5, stride=2, padding=2, padding_mode="zeros"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, out_channel, 5, stride=2, padding=2, padding_mode="zeros"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class _RefHyperDecoder(nn.Module):
    """``z_hat -> z_feature`` synthesis: 5x5 deconv x2 + 3x3 conv + 1x1 head."""

    def __init__(
        self, in_channel: int = 192, out_channel: int = 768, channel: int = 192
    ) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            _Balle2Upsample(in_channel, channel, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _Balle2Upsample(channel, channel, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode="zeros"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, out_channel, 1, stride=1, padding=0, padding_mode="zeros"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


# ---------------------------------------------------------------------------
# GSDN (used only by RefBasedAR)
# ---------------------------------------------------------------------------


class _GSDN(nn.Module):
    """Generalized Subtractive + Divisive Normalisation layer.

    GSDN extends GDN with a per-channel learnable subtractive (mean) term
    *before* the divisive (variance) normalisation::

        y[i] = (x[i] - mu[i]) / sqrt(beta[i] + sum_j gamma[j, i] * (x[j] - mu[j])**2)

    Only the divisive ``beta`` / ``gamma`` are reparametrised through
    :class:`compressai.ops.parametrizers.NonNegativeParametrizer`; the
    subtractive ``beta2`` / ``gamma2`` follow the same reparametrisation but
    with a different positivity floor on ``beta2`` (zero by default upstream).

    Module attribute names ``beta`` / ``gamma`` / ``beta2`` / ``gamma2`` and
    init values are kept identical to upstream ``module/ops.py::GSDN`` so the
    released checkpoints load 1:1.
    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.inverse = bool(inverse)

        # Divisive (beta, gamma): identical to GDN.
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

        # Subtractive (beta2, gamma2): upstream initialises beta2 to zeros.
        self.beta2_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta2 = torch.zeros(in_channels)
        beta2 = self.beta2_reparam.init(beta2)
        self.beta2 = nn.Parameter(beta2)

        self.gamma2_reparam = NonNegativeParametrizer()
        gamma2 = gamma_init * torch.eye(in_channels)
        gamma2 = self.gamma2_reparam.init(gamma2)
        self.gamma2 = nn.Parameter(gamma2)

    def _norm_params(self, beta_p, gamma_p, beta_repar, gamma_repar, C):
        beta = beta_repar(beta_p)
        gamma = gamma_repar(gamma_p)
        gamma = gamma.reshape(C, C, 1, 1)
        return beta, gamma

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        if self.inverse:
            # Decoder side: inverse divisive (multiply) then add learned mean.
            beta, gamma = self._norm_params(
                self.beta, self.gamma, self.beta_reparam, self.gamma_reparam, C
            )
            norm = torch.sqrt(F.conv2d(x**2, gamma, beta))
            x = x * norm

            beta2, gamma2 = self._norm_params(
                self.beta2, self.gamma2, self.beta2_reparam, self.gamma2_reparam, C
            )
            mean = F.conv2d(x, gamma2, beta2)
            return x + mean

        # Encoder side: subtract learned mean then divisive normalise.
        beta2, gamma2 = self._norm_params(
            self.beta2, self.gamma2, self.beta2_reparam, self.gamma2_reparam, C
        )
        mean = F.conv2d(x, gamma2, beta2)
        x = x - mean

        beta, gamma = self._norm_params(
            self.beta, self.gamma, self.beta_reparam, self.gamma_reparam, C
        )
        norm = torch.rsqrt(F.conv2d(x**2, gamma, beta))
        return x * norm


# ---------------------------------------------------------------------------
# Entroformer (ICLR 2022)
# ---------------------------------------------------------------------------


def _make_entroformer_param_net(
    dim_embed: int, mlp_ratio: int, latent_channels: int, num_parameter: int, K: int
) -> nn.Sequential:
    """1x1 Conv -> LeakyReLU(0.2) -> 1x1 Conv head; matches upstream ``cit_pn``."""
    inner = dim_embed * mlp_ratio
    out = latent_channels * K * num_parameter
    return nn.Sequential(
        nn.Conv2d(dim_embed * 2, inner, 1, 1, 0),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(inner, out, 1, 1, 0),
    )


@register_model("entroformer")
class Entroformer(CompressionModel):
    r"""Entroformer (Qian et al., ICLR 2022, `arXiv:2202.05492`_).

    Ballé-style 4-stage Conv+GDN auto-encoder with a Transformer-based
    hyperprior + auto-regressive entropy model:

    * ``g_a / g_s`` -- Ballé-style 4x stride-2 5x5 Conv + GDN, channels ``N -> M``.
    * ``cit_he`` -- :class:`compressai.layers.attn.TransHyperScale` ``(down=True)``,
      three Transformer stages alternated with two stride-2 Conv down-samplers
      producing the hyper latent ``z`` of shape ``(B, Z, H/16, W/16)``.
    * ``cit_hd`` -- :class:`TransHyperScale` ``(down=False)`` with sub-pixel
      up-samplers, producing a feature map of shape ``(B, dim_embed, H/4, W/4)``.
    * ``cit_ar`` -- :class:`compressai.layers.attn.TransDecoder`, raster causal
      Transformer over the noise-quantised ``y``.
    * ``cit_pn`` -- 1x1 Conv MLP that fuses ``[feat_hyper, feat_ar]`` and outputs
      ``num_parameter * M`` channels split into per-pixel scales / means for the
      Gaussian conditional likelihood of ``y``.

    The hyperprior is modelled with
    :class:`compressai.entropy_models.LearnedGaussianBottleneck` (per-channel
    single Gaussian, equivalent to upstream ``module/prob_model.py::Entropy``).

    Inputs are expected in ``[0, 1]``; the model internally rescales to
    ``[-1, 1]`` for ``g_a`` and back to ``[0, 1]`` after ``g_s`` to match upstream
    training data normalisation, so released checkpoints load 1:1 by parameter
    name (handled by :func:`examples/convert_entroformer_checkpoint.py`).

    Args:
        N: ``g_a`` intermediate channel count (upstream ``channels``, default 192).
        M: latent ``y`` channel count (upstream ``last_channels``, default 384).
        Z: hyper latent ``z`` channel count (upstream ``hyper_channels``,
            default 192).
        dim_embed: Transformer embedding dim (default 384).
        depth: Number of Transformer layers in ``cit_ar`` (default 6).
        heads: Attention heads in each Transformer block (default 6).
        dim_head: Per-head dim (default 64).
        mlp_ratio: FFN expansion (default 4).
        position_num: 2D-RPE bucket count for the AR Transformer; ``cit_he``
            stages halve it down stage-by-stage (default 7).
        scale: Number of down/up scaling stages in ``cit_he`` / ``cit_hd``
            (default 2).
        attn_topk: Top-k filter for self-attention; ``-1`` disables (default 32
            for the released ``unidirectional`` configuration).
        num_parameter: Number of distribution parameters per pixel; supported
            values are ``2`` (mean + log-sigma) and ``3`` (logit_pi + mean + log-sigma
            with ``K=1``).
        K: Number of mixture components. Only ``K=1`` is supported.
        rpe_shared: Share the 2D-RPE table across Transformer layers (the
            non-zeroth layers reuse the position bias computed by layer 0;
            default True).

    .. _arXiv:2202.05492: https://arxiv.org/abs/2202.05492
    """

    g_a: nn.Module
    g_s: nn.Module
    latent_codec: TransformerARLatentCodec

    def __init__(
        self,
        N: int = 192,
        M: int = 384,
        Z: int = 192,
        *,
        dim_embed: int = 384,
        depth: int = 6,
        heads: int = 6,
        dim_head: int = 64,
        mlp_ratio: int = 4,
        position_num: int = 7,
        scale: int = 2,
        attn_topk: int = 32,
        num_parameter: int = 2,
        K: int = 1,
        rpe_shared: bool = True,
        att_scale: bool = True,
        dropout: float = 0.0,
        mask_ratio: float = 0.0,
        log_scale_min: float = -7.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if K != 1:
            raise ValueError("Entroformer migration currently supports K=1 only.")

        self.N = int(N)
        self.M = int(M)
        self.Z = int(Z)

        self.g_a = _Balle2Encoder(channels=N, last_channels=M)
        self.g_s = _Balle2Decoder(channels=N, last_channels=M)

        cit_he = TransHyperScale(
            cin=M,
            cout=Z,
            scale=scale,
            down=True,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            position_num=position_num,
            attn_topk=attn_topk,
            att_scale=att_scale,
            rpe_shared=rpe_shared,
            mask_ratio=0.0,
        )
        cit_hd = TransHyperScale(
            cin=Z,
            cout=0,
            scale=scale,
            down=False,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            position_num=position_num,
            attn_topk=attn_topk,
            att_scale=att_scale,
            rpe_shared=rpe_shared,
            mask_ratio=0.0,
        )
        cit_ar = TransDecoder(
            cin=M,
            cout=0,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            position_num=position_num,
            attn_topk=attn_topk,
            att_scale=att_scale,
            rpe_shared=rpe_shared,
            mask_ratio=mask_ratio,
        )
        cit_pn = _make_entroformer_param_net(dim_embed, mlp_ratio, M, num_parameter, K)

        self.latent_codec = TransformerARLatentCodec(
            latent_channels=M,
            hyper_channels=Z,
            y_hyper_encode=cit_he,
            y_hyper_decode=cit_hd,
            y_ar=cit_ar,
            param_net=cit_pn,
            entropy_bottleneck=LearnedGaussianBottleneck(Z),
            gaussian_conditional=GaussianConditional(
                None,
                scale_bound=float(math.exp(log_scale_min)),
            ),
            num_parameter=num_parameter,
            log_scale_min=log_scale_min,
        )

        # Stash hyperparameters for downstream introspection / `from_state_dict`.
        self._hparams = dict(
            N=N,
            M=M,
            Z=Z,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            position_num=position_num,
            scale=scale,
            attn_topk=attn_topk,
            num_parameter=num_parameter,
            K=K,
            rpe_shared=rpe_shared,
            att_scale=att_scale,
        )

    @property
    def downsampling_factor(self) -> int:
        # 4 stride-2 conv stages in g_a, plus `scale` extra in cit_he.
        return 2 ** (4 + self._hparams["scale"])

    def forward(self, x: Tensor) -> Dict[str, Any]:
        # Upstream rescales the image to [-1, 1] before g_a; mirror that here so
        # the released checkpoints reproduce the trained reconstructions exactly.
        x_in = x * 2.0 - 1.0
        y = self.g_a(x_in)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)
        x_hat = x_hat / 2.0 + 0.5
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "Entroformer":
        """Instantiate ``Entroformer`` from a converted compressai-layout state dict.

        Hyperparameters are inferred from tensor shapes:

        * ``N`` from ``g_a.encoder.0.weight`` (out channels of the first conv);
        * ``M`` from ``g_a.encoder.6.weight``;
        * ``Z`` from ``latent_codec.entropy_bottleneck.mu``;
        * ``dim_embed`` from ``latent_codec.y_ar.to_patch_embedding.weight``
          (out features);
        * ``depth`` from the count of ``latent_codec.y_ar.blocks.{i}.layer.0...``
          entries;
        * ``heads`` / ``dim_head`` from ``...SelfAttention.qkv.weight``
          (3 * heads * dim_head, dim_embed);
        * ``position_num`` from ``...SelfAttention.relative_attention_bias.weight``
          (rows = ``position_num**2``);
        * ``scale`` from ``latent_codec.y_hyper_encode.scale_blocks.{i}``;
        * ``num_parameter`` from ``latent_codec.param_net.2.weight`` (out
          channels = ``M * K * num_parameter``).
        """
        N = state_dict["g_a.encoder.0.weight"].size(0)
        M = state_dict["g_a.encoder.6.weight"].size(0)
        Z = state_dict["latent_codec.entropy_bottleneck.mu"].size(1)

        ar_embed_w = state_dict["latent_codec.y_ar.to_patch_embedding.weight"]
        dim_embed = ar_embed_w.size(0)

        depth = sum(
            1
            for k in state_dict
            if k.startswith("latent_codec.y_ar.blocks.")
            and k.endswith(".layer.0.SelfAttention.qkv.weight")
        )

        qkv_w = state_dict["latent_codec.y_ar.blocks.0.layer.0.SelfAttention.qkv.weight"]
        # qkv_w: (3 * heads * dim_head, dim_embed)
        rpe_w = state_dict[
            "latent_codec.y_ar.blocks.0.layer.0.SelfAttention.relative_attention_bias.weight"
        ]
        position_num = int(round(rpe_w.size(0) ** 0.5))
        dim_head = rpe_w.size(1)
        heads = qkv_w.size(0) // (3 * dim_head)

        # mlp_ratio from FFN inner-dim (`net.0.weight` out features = dim_embed * mlp_ratio)
        ffn_w = state_dict["latent_codec.y_ar.blocks.0.layer.1.fn.net.0.weight"]
        mlp_ratio = ffn_w.size(0) // dim_embed

        # `scale` from number of scale_blocks in the hyper-encoder.
        scale = sum(
            1
            for k in state_dict
            if k.startswith("latent_codec.y_hyper_encode.scale_blocks.")
            and k.endswith(".weight")
        )

        # num_parameter: cit_pn.2 out channels = M * K * num_parameter, with K=1.
        pn_out = state_dict["latent_codec.param_net.2.weight"].size(0)
        if pn_out % M != 0:
            raise ValueError(
                f"param_net output channels ({pn_out}) is not divisible by M ({M})"
            )
        num_parameter = pn_out // M

        net = cls(
            N=N,
            M=M,
            Z=Z,
            dim_embed=dim_embed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            position_num=position_num,
            scale=scale,
            num_parameter=num_parameter,
            K=1,
        )
        net.load_state_dict(state_dict)
        return net

    def compress(self, x: Tensor):
        raise NotImplementedError(
            "Entroformer.compress is not implemented yet; the upstream raster-scan "
            "AR bitstream coding requires per-pixel re-evaluation of the Transformer "
            "AR head and is a planned follow-up. Use forward() for rate estimation."
        )

    def decompress(self, *args, **kwargs):
        raise NotImplementedError(
            "Entroformer.decompress is not implemented yet; see compress()."
        )


# ---------------------------------------------------------------------------
# Reference-Based AR (ICLR 2021)
# ---------------------------------------------------------------------------


@register_model("qian2021-ref")
class RefBasedAR(CompressionModel):
    r"""Reference-Based AR (Qian et al., ICLR 2021, `arXiv:2010.08321`_).

    Ballé-style 4-stage Conv+GSDN auto-encoder with a 3-conv hyper-prior and a
    3-cascade auto-regressive entropy model that combines a local masked-conv
    context, a global reference search and a hyperprior-derived feature into a
    discretised K=3 Gaussian mixture likelihood.

    Args:
        N: ``g_a`` intermediate channel count (upstream ``channels``,
            default 192).
        M: latent ``y`` channel count (upstream ``last_channels``, default
            384).
        Z: hyper latent ``z`` channel count (upstream ``hyper_channels``,
            default 192).
        norm: Either ``"GSDN"`` (default, matches the released MSE config) or
            ``"GDN"`` (compressai's standard GDN; mathematically the
            divisive-only variant).
        sk: Reference patch size for the global-reference search (default 3,
            matches upstream).
        head_channels: Hidden width of the three 1x1 cascade heads inside the
            AR codec (upstream ``channels = last_channels * 3``;
            default ``3 * M``).
        num_parameter: Number of distribution parameters per pixel; only
            ``3`` (logit_pi + mean + log-sigma) is supported.
        log_scale_min: Lower clamp on predicted ``log sigma``. Upstream sets
            this to ``-7`` (sigma >= 9e-4).

    .. _arXiv:2010.08321: https://arxiv.org/abs/2010.08321
    """

    g_a: nn.Module
    g_s: nn.Module

    def __init__(
        self,
        N: int = 192,
        M: int = 384,
        Z: int = 192,
        *,
        norm: str = "GSDN",
        sk: int = 3,
        head_channels: int | None = None,
        num_parameter: int = 3,
        log_scale_min: float = -7.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if num_parameter != 3:
            raise ValueError(
                "RefBasedAR migration supports num_parameter=3 only "
                "(matches the released MSE config)."
            )
        if norm not in ("GSDN", "GDN"):
            raise ValueError(f"unknown norm {norm!r}; expected GSDN or GDN")

        norm_cls = _GSDN if norm == "GSDN" else GDN

        self.N = int(N)
        self.M = int(M)
        self.Z = int(Z)
        self.norm = norm
        self.sk = int(sk)
        self.head_channels = (
            int(head_channels) if head_channels is not None else 3 * M
        )

        self.g_a = _Balle2Encoder(channels=N, last_channels=M, norm_cls=norm_cls)
        self.g_s = _Balle2Decoder(channels=N, last_channels=M, norm_cls=norm_cls)
        self.h_a = _RefHyperEncoder(in_channel=M, out_channel=Z, channel=N)
        self.h_s = _RefHyperDecoder(in_channel=Z, out_channel=2 * M, channel=N)

        self.entropy_bottleneck = LearnedGaussianBottleneck(Z)
        self.latent_codec = RefAutoregressiveLatentCodec(
            latent_channels=M,
            hyper_channels=Z,
            head_channels=self.head_channels,
            sk=self.sk,
            num_parameter=num_parameter,
            log_scale_min=log_scale_min,
            entropy_bottleneck=self.entropy_bottleneck,
        )

        self._hparams = dict(
            N=N,
            M=M,
            Z=Z,
            norm=norm,
            sk=sk,
            head_channels=self.head_channels,
            num_parameter=num_parameter,
        )

    @property
    def downsampling_factor(self) -> int:
        # 4 stride-2 stages in g_a + 2 stride-2 stages in h_a.
        return 2 ** (4 + 2)

    def forward(self, x: Tensor) -> Dict[str, Any]:
        # Upstream's ref model trains directly on [0, 1] images (g_a takes the
        # image as-is, g_s outputs in [0, 1] before clamp), so no [-1, 1] shell
        # like Entroformer.
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_feature = self.h_s(z_hat)
        codec_out = self.latent_codec(y, z_feature)
        x_hat = self.g_s(codec_out["y_hat"])
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": codec_out["likelihoods"]["y"],
                "z": z_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "RefBasedAR":
        """Instantiate ``RefBasedAR`` from a converted compressai-layout state dict.

        ``N`` is inferred from ``g_a.encoder.0.weight`` (out channels), ``M``
        from ``g_a.encoder.6.weight`` and ``Z`` from
        ``entropy_bottleneck.mu``. The presence of ``encoder.1.beta2`` (i.e.
        the GSDN subtractive parameter) selects ``norm='GSDN'`` versus
        ``norm='GDN'``.
        """
        N = state_dict["g_a.encoder.0.weight"].size(0)
        M = state_dict["g_a.encoder.6.weight"].size(0)
        Z = state_dict["entropy_bottleneck.mu"].size(1)
        norm = "GSDN" if "g_a.encoder.1.beta2" in state_dict else "GDN"

        # Infer head_channels from the cascade-1 head's hidden width.
        # `latent_codec.conv_1x1_1.0.weight` shape: (head, 2*M, 1, 1) -> head.
        head_w = state_dict["latent_codec.conv_1x1_1.0.weight"]
        head_channels = head_w.size(0)

        # `mask_conv_ref.weight` shape (2M, M, sk, sk) -> infer sk.
        sk = state_dict["latent_codec.mask_conv_ref.weight"].size(-1)

        net = cls(
            N=N,
            M=M,
            Z=Z,
            norm=norm,
            sk=sk,
            head_channels=head_channels,
            num_parameter=3,
        )
        net.load_state_dict(state_dict)
        return net

    def compress(self, x: Tensor):
        raise NotImplementedError(
            "RefBasedAR.compress is not implemented yet; the upstream raster-scan AR "
            "bitstream coding requires per-pixel re-evaluation of the search/cascade "
            "heads, and is a planned follow-up. Use forward() for rate estimation."
        )

    def decompress(self, *args, **kwargs):
        raise NotImplementedError(
            "RefBasedAR.decompress is not implemented yet; see compress()."
        )
