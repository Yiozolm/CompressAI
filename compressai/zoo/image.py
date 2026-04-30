# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from torch.hub import load_state_dict_from_url

from compressai.layers import is_pytorch_wavelets_available
from compressai.models import (
    CCAModel,
    CMIC,
    Cheng2020Anchor,
    Cheng2020Attention,
    DCAE,
    Entroformer,
    FactorizedPrior,
    FactorizedPriorReLU,
    FrequencyAwareTransFormer,
    GainedMSHyperprior,
    GainedScaleHyperprior,
    GLIC,
    HPCM,
    Informer,
    InvCompress,
    MambaIC,
    MambaVC,
    JointAutoregressiveHierarchicalPriors,
    MLICPlusPlus,
    MeanScaleHyperprior,
    RefBasedAR,
    SAAF,
    SCGainedMSHyperprior,
    ScaleHyperprior,
    ShiftLIC,
    SymmetricalTransFormer,
    TCM,
    TIC,
    TinyLIC,
    WeConvene,
    WACNN,
    is_freia_available,
)

from .pretrained import load_pretrained

__all__ = [
    "bmshj2018_factorized",
    "bmshj2018_factorized_relu",
    "bmshj2018_hyperprior",
    "bmshj2018_hyperprior_gained",
    "cca",
    "cmic",
    "dcae",
    "entroformer",
    "ftic",
    "glic",
    "hpcm",
    "hpcm_base",
    "hpcm_large",
    "hpcm_phi",
    "informer",
    "invcompress",
    "mambaic",
    "mambavc",
    "mlicpp",
    "qian2021_ref",
    "saaf",
    "shiftlic_small",
    "shiftlic_middle",
    "shiftlic_large",
    "stf",
    "stf_wacnn",
    "tcm",
    "tic",
    "tinylic",
    "weconvene",
    "mbt2018",
    "mbt2018_mean",
    "mbt2018_mean_gained",
    "mbt2018_mean_gained_sc",
    "cheng2020_anchor",
    "cheng2020_attn",
    "candidate_model_architectures",
    "candidate_model_urls",
]

model_architectures = {
    "bmshj2018-factorized": FactorizedPrior,
    "bmshj2018_factorized_relu": FactorizedPriorReLU,
    "bmshj2018-hyperprior": ScaleHyperprior,
    "mbt2018-mean": MeanScaleHyperprior,
    "mbt2018": JointAutoregressiveHierarchicalPriors,
    "cheng2020-anchor": Cheng2020Anchor,
    "cheng2020-attn": Cheng2020Attention,
}

candidate_model_architectures = {
    "glic": GLIC,
    "cmic": CMIC if is_pytorch_wavelets_available() else None,
    "mlicpp": MLICPlusPlus,
    "qian2021-ref": RefBasedAR,
    "stf-wacnn": WACNN,
    "stf": SymmetricalTransFormer,
    "lic-tcm": TCM,
    "tcm": TCM,
    "dcae": DCAE,
    "entroformer": Entroformer,
    "saaf": SAAF,
    "weconvene": WeConvene if is_pytorch_wavelets_available() else None,
    "ftic": FrequencyAwareTransFormer,
    "hpcm": HPCM,
    "hpcm-base": HPCM,
    "hpcm-large": HPCM,
    "hpcm-phi": HPCM,
    "informer": Informer,
    "invcompress": InvCompress if is_freia_available() else None,
    "cca": CCAModel,
    "mambaic": MambaIC,
    "mambavc": MambaVC,
    "tinylic": TinyLIC,
    "tic": TIC,
    "shiftlic-small": ShiftLIC,
    "shiftlic-middle": ShiftLIC,
    "shiftlic-large": ShiftLIC,
    "bmshj2018-hyperprior-gained": GainedScaleHyperprior,
    "mbt2018-mean-gained": GainedMSHyperprior,
    "mbt2018-mean-gained-sc": SCGainedMSHyperprior,
}

root_url = "https://compressai.s3.amazonaws.com/models/v1"
model_urls = {
    "bmshj2018-factorized": {
        "mse": {
            1: f"{root_url}/bmshj2018-factorized-prior-1-446d5c7f.pth.tar",
            2: f"{root_url}/bmshj2018-factorized-prior-2-87279a02.pth.tar",
            3: f"{root_url}/bmshj2018-factorized-prior-3-5c6f152b.pth.tar",
            4: f"{root_url}/bmshj2018-factorized-prior-4-1ed4405a.pth.tar",
            5: f"{root_url}/bmshj2018-factorized-prior-5-866ba797.pth.tar",
            6: f"{root_url}/bmshj2018-factorized-prior-6-9b02ea3a.pth.tar",
            7: f"{root_url}/bmshj2018-factorized-prior-7-6dfd6734.pth.tar",
            8: f"{root_url}/bmshj2018-factorized-prior-8-5232faa3.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/bmshj2018-factorized-ms-ssim-1-9781d705.pth.tar",
            2: f"{root_url}/bmshj2018-factorized-ms-ssim-2-4a584386.pth.tar",
            3: f"{root_url}/bmshj2018-factorized-ms-ssim-3-5352f123.pth.tar",
            4: f"{root_url}/bmshj2018-factorized-ms-ssim-4-4f91b847.pth.tar",
            5: f"{root_url}/bmshj2018-factorized-ms-ssim-5-b3a88897.pth.tar",
            6: f"{root_url}/bmshj2018-factorized-ms-ssim-6-ee028763.pth.tar",
            7: f"{root_url}/bmshj2018-factorized-ms-ssim-7-8c265a29.pth.tar",
            8: f"{root_url}/bmshj2018-factorized-ms-ssim-8-8811bd14.pth.tar",
        },
    },
    "bmshj2018-hyperprior": {
        "mse": {
            1: f"{root_url}/bmshj2018-hyperprior-1-7eb97409.pth.tar",
            2: f"{root_url}/bmshj2018-hyperprior-2-93677231.pth.tar",
            3: f"{root_url}/bmshj2018-hyperprior-3-6d87be32.pth.tar",
            4: f"{root_url}/bmshj2018-hyperprior-4-de1b779c.pth.tar",
            5: f"{root_url}/bmshj2018-hyperprior-5-f8b614e1.pth.tar",
            6: f"{root_url}/bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
            7: f"{root_url}/bmshj2018-hyperprior-7-3804dcbd.pth.tar",
            8: f"{root_url}/bmshj2018-hyperprior-8-a583f0cf.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/bmshj2018-hyperprior-ms-ssim-1-5cf249be.pth.tar",
            2: f"{root_url}/bmshj2018-hyperprior-ms-ssim-2-1ff60d1f.pth.tar",
            3: f"{root_url}/bmshj2018-hyperprior-ms-ssim-3-92dd7878.pth.tar",
            4: f"{root_url}/bmshj2018-hyperprior-ms-ssim-4-4377354e.pth.tar",
            5: f"{root_url}/bmshj2018-hyperprior-ms-ssim-5-c34afc8d.pth.tar",
            6: f"{root_url}/bmshj2018-hyperprior-ms-ssim-6-3a6d8229.pth.tar",
            7: f"{root_url}/bmshj2018-hyperprior-ms-ssim-7-8747d3bc.pth.tar",
            8: f"{root_url}/bmshj2018-hyperprior-ms-ssim-8-cc15b5f3.pth.tar",
        },
    },
    "mbt2018-mean": {
        "mse": {
            1: f"{root_url}/mbt2018-mean-1-e522738d.pth.tar",
            2: f"{root_url}/mbt2018-mean-2-e54a039d.pth.tar",
            3: f"{root_url}/mbt2018-mean-3-723404a8.pth.tar",
            4: f"{root_url}/mbt2018-mean-4-6dba02a3.pth.tar",
            5: f"{root_url}/mbt2018-mean-5-d504e8eb.pth.tar",
            6: f"{root_url}/mbt2018-mean-6-a19628ab.pth.tar",
            7: f"{root_url}/mbt2018-mean-7-d5d441d1.pth.tar",
            8: f"{root_url}/mbt2018-mean-8-8089ae3e.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/mbt2018-mean-ms-ssim-1-5bf9c0b6.pth.tar",
            2: f"{root_url}/mbt2018-mean-ms-ssim-2-e2a1bf3f.pth.tar",
            3: f"{root_url}/mbt2018-mean-ms-ssim-3-640ce819.pth.tar",
            4: f"{root_url}/mbt2018-mean-ms-ssim-4-12626c13.pth.tar",
            5: f"{root_url}/mbt2018-mean-ms-ssim-5-1be7f059.pth.tar",
            6: f"{root_url}/mbt2018-mean-ms-ssim-6-b83bf379.pth.tar",
            7: f"{root_url}/mbt2018-mean-ms-ssim-7-ddf9644c.pth.tar",
            8: f"{root_url}/mbt2018-mean-ms-ssim-8-0cc7b94f.pth.tar",
        },
    },
    "mbt2018": {
        "mse": {
            1: f"{root_url}/mbt2018-1-3f36cd77.pth.tar",
            2: f"{root_url}/mbt2018-2-43b70cdd.pth.tar",
            3: f"{root_url}/mbt2018-3-22901978.pth.tar",
            4: f"{root_url}/mbt2018-4-456e2af9.pth.tar",
            5: f"{root_url}/mbt2018-5-b4a046dd.pth.tar",
            6: f"{root_url}/mbt2018-6-7052e5ea.pth.tar",
            7: f"{root_url}/mbt2018-7-8ba2bf82.pth.tar",
            8: f"{root_url}/mbt2018-8-dd0097aa.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/mbt2018-ms-ssim-1-2878436b.pth.tar",
            2: f"{root_url}/mbt2018-ms-ssim-2-c41cb208.pth.tar",
            3: f"{root_url}/mbt2018-ms-ssim-3-d0dd64e8.pth.tar",
            4: f"{root_url}/mbt2018-ms-ssim-4-a120e037.pth.tar",
            5: f"{root_url}/mbt2018-ms-ssim-5-9b30e3b7.pth.tar",
            6: f"{root_url}/mbt2018-ms-ssim-6-f8b3626f.pth.tar",
            7: f"{root_url}/mbt2018-ms-ssim-7-16e6ff50.pth.tar",
            8: f"{root_url}/mbt2018-ms-ssim-8-0cb49d43.pth.tar",
        },
    },
    "cheng2020-anchor": {
        "mse": {
            1: f"{root_url}/cheng2020-anchor-1-dad2ebff.pth.tar",
            2: f"{root_url}/cheng2020-anchor-2-a29008eb.pth.tar",
            3: f"{root_url}/cheng2020-anchor-3-e49be189.pth.tar",
            4: f"{root_url}/cheng2020-anchor-4-98b0b468.pth.tar",
            5: f"{root_url}/cheng2020-anchor-5-23852949.pth.tar",
            6: f"{root_url}/cheng2020-anchor-6-4c052b1a.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/cheng2020_anchor-ms-ssim-1-20f521db.pth.tar",
            2: f"{root_url}/cheng2020_anchor-ms-ssim-2-c7ff5812.pth.tar",
            3: f"{root_url}/cheng2020_anchor-ms-ssim-3-c23e22d5.pth.tar",
            4: f"{root_url}/cheng2020_anchor-ms-ssim-4-0e658304.pth.tar",
            5: f"{root_url}/cheng2020_anchor-ms-ssim-5-c0a95e77.pth.tar",
            6: f"{root_url}/cheng2020_anchor-ms-ssim-6-f2dc1913.pth.tar",
        },
    },
    "cheng2020-attn": {
        "mse": {
            1: f"{root_url}/cheng2020_attn-mse-1-465f2b64.pth.tar",
            2: f"{root_url}/cheng2020_attn-mse-2-e0805385.pth.tar",
            3: f"{root_url}/cheng2020_attn-mse-3-2d07bbdf.pth.tar",
            4: f"{root_url}/cheng2020_attn-mse-4-f7b0ccf2.pth.tar",
            5: f"{root_url}/cheng2020_attn-mse-5-26c8920e.pth.tar",
            6: f"{root_url}/cheng2020_attn-mse-6-730501f2.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/cheng2020_attn-ms-ssim-1-c5381d91.pth.tar",
            2: f"{root_url}/cheng2020_attn-ms-ssim-2-5dad201d.pth.tar",
            3: f"{root_url}/cheng2020_attn-ms-ssim-3-5c9be841.pth.tar",
            4: f"{root_url}/cheng2020_attn-ms-ssim-4-8b2f647e.pth.tar",
            5: f"{root_url}/cheng2020_attn-ms-ssim-5-5ca1f34c.pth.tar",
            6: f"{root_url}/cheng2020_attn-ms-ssim-6-216423ec.pth.tar",
        },
    },
}

candidate_model_urls = {name: {} for name in candidate_model_architectures}

cfgs = {
    "bmshj2018-factorized": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "bmshj2018-factorized-relu": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "bmshj2018-hyperprior": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "mbt2018-mean": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (192, 320),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "mbt2018": {
        1: (192, 192),
        2: (192, 192),
        3: (192, 192),
        4: (192, 192),
        5: (192, 320),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
    "cheng2020-anchor": {
        1: (128,),
        2: (128,),
        3: (128,),
        4: (192,),
        5: (192,),
        6: (192,),
    },
    "cheng2020-attn": {
        1: (128,),
        2: (128,),
        3: (128,),
        4: (192,),
        5: (192,),
        6: (192,),
    },
}


def _load_model(
    architecture, metric, quality, pretrained=False, progress=True, **kwargs
):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if (
            architecture not in model_urls
            or metric not in model_urls[architecture]
            or quality not in model_urls[architecture][metric]
        ):
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric][quality]
        state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = load_pretrained(state_dict)
        model = model_architectures[architecture].from_state_dict(state_dict)
        return model

    model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)
    return model


def glic(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return GLIC(**kwargs)


def _hpcm_default_kwargs(variant: str) -> dict:
    if variant == "base":
        return dict(g_a_depth=4, g_s_depth=4, y_prior_depth=2, use_attention=True)
    if variant == "large":
        return dict(g_a_depth=8, g_s_depth=8, y_prior_depth=3, use_attention=True)
    if variant == "phi":
        return dict(g_a_depth=4, g_s_depth=4, y_prior_depth=2, use_attention=False)
    raise ValueError(f"Unknown HPCM variant: {variant}")


def hpcm(pretrained: bool = False, progress: bool = True, **kwargs):
    """HPCM (Hierarchical Progressive Context Modeling), default base variant."""
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    cfg = _hpcm_default_kwargs("base")
    cfg.update(kwargs)
    return HPCM(**cfg)


def hpcm_base(pretrained: bool = False, progress: bool = True, **kwargs):
    """HPCM_Base — base depth, with cross-attention progressive context fusion."""
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    cfg = _hpcm_default_kwargs("base")
    cfg.update(kwargs)
    return HPCM(**cfg)


def hpcm_large(pretrained: bool = False, progress: bool = True, **kwargs):
    """HPCM_Large — deeper analysis/synthesis and y prior, with cross-attention."""
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    cfg = _hpcm_default_kwargs("large")
    cfg.update(kwargs)
    return HPCM(**cfg)


def hpcm_phi(pretrained: bool = False, progress: bool = True, **kwargs):
    """HPCM_Phi — base depth without cross-attention (uses ψᵢ as progressive context)."""
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    cfg = _hpcm_default_kwargs("phi")
    cfg.update(kwargs)
    return HPCM(**cfg)


def cmic(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    if not is_pytorch_wavelets_available():
        raise ModuleNotFoundError(
            "CMIC requires the optional dependency `pytorch_wavelets`. "
            "Install `compressai[lic]` to enable this model."
        )
    return CMIC(**kwargs)


def cca(pretrained: bool = False, progress: bool = True, **kwargs):
    """CCA — standalone CCA-loss autoencoder (Han et al., NeurIPS 2024)."""
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return CCAModel(**kwargs)


def dcae(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return DCAE(**kwargs)


def ftic(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return FrequencyAwareTransFormer(**kwargs)


def invcompress(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    if not is_freia_available():
        raise ModuleNotFoundError(
            "InvCompress requires the optional dependency `FrEIA`. "
            "Install `compressai[invcompress]` to enable this model."
        )
    return InvCompress(**kwargs)


def informer(pretrained: bool = False, progress: bool = True, **kwargs):
    """Informer — Joint Global and Local Hierarchical Priors (Kim et al., CVPR 2022).

    Combines a vanilla 4-stage Conv+GDN encoder/decoder with a local 1×1-conv
    hyperprior and a global cross-attention hyperprior over ``num_global``
    learnable tokens. Pre-trained weights are not mirrored.
    """
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return Informer(**kwargs)


def entroformer(pretrained: bool = False, progress: bool = True, **kwargs):
    """Entroformer — Transformer-based entropy model (Qian et al., ICLR 2022).

    Ballé-style 4-stage Conv+GDN auto-encoder with a Transformer hyperprior
    (3-stage ``cit_he`` / ``cit_hd``) and a causal Transformer auto-regressive
    entropy model (``cit_ar``). Latent likelihood uses a single Gaussian
    (``num_parameter=2``, ``K=1``); the hyperprior uses
    :class:`compressai.entropy_models.LearnedGaussianBottleneck`. Pre-trained
    weights are not mirrored — use ``examples/convert_entroformer_checkpoint.py``
    to convert the official ``entroformer_lambda*.pth`` checkpoints.
    """
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return Entroformer(**kwargs)


def qian2021_ref(pretrained: bool = False, progress: bool = True, **kwargs):
    """Reference-Based AR — Global reference + local context entropy model
    (Qian et al., ICLR 2021).

    Ballé-style 4-stage Conv+GSDN auto-encoder with a 3-conv hyperprior and
    a 3-cascade auto-regressive entropy model that combines a 5×5 masked-conv
    local context, a global reference search and a hyperprior-derived feature
    into a discretised K=3 Gaussian mixture. The hyperprior uses
    :class:`compressai.entropy_models.LearnedGaussianBottleneck`. Pre-trained
    weights are not mirrored — use ``examples/convert_qian2021ref_checkpoint.py``
    to convert the official ``model_mse*.pth`` checkpoints.
    """
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return RefBasedAR(**kwargs)


def mambaic(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return MambaIC(**kwargs)


def mambavc(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return MambaVC(**kwargs)


def tinylic(pretrained: bool = False, progress: bool = True, **kwargs):
    """TinyLIC — NAT-based learned image compression (Lu & Ma, 2022).

    Pre-trained checkpoints are hosted on the upstream NJU Box (Q1–Q8); not
    mirrored here. Download them from the upstream repo and load via::

        state_dict = torch.load(path, map_location='cpu')['state_dict']
        # If the checkpoint was saved with DataParallel, strip `module.`:
        # state_dict = {k.partition('module.')[2] or k: v for k, v in
        #               state_dict.items()}
        model = TinyLIC.from_state_dict(state_dict)
    """
    del progress
    if pretrained:
        raise RuntimeError(
            "Pre-trained TinyLIC weights are not mirrored. See the upstream "
            "repository for NJU Box download links."
        )
    return TinyLIC(**kwargs)


def tic(pretrained: bool = False, progress: bool = True, **kwargs):
    """TIC — Transformer-based Image Compression (Lu et al., DCC 2022).

    `arXiv:2111.06707 <https://arxiv.org/abs/2111.06707>`_. Pre-trained
    weights are not mirrored.
    """
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return TIC(**kwargs)


def _shiftlic_factory(variant: str):
    def _factory(pretrained: bool = False, progress: bool = True, **kwargs):
        del progress
        if pretrained:
            raise RuntimeError(
                f"Pre-trained ShiftLIC ({variant}) weights are not yet "
                "available upstream."
            )
        kwargs.pop("variant", None)
        return ShiftLIC(variant=variant, **kwargs)

    _factory.__name__ = f"shiftlic_{variant}"
    _factory.__doc__ = (
        f"ShiftLIC ({variant}) — Bao et al., TCSVT 2025 "
        "(`arXiv:2503.23052 <https://arxiv.org/abs/2503.23052>`_)."
    )
    return _factory


shiftlic_small = _shiftlic_factory("small")
shiftlic_middle = _shiftlic_factory("middle")
shiftlic_large = _shiftlic_factory("large")


def mlicpp(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return MLICPlusPlus(**kwargs)


def saaf(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return SAAF(**kwargs)


def stf_wacnn(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return WACNN(**kwargs)


def stf(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return SymmetricalTransFormer(**kwargs)


def tcm(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return TCM(**kwargs)


def weconvene(pretrained: bool = False, progress: bool = True, **kwargs):
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    if not is_pytorch_wavelets_available():
        raise ModuleNotFoundError(
            "WeConvene requires the optional dependency `pytorch_wavelets`. "
            "Install `compressai[lic]` to enable this model."
        )
    return WeConvene(**kwargs)


def bmshj2018_hyperprior_gained(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Gained variable-rate ScaleHyperprior (Cui et al., CVPR 2021).

    A single model trained over a discrete lambda grid; per-channel learned
    ``Gain``/``InverseGain`` vectors plus power-mean interpolation between
    adjacent levels enable continuous rate adaptation. Pre-trained weights
    are not mirrored (the upstream community ckpt only covers the MS variant).
    """
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return GainedScaleHyperprior(**kwargs)


def mbt2018_mean_gained(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Gained variable-rate MeanScaleHyperprior (Cui et al., CVPR 2021).

    The community-trained checkpoint at
    ``candidate/GainedVAE/checkpoint_gainmshp_epoch90.pth`` (N=128, M=192,
    8 levels) loads via :meth:`GainedMSHyperprior.from_state_dict`.
    """
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return GainedMSHyperprior(**kwargs)


def mbt2018_mean_gained_sc(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Spatial-channel gained MS hyperprior with SFT modulation.

    Adds a quality-map (qmap) input fused via SPADE-style SFT blocks at three
    encoder stages and re-injected on the decoder side. No upstream pretrained
    checkpoint exists for this variant.
    """
    del progress
    if pretrained:
        raise RuntimeError("Pre-trained model not yet available")
    return SCGainedMSHyperprior(**kwargs)


def bmshj2018_factorized(
    quality, metric="mse", pretrained=False, progress=True, **kwargs
):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model(
        "bmshj2018-factorized", metric, quality, pretrained, progress, **kwargs
    )


def bmshj2018_factorized_relu(
    quality, metric="mse", pretrained=False, progress=True, **kwargs
):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.
    GDN activations are replaced by ReLU
    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model(
        "bmshj2018-factorized", metric, quality, pretrained, progress, **kwargs
    )


def bmshj2018_hyperprior(
    quality, metric="mse", pretrained=False, progress=True, **kwargs
):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model(
        "bmshj2018-hyperprior", metric, quality, pretrained, progress, **kwargs
    )


def mbt2018_mean(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("mbt2018-mean", metric, quality, pretrained, progress, **kwargs)


def mbt2018(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("mbt2018", metric, quality, pretrained, progress, **kwargs)


def cheng2020_anchor(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model(
        "cheng2020-anchor", metric, quality, pretrained, progress, **kwargs
    )


def cheng2020_attn(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model(
        "cheng2020-attn", metric, quality, pretrained, progress, **kwargs
    )
