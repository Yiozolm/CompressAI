# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0.

"""SPADE-style Spatial Feature Transform (SFT).

Used by GainedVAE's ``SCGainedMSHyperprior`` for spatial-channel feature
modulation, and reusable for other quality-map / structure-conditioned
encoders (e.g. QmapCompression, ICCV 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["SFT"]


class SFT(nn.Module):
    """Spatially-adaptive feature transform.

    Predicts per-pixel ``(gamma, beta)`` from a conditioning map and applies
    ``out = x * (1 + gamma) + beta``. The conditioning map is adaptive-avg-pooled
    to ``x``'s spatial size, so resolution mismatches are tolerated.

    Args:
        x_nc: number of channels of the modulated feature ``x``.
        prior_nc: number of channels of the conditioning map.
        ks: kernel size for the gamma / beta predictor convs.
        nhidden: hidden width of the shared MLP.
    """

    def __init__(self, x_nc: int, prior_nc: int = 1, ks: int = 3, nhidden: int = 128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x: torch.Tensor, qmap: torch.Tensor) -> torch.Tensor:
        qmap = F.adaptive_avg_pool2d(qmap, x.size()[2:])
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return x * (1 + gamma) + beta
