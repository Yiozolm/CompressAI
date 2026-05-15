# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

import torch

from compressai.models.gllmm import (
    GLLMM,
    GLLMMAnalysisTransform,
    GLLMMHyperAnalysisTransform,
    GLLMMHyperSynthesisTransform,
    GLLMMSynthesisTransform,
)


def test_gllmm_transform_shapes():
    channels = 8
    x = torch.randn(1, 3, 64, 64)
    g_a = GLLMMAnalysisTransform(channels)
    h_a = GLLMMHyperAnalysisTransform(channels)
    h_s = GLLMMHyperSynthesisTransform(channels)
    g_s = GLLMMSynthesisTransform(channels)

    y = g_a(x)
    z = h_a(y)
    phi = h_s(z)
    x_hat = g_s(y)

    assert y.shape == (1, channels, 4, 4)
    assert z.shape == (1, channels, 1, 1)
    assert phi.shape == (1, 2 * channels, 4, 4)
    assert x_hat.shape == x.shape


def test_gllmm_transform_gradients():
    channels = 8
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    y = GLLMMAnalysisTransform(channels)(x)
    x_hat = GLLMMSynthesisTransform(channels)(y)

    loss = x_hat.square().mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_gllmm_forward_smoke():
    channels = 8
    model = GLLMM(N=channels).eval()
    x = torch.rand(1, 3, 64, 64)

    out = model(x)

    assert out["x_hat"].shape == x.shape
    assert out["likelihoods"]["y"].shape == (1, channels, 4, 4)
    assert out["likelihoods"]["z"].shape == (1, channels, 1, 1)
    assert (out["likelihoods"]["y"] > 0).all()
    assert (out["likelihoods"]["z"] > 0).all()


def test_gllmm_compression_smoke():
    channels = 4
    model = GLLMM(N=channels).eval()
    model.update(force=True)
    x = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        compressed = model.compress(x)
        decoded = model.decompress(compressed["strings"], compressed["shape"])

    assert len(compressed["strings"]) == 2
    assert len(compressed["strings"][0]) == x.size(0)
    assert len(compressed["strings"][1]) == x.size(0)
    assert decoded["x_hat"].shape == x.shape
    assert torch.isfinite(decoded["x_hat"]).all()
