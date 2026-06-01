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

import pytest
import torch

from compressai._CXX import pmf_to_quantized_cdf
from compressai.ops import (
    DiamondLatticeQuantizer,
    LowerBound,
    NonNegativeParametrizer,
    diamond_lattice_quantize,
    quantize_ste,
)


class TestQuantizeSTE:
    def test_quantize_ste_ok(self):
        x = torch.rand(16)
        assert (quantize_ste(x) == torch.round(x)).all()

    def test_quantize_ste_grads(self):
        x = torch.rand(24, requires_grad=True)
        y = quantize_ste(x)
        y.backward(x)
        assert x.grad is not None
        assert (x.grad == x).all()


class TestDiamondLatticeQuantizer:
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
    def test_diamond_lattice_quantize_shape(self, dtype):
        x = torch.rand(2, 4, 3, 5, dtype=dtype)
        y = diamond_lattice_quantize(x, round_mode="hard")
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()

    def test_diamond_lattice_quantize_branch_index(self):
        x = torch.rand(2, 4, 3, 5)
        y, indexes = diamond_lattice_quantize(
            x,
            round_mode="hard",
            return_indexes=True,
        )
        assert y.shape == x.shape
        assert indexes.shape == (2, 3, 5)
        assert indexes.dtype == torch.int64
        assert ((indexes == 0) | (indexes == 1)).all()

    def test_diamond_lattice_quantize_hard_nearest_coset(self):
        x = torch.tensor([[[[0.49]], [[0.49]]], [[[0.10]], [[0.10]]]])
        y, indexes = diamond_lattice_quantize(
            x,
            round_mode="hard",
            return_indexes=True,
        )

        expected = torch.tensor([[[[0.50]], [[0.50]]], [[[0.00]], [[0.00]]]])
        expected_indexes = torch.tensor([[[1]], [[0]]])
        assert torch.equal(y, expected)
        assert torch.equal(indexes, expected_indexes)

    def test_diamond_lattice_quantize_ste_grad(self):
        x = torch.rand(2, 4, 3, 5, requires_grad=True)
        y = diamond_lattice_quantize(x, round_mode="ste")
        y.sum().backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_diamond_lattice_quantize_step_scaling(self):
        x = torch.full((1, 2, 1, 1), 0.26)
        y_step_1 = diamond_lattice_quantize(x, step=1.0, round_mode="hard")
        y_step_half = diamond_lattice_quantize(x, step=0.5, round_mode="hard")

        assert torch.equal(y_step_1, torch.full_like(x, 0.5))
        assert torch.equal(y_step_half, torch.full_like(x, 0.25))

    def test_diamond_lattice_quantize_channel_step(self):
        x = torch.tensor([[[[0.26]], [[0.76]]]])
        step = torch.tensor([0.5, 1.0])
        y = diamond_lattice_quantize(x, step=step, round_mode="hard")

        expected = torch.tensor([[[[0.25]], [[0.50]]]])
        assert torch.equal(y, expected)

    def test_diamond_lattice_quantizer_module(self):
        x = torch.rand(2, 4, 3, 5)
        quantizer = DiamondLatticeQuantizer(step=0.5, round_mode="hard")
        expected = diamond_lattice_quantize(x, step=0.5, round_mode="hard")

        assert torch.equal(quantizer(x), expected)

    def test_diamond_lattice_quantize_invalid_config(self):
        x = torch.rand(2, 4, 3, 5)

        with pytest.raises(ValueError):
            diamond_lattice_quantize(x, step=0.0)
        with pytest.raises(ValueError):
            diamond_lattice_quantize(x, round_mode="invalid")
        with pytest.raises(ValueError):
            diamond_lattice_quantize(x, block_axis="invalid")
        with pytest.raises(ValueError):
            diamond_lattice_quantize(torch.rand(4), block_axis="channel")


class TestLowerBound:
    def test_lower_bound_ok(self):
        x = torch.rand(16)
        bound = torch.rand(1)
        lower_bound = LowerBound(bound)
        assert (lower_bound(x) == torch.max(x, bound)).all()

    def test_lower_bound_script(self):
        x = torch.rand(16)
        bound = torch.rand(1)
        lower_bound = LowerBound(bound)
        scripted = torch.jit.script(lower_bound)
        assert (scripted(x) == torch.max(x, bound)).all()

    def test_lower_bound_grads(self):
        x = torch.rand(16, requires_grad=True)
        bound = torch.rand(1)
        lower_bound = LowerBound(bound)
        y = lower_bound(x)
        y.backward(x)

        assert x.grad is not None
        assert (x.grad == ((x >= bound) * x)).all()


class TestNonNegativeParametrizer:
    def test_non_negative(self):
        parametrizer = NonNegativeParametrizer()
        x = torch.rand(1, 8, 8, 8) * 2 - 1  # [0, 1] -> [-1, 1]
        x_reparam = parametrizer(x)

        assert x_reparam.shape == x.shape
        assert x_reparam.min() >= 0

    def test_non_negative_init(self):
        parametrizer = NonNegativeParametrizer()
        x = torch.rand(1, 8, 8, 8) * 2 - 1
        x_init = parametrizer.init(x)

        assert x_init.shape == x.shape
        assert torch.allclose(x_init, torch.sqrt(torch.max(x, x - x)), atol=2**-18)

    def test_non_negative_min(self):
        for _ in range(10):
            minimum = torch.rand(1)
            parametrizer = NonNegativeParametrizer(minimum.item())
            x = torch.rand(1, 8, 8, 8) * 2 - 1
            x_reparam = parametrizer(x)

            assert x_reparam.shape == x.shape
            assert torch.allclose(x_reparam.min(), minimum)


class TestPmfToQuantizedCDF:
    def test_ok(self):
        out = pmf_to_quantized_cdf([0.1, 0.2, 0, 0], 16)
        assert out == [0, 21845, 65534, 65535, 65536]

    def test_negative_prob(self):
        with pytest.raises(ValueError):
            pmf_to_quantized_cdf([1, 0, -1], 16)

    @pytest.mark.parametrize("v", ("inf", "-inf", "nan"))
    def test_non_finite_prob(self, v):
        with pytest.raises(ValueError):
            pmf_to_quantized_cdf([1, 0, float(v)], 16)

        with pytest.raises(ValueError):
            pmf_to_quantized_cdf([1, 0, float(v), 2, 3, 4], 16)
