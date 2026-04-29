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

from compressai.layers import (
    GDN,
    GDN1,
    AttentionBlock,
    CheapCS1,
    MaskedConv2d,
    MultistageMaskedConv2d,
    QReLU,
    GatedTransformCNN,
    LayerNorm2d,
    ResidualBlock,
    ResidualBlockShift,
    ResidualBottleneckBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    ResidualShiftStack,
    Shift4,
)
from compressai.ops import (
    demultiplex,
    demultiplex_v2,
    multiplex,
    multiplex_v2,
)


class TestMaskedConv2d:
    @staticmethod
    def test_mask_type():
        MaskedConv2d(1, 3, 3, mask_type="A")
        MaskedConv2d(1, 3, 3, mask_type="B")

        with pytest.raises(ValueError):
            MaskedConv2d(1, 3, 3, mask_type="C")

    @staticmethod
    def test_mask_A():
        conv = MaskedConv2d(1, 3, 5, mask_type="A")

        assert (conv.mask[0] == conv.mask[1]).all()
        assert (conv.mask[0] == conv.mask[2]).all()

        _, _, h, w = conv.mask.size()
        a = torch.ones_like(conv.mask)
        a[:, :, h // 2, w // 2 :] = 0
        a[:, :, h // 2 + 1 :] = 0

        assert (conv.mask == a).all()

    @staticmethod
    def test_mask_B():
        conv = MaskedConv2d(1, 3, 5, mask_type="B")

        assert (conv.mask[0] == conv.mask[1]).all()
        assert (conv.mask[0] == conv.mask[2]).all()

        _, _, h, w = conv.mask.size()
        b = torch.ones_like(conv.mask)
        b[:, :, h // 2, w // 2 + 1 :] = 0
        b[:, :, h // 2 + 1 :] = 0

        assert (conv.mask == b).all()

    @staticmethod
    def test_mask_A_1d():
        conv = MaskedConv2d(1, 3, (1, 5), mask_type="A")

        assert (conv.mask[0] == conv.mask[1]).all()
        assert (conv.mask[0] == conv.mask[2]).all()

        _, _, h, w = conv.mask.size()
        a = torch.ones_like(conv.mask)
        a[:, :, h // 2, w // 2 :] = 0
        a[:, :, h // 2 + 1 :] = 0

        assert (conv.mask == a).all()

    @staticmethod
    def test_mask_B_1d():
        conv = MaskedConv2d(3, 1, (5, 1), mask_type="B")

        assert (conv.mask[:, 0] == conv.mask[:, 1]).all()
        assert (conv.mask[:, 0] == conv.mask[:, 2]).all()

        _, _, h, w = conv.mask.size()
        b = torch.ones_like(conv.mask)
        b[:, :, h // 2, w // 2 + 1 :] = 0
        b[:, :, h // 2 + 1 :] = 0

        assert (conv.mask == b).all()

    @staticmethod
    def test_mask_multiple():
        cfgs = [
            # (in, out, kernel_size)
            (1, 3, 5),
            (3, 1, 3),
            (3, 3, 7),
        ]

        for cfg in cfgs:
            in_ch, out_ch, k = cfg
            conv = MaskedConv2d(in_ch, out_ch, k, mask_type="A")

            assert conv.mask[0].sum() != 0
            assert (conv.mask - conv.mask[0]).sum() == 0

            _, _, h, w = conv.mask.size()
            a = torch.ones_like(conv.mask)
            a[:, :, h // 2, w // 2 :] = 0
            a[:, :, h // 2 + 1 :] = 0

            assert (conv.mask == a).all()


class TestGDN:
    def test_gdn(self):
        g = GDN(32)
        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = g(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x / torch.sqrt(1 + 0.1 * (x**2))
        assert torch.allclose(y_ref, y)

    def test_igdn(self):
        g = GDN(32, inverse=True)
        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = g(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x * torch.sqrt(1 + 0.1 * (x**2))
        assert torch.allclose(y_ref, y)

    def test_gdn1(self):
        g = GDN1(32)
        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = g(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x / (1 + 0.1 * torch.abs(x))
        assert torch.allclose(y_ref, y)


def test_ResidualBlockWithStride():
    layer = ResidualBlockWithStride(32, 64, stride=1)
    layer(torch.rand(1, 32, 4, 4))

    layer = ResidualBlockWithStride(32, 32, stride=1)
    layer(torch.rand(1, 32, 4, 4))

    layer = ResidualBlockWithStride(32, 32, stride=2)
    layer(torch.rand(1, 32, 4, 4))

    layer = ResidualBlockWithStride(32, 64, stride=2)
    layer(torch.rand(1, 32, 4, 4))


def test_ResidualBlockUpsample():
    layer = ResidualBlockUpsample(8, 16)
    layer(torch.rand(1, 8, 4, 4))


def test_ResidualBlock():
    layer = ResidualBlock(8, 8)
    layer(torch.rand(1, 8, 4, 4))

    layer = ResidualBlock(8, 16)
    layer(torch.rand(1, 8, 4, 4))


def test_ResidualBottleneckBlock():
    layer = ResidualBottleneckBlock(8, 16)
    output = layer(torch.rand(1, 8, 4, 4))

    assert output.shape == (1, 16, 4, 4)


def test_LayerNorm2d():
    layer = LayerNorm2d(8)
    input_tensor = torch.rand(1, 8, 4, 4, requires_grad=True)
    output = layer(input_tensor)
    output.backward(input_tensor)

    assert output.shape == input_tensor.shape
    assert input_tensor.grad is not None
    assert input_tensor.grad.shape == input_tensor.shape


def test_GatedTransformCNN():
    layer = GatedTransformCNN(8, 16)
    output = layer(torch.rand(1, 8, 4, 4))

    assert output.shape == (1, 16, 4, 4)


def test_AttentionBlock():
    layer = AttentionBlock(8)
    layer(torch.rand(1, 8, 4, 4))


class TestMultistageMaskedConv2d:
    @staticmethod
    def test_invalid_mask_type():
        with pytest.raises(ValueError):
            MultistageMaskedConv2d(1, 1, 3, mask_type="Z")

    @staticmethod
    def test_mask_A():
        # 3x3, keeps only (even, even).
        conv = MultistageMaskedConv2d(1, 1, 3, mask_type="A")
        expected = torch.zeros_like(conv.mask)
        expected[:, :, 0::2, 0::2] = 1
        assert (conv.mask == expected).all()

    @staticmethod
    def test_mask_B():
        # 5x5, keeps the two anti-diagonal halves.
        conv = MultistageMaskedConv2d(1, 1, 5, padding=2, mask_type="B")
        expected = torch.zeros_like(conv.mask)
        expected[:, :, 0::2, 1::2] = 1
        expected[:, :, 1::2, 0::2] = 1
        assert (conv.mask == expected).all()

    @staticmethod
    def test_mask_C():
        # 5x5, keeps (even, odd) + all odd rows.
        conv = MultistageMaskedConv2d(1, 1, 5, padding=2, mask_type="C")
        expected = torch.zeros_like(conv.mask)
        expected[:, :, 0::2, 1::2] = 1
        expected[:, :, 1::2, :] = 1
        assert (conv.mask == expected).all()

    @staticmethod
    def test_forward_zeroes_masked_positions():
        conv = MultistageMaskedConv2d(2, 2, 3, padding=1, mask_type="A")
        torch.nn.init.normal_(conv.weight)
        _ = conv(torch.randn(1, 2, 4, 4))
        # Mask-A keeps only (even, even); other positions must be zero.
        assert (conv.weight[:, :, 0, 1] == 0).all()
        assert (conv.weight[:, :, 1, :] == 0).all()


class TestMultiplex:
    @staticmethod
    def test_demultiplex_round_trip():
        x = torch.randn(2, 8, 4, 6)
        anchor, non_anchor = demultiplex(x)
        # space2depth r=2 expands C 8→32; then split into 16 + 16.
        assert anchor.shape == (2, 16, 2, 3)
        assert non_anchor.shape == (2, 16, 2, 3)
        x_round = multiplex(anchor, non_anchor)
        assert x_round.shape == x.shape
        assert torch.equal(x_round, x)

    @staticmethod
    def test_demultiplex_v2_round_trip():
        x = torch.randn(2, 8, 4, 6)
        y1, y2, y3, y4 = demultiplex_v2(x)
        # 4-way split of the 32-channel space2depth result: each is C/4 = 8.
        for y in (y1, y2, y3, y4):
            assert y.shape == (2, 8, 2, 3)
        x_round = multiplex_v2(y1, y2, y3, y4)
        assert x_round.shape == x.shape
        assert torch.equal(x_round, x)


class TestShift:
    @staticmethod
    def test_shift4_directions():
        # 4 channels, 4 groups of 1 channel each. Each group shifts in one
        # direction by stride=1; verify a single "1" at center moves into
        # the correct neighbor cell for each group.
        layer = Shift4(groups=1, stride=1, mode="constant")
        x = torch.zeros(1, 4, 5, 5)
        x[:, :, 2, 2] = 1.0
        y = layer(x)
        # Group 0: the 1 originally at (2,2) appears at (3,2) (down-shift).
        assert y[0, 0, 3, 2].item() == 1.0
        # Group 1: appears at (1,2) (up-shift).
        assert y[0, 1, 1, 2].item() == 1.0
        # Group 2: appears at (2,3) (right-shift).
        assert y[0, 2, 2, 3].item() == 1.0
        # Group 3: appears at (2,1) (left-shift).
        assert y[0, 3, 2, 1].item() == 1.0
        assert y.sum().item() == 4.0

    @staticmethod
    def test_residual_block_shift_forward():
        layer = ResidualBlockShift(8, 8)
        out = layer(torch.randn(2, 8, 6, 6))
        assert out.shape == (2, 8, 6, 6)
        layer = ResidualBlockShift(8, 16)
        out = layer(torch.randn(2, 8, 6, 6))
        assert out.shape == (2, 16, 6, 6)

    @staticmethod
    def test_residual_shift_stack_shape():
        layer = ResidualShiftStack(in_ch=128, out_ch=64)
        out = layer(torch.randn(1, 128, 8, 8))
        assert out.shape == (1, 64, 8, 8)

    @staticmethod
    def test_residual_shift_stack_odd_out_ch_rejected():
        with pytest.raises(ValueError, match="even"):
            ResidualShiftStack(in_ch=8, out_ch=7)

    @staticmethod
    def test_cheap_cs1_forward():
        layer = CheapCS1(dim=64)
        out = layer(torch.randn(1, 64, 16, 16))
        assert out.shape == (1, 64, 16, 16)


class TestQReLU:
    @staticmethod
    def test_QReLU():
        def qrelu(input, bit_depth=8, beta=100):
            return QReLU.apply(input, bit_depth, beta)

        x = torch.rand(1, 32, 16, 16, requires_grad=True)
        y = qrelu(x)
        y.backward(x)

        assert y.shape == x.shape
        assert x.grad is not None
        assert x.grad.shape == x.shape

        y_ref = x.clamp(min=0, max=2**8 - 1)
        assert torch.allclose(y_ref, y)
