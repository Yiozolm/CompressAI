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
    MaskedConv2d,
    MultistageMaskedConv2d,
    QReLU,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
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


def test_AttentionBlock():
    layer = AttentionBlock(8)
    layer(torch.rand(1, 8, 4, 4))


class TestMutiScaleDictionaryCrossAttentionGLU:
    @staticmethod
    def test_forward_shape():
        from compressai.layers.attn.dictionary import (
            MutiScaleDictionaryCrossAttentionGLU,
        )

        mod = MutiScaleDictionaryCrossAttentionGLU(
            input_dim=192,
            output_dim=320,
            head_num=4,
            dictionary_dim=128,
        )
        x = torch.randn(2, 192, 4, 4)
        dictionary = torch.randn(2, 16, 128)
        out = mod(x, dictionary)
        assert out.shape == (2, 320, 4, 4)

    @staticmethod
    def test_state_dict_round_trip():
        from compressai.layers.attn.dictionary import (
            MutiScaleDictionaryCrossAttentionGLU,
        )

        mod = MutiScaleDictionaryCrossAttentionGLU(
            input_dim=192,
            output_dim=320,
            head_num=4,
            dictionary_dim=128,
        )
        mod2 = MutiScaleDictionaryCrossAttentionGLU(
            input_dim=192,
            output_dim=320,
            head_num=4,
            dictionary_dim=128,
        )
        mod2.load_state_dict(mod.state_dict(), strict=True)
        x = torch.randn(2, 192, 4, 4)
        dictionary = torch.randn(2, 16, 128)
        assert torch.allclose(mod(x, dictionary), mod2(x, dictionary))

    @staticmethod
    def test_dictionary_dim_default_matches_head_num():
        from compressai.layers.attn.dictionary import (
            MutiScaleDictionaryCrossAttentionGLU,
        )

        # Default dictionary_dim = 32 * head_num
        mod = MutiScaleDictionaryCrossAttentionGLU(
            input_dim=64, output_dim=128, head_num=4
        )
        x = torch.randn(1, 64, 2, 2)
        dictionary = torch.randn(1, 8, 128)
        out = mod(x, dictionary)
        assert out.shape == (1, 128, 2, 2)

    @staticmethod
    def test_dictionary_dim_must_divide_head_num():
        from compressai.layers.attn.dictionary import (
            MutiScaleDictionaryCrossAttentionGLU,
        )

        with pytest.raises(ValueError, match="divisible"):
            MutiScaleDictionaryCrossAttentionGLU(
                input_dim=32, output_dim=64, head_num=3, dictionary_dim=128
            )


class TestWavelet:
    @staticmethod
    def test_dwt_idwt_round_trip():
        pytest.importorskip("pytorch_wavelets")
        from compressai.layers.wave import DWT2D, IDWT2D

        dwt = DWT2D(wave="haar")
        idwt = IDWT2D(wave="haar")
        x = torch.randn(2, 3, 16, 16)
        sub = dwt(x)
        # 4 subbands -> output channels = 4 * input
        assert sub.shape == (2, 12, 8, 8)
        rec = idwt(sub)
        assert rec.shape == x.shape
        assert (rec - x).abs().max().item() < 1e-5

    @staticmethod
    def test_is_pytorch_wavelets_available_returns_bool():
        from compressai.layers.wave import is_pytorch_wavelets_available

        assert isinstance(is_pytorch_wavelets_available(), bool)


class TestGraph:
    @staticmethod
    def _gfa(**overrides):
        from compressai.layers.graph import GFA

        kwargs = dict(
            dim=64,
            depth=2,
            num_heads=8,
            window_size=8,
            sample_size=16,
            graph_flags=True,
            top_k=16,
            diff_scales=1.5,
            stages=["GN", "GS"],
        )
        kwargs.update(overrides)
        return GFA(**kwargs)

    def test_gfa_forward_shape(self):
        pytest.importorskip("timm")
        # GFA consumes a 4-D BCHW feature map plus its (H, W) size; smallest
        # spatial size it tolerates is 16x16 (global sampling uses sample_size=16).
        gfa = self._gfa().eval()
        x = torch.rand(1, 64, 16, 16)
        with torch.no_grad():
            out = gfa(x, (16, 16))
        assert out.shape == x.shape

    def test_gfa_state_dict_round_trip(self):
        pytest.importorskip("timm")
        gfa = self._gfa().eval()
        gfa2 = self._gfa().eval()
        gfa2.load_state_dict(gfa.state_dict(), strict=True)
        x = torch.rand(1, 64, 16, 16)
        with torch.no_grad():
            assert torch.allclose(gfa(x, (16, 16)), gfa2(x, (16, 16)))

    def test_feature_reshape_restore_round_trip(self):
        pytest.importorskip("timm")
        from compressai.layers.graph import FeatureReshape, FeatureRestore

        reshape = FeatureReshape(embed_dim=16)
        restore = FeatureRestore(embed_dim=16)
        x = torch.randn(2, 16, 8, 8)
        embedded = reshape(x)  # BCHW -> B(HW)C
        assert embedded.shape == (2, 64, 16)
        restored = restore(embedded, (8, 8))  # B(HW)C -> BCHW
        assert torch.allclose(x, restored)

    def test_graph_ops_numerics(self):
        pytest.importorskip("timm")
        from compressai.layers.graph import (
            compute_sobel_gradients,
            cosine_similarity,
            gaussian_blur,
        )

        orthogonal = cosine_similarity(
            torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]]),
        )
        assert orthogonal.abs().max().item() < 1e-6
        identical = cosine_similarity(
            torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]),
        )
        assert abs(identical.max().item() - 1.0) < 1e-6

        # gaussian_blur preserves shape; sobel returns a per-pixel edge map.
        assert gaussian_blur(torch.randn(2, 3, 16, 16)).shape == (2, 3, 16, 16)
        assert compute_sobel_gradients(
            torch.randn(2, 8 * 8, 16), shape=(8, 8)
        ).shape == (2, 8, 8)


class TestSSM:
    def test_backend_selection_pure_pytorch(self):
        pytest.importorskip("timm")
        from compressai.layers.ssm import (
            get_selective_scan_backend,
            is_mamba_ssm_available,
            is_selective_scan_cuda_available,
        )

        # Without an accelerated backend installed the resolver must fall back
        # to the pure-PyTorch reference implementation.
        if not (is_mamba_ssm_available() or is_selective_scan_cuda_available()):
            assert get_selective_scan_backend() == "torch"

    def test_selective_scan_torch_matches_reference(self):
        pytest.importorskip("timm")
        from compressai.layers.ssm import selective_scan, selective_scan_ref

        torch.manual_seed(0)
        batch, channels, length, states = 1, 4, 8, 4
        u = torch.randn(batch, channels, length)
        delta = torch.rand(batch, channels, length)
        A = -torch.rand(channels, states)
        B = torch.randn(batch, 1, states, length)
        C = torch.randn(batch, 1, states, length)
        D = torch.randn(channels)
        delta_bias = torch.randn(channels)

        out = selective_scan(
            u, delta, A, B, C, D, delta_bias=delta_bias, backend="torch"
        )
        ref = selective_scan_ref(u, delta, A, B, C, D, delta_bias=delta_bias)
        assert out.shape == (batch, channels, length)
        assert torch.allclose(out, ref)

    def test_cross_scan_merge_shapes(self):
        pytest.importorskip("timm")
        from compressai.layers.ssm import cross_merge, cross_scan

        x = torch.randn(2, 4, 6, 6)
        scanned = cross_scan(x)  # (B, 4 directions, C, H*W)
        assert scanned.shape == (2, 4, 4, 36)
        merged = cross_merge(scanned.view(2, 4, 4, 6, 6))  # (B, C, H*W)
        assert merged.shape == (2, 4, 36)

    def test_ss2d_forward_shape(self):
        pytest.importorskip("timm")
        from compressai.layers.ssm import SS2D

        ss2d = SS2D(d_model=16, d_state=4, ssm_ratio=2.0, d_conv=3).eval()
        x = torch.rand(1, 8, 8, 16)  # SS2D operates on (B, H, W, C)
        with torch.no_grad():
            out = ss2d(x)
        assert out.shape == x.shape

    def test_vssblock_forward_shape_and_round_trip(self):
        pytest.importorskip("timm")
        from compressai.layers.ssm import VSSBlock

        block = VSSBlock(hidden_dim=16, d_state=4, ssm_ratio=2.0, d_conv=3).eval()
        block2 = VSSBlock(hidden_dim=16, d_state=4, ssm_ratio=2.0, d_conv=3).eval()
        block2.load_state_dict(block.state_dict(), strict=True)
        x = torch.rand(1, 16, 8, 8)  # VSSBlock operates on (B, C, H, W)
        with torch.no_grad():
            out = block(x)
            out2 = block2(x)
        assert out.shape == x.shape
        assert torch.allclose(out, out2)

    def test_build_vss_backbone_shapes(self):
        pytest.importorskip("timm")
        from compressai.layers.ssm import build_vss_backbone

        g_a, g_s, h_a, h_mean_s, h_scale_s = build_vss_backbone(
            depths=(1, 1, 1, 1),
            drop_path_rate=0.0,
            N=8,
            M=16,
            hyper_channels=12,
            ssm_d_state=4,
        )
        x = torch.rand(1, 3, 64, 64)
        y = g_a(x)
        assert y.shape == (1, 16, 4, 4)  # four stride-2 stages
        z = h_a(y)
        assert z.shape == (1, 12, 1, 1)
        assert h_mean_s(z).shape == (1, 16, 4, 4)
        assert h_scale_s(z).shape == (1, 16, 4, 4)
        assert g_s(y).shape == x.shape

    def test_infer_vss_depths_and_block_kwargs(self):
        pytest.importorskip("timm")
        from compressai.layers.ssm import (
            infer_vss_block_kwargs,
            infer_vss_depths,
        )
        from compressai.models.mambaic import MambaIC

        model = MambaIC(
            depths=(2, 1, 3, 1),
            N=16,
            M=32,
            hyper_channels=24,
            num_slices=2,
            max_support_slices=2,
            context_depths=(1, 1),
            window_size=4,
            support_head_dim=4,
            context_head_dim=4,
            support_attention_dim=8,
            context_attention_dim=8,
            ssm_d_state=4,
            ssm_ratio=2.0,
            ssm_conv=3,
        )
        sd = model.state_dict()
        assert infer_vss_depths(sd) == (2, 1, 3, 1)
        kwargs = infer_vss_block_kwargs(sd)
        assert kwargs["ssm_d_state"] == 4
        assert kwargs["ssm_ratio"] == 2.0
        assert kwargs["ssm_conv"] == 3


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


class TestMultistageMaskedConv2d:
    def test_mask_a_keeps_even_even(self):
        conv = MultistageMaskedConv2d(2, 2, kernel_size=3, padding=1, mask_type="A")
        mask = conv.mask[0, 0]
        assert mask[0, 0] == 1 and mask[0, 2] == 1
        assert mask[2, 0] == 1 and mask[2, 2] == 1
        assert mask[0, 1] == 0 and mask[1, 0] == 0 and mask[1, 1] == 0

    def test_mask_b_keeps_anti_diagonal(self):
        conv = MultistageMaskedConv2d(2, 2, kernel_size=5, padding=2, mask_type="B")
        mask = conv.mask[0, 0]
        assert mask[0, 1] == 1 and mask[1, 0] == 1
        assert mask[0, 0] == 0 and mask[1, 1] == 0

    def test_mask_c_keeps_upper_half_plus_odd_rows(self):
        conv = MultistageMaskedConv2d(2, 2, kernel_size=5, padding=2, mask_type="C")
        mask = conv.mask[0, 0]
        assert mask[0, 1] == 1
        assert mask[1, 0] == 1 and mask[1, 1] == 1 and mask[1, 2] == 1
        assert mask[0, 0] == 0 and mask[0, 2] == 0

    def test_invalid_mask_type(self):
        with pytest.raises(ValueError):
            MultistageMaskedConv2d(2, 2, kernel_size=3, mask_type="Z")

    def test_forward_shape(self):
        conv = MultistageMaskedConv2d(2, 4, kernel_size=3, padding=1, mask_type="A")
        x = torch.rand(1, 2, 8, 8)
        assert conv(x).shape == (1, 4, 8, 8)


class TestMultiplex:
    def test_space2depth_depth2space_round_trip(self):
        from compressai.ops import depth2space, space2depth

        x = torch.randn(2, 8, 16, 16)
        assert torch.allclose(depth2space(space2depth(x)), x)

    def test_demultiplex_round_trip(self):
        from compressai.ops import demultiplex, multiplex

        x = torch.randn(2, 8, 16, 16)
        anchor, non_anchor = demultiplex(x)
        assert torch.allclose(multiplex(anchor, non_anchor), x)

    def test_demultiplex_v2_round_trip(self):
        from compressai.ops import demultiplex_v2, multiplex_v2

        x = torch.randn(2, 8, 16, 16)
        y1, y2, y3, y4 = demultiplex_v2(x)
        assert torch.allclose(multiplex_v2(y1, y2, y3, y4), x)


class TestNSA:
    def test_resvitblock_forward_shape(self):
        pytest.importorskip("timm")
        from compressai.layers.attn import ResViTBlock

        model = ResViTBlock(dim=16, depth=2, num_heads=2, kernel_size=3).eval()
        x = torch.rand(1, 16, 8, 8)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_resvitblock_state_dict_round_trip(self):
        pytest.importorskip("timm")
        from compressai.layers.attn import ResViTBlock

        model = ResViTBlock(dim=16, depth=2, num_heads=2, kernel_size=3).eval()
        x = torch.rand(1, 16, 8, 8)
        with torch.no_grad():
            y = model(x)
        clone = ResViTBlock(dim=16, depth=2, num_heads=2, kernel_size=3).eval()
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            y2 = clone(x)
        assert torch.allclose(y, y2)
        keys = set(model.state_dict().keys())
        assert "residual_group.blocks.0.attn.qkv.weight" in keys
        assert "residual_group.blocks.0.attn.rpb" in keys
