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
import torch.nn as nn


class TestSharedDictionary:
    def test_dt_shape_and_state_dict_path(self):
        from compressai.models._helpers.dictionary_context import SharedDictionary

        shared = SharedDictionary(dict_num=16, dictionary_dim=64)
        assert shared.dt.shape == (16, 64)
        assert list(shared.state_dict().keys()) == ["dt"]

    def test_expand_for_broadcasts_without_copy(self):
        from compressai.models._helpers.dictionary_context import SharedDictionary

        shared = SharedDictionary(dict_num=8, dictionary_dim=32)
        out = shared.expand_for(4)
        assert out.shape == (4, 8, 32)
        # All B copies share storage with the underlying dt
        assert out.data_ptr() == shared.dt.data_ptr()


class TestBuildDictionaryMeanScaleHead:
    def _build(self, *, emit_mean_support=False):
        from compressai.models._helpers.dictionary_context import (
            SharedDictionary,
            build_dictionary_mean_scale_head,
        )

        # Tiny config: M=32, slice_ch=8, support_count=2
        m = 32
        slice_ch = 8
        support_count = 2
        support_ch = 2 * m + slice_ch * support_count
        shared = SharedDictionary(dict_num=8, dictionary_dim=64)
        head = build_dictionary_mean_scale_head(
            slice_ch=slice_ch,
            support_ch=support_ch,
            shared_dictionary=shared,
            dict_output_ch=m,
            cross_attention_kwargs={"head_num": 4, "mlp_rate": 2},
            widths=(16,),
            emit_mean_support=emit_mean_support,
        )
        return shared, head, m, slice_ch, support_ch

    def test_forward_shape_no_emit(self):
        shared, head, m, slice_ch, support_ch = self._build(emit_mean_support=False)
        x = torch.randn(2, support_ch, 4, 4)
        out = head(x)
        # Output: cat([scale, mean]) -> 2 * slice_ch
        assert out.shape == (2, 2 * slice_ch, 4, 4)

    def test_forward_shape_with_emit_mean_support(self):
        shared, head, m, slice_ch, support_ch = self._build(emit_mean_support=True)
        x = torch.randn(2, support_ch, 4, 4)
        out = head(x)
        # Output: cat([scale, mean, support]) where support = cat([x, dict_info(M)])
        expected = 2 * slice_ch + (support_ch + m)
        assert out.shape == (2, expected, 4, 4)

    def test_dt_not_duplicated_in_head_state_dict(self):
        shared, head, *_ = self._build()
        head_keys = list(head.state_dict().keys())
        assert all(
            "dt" not in k for k in head_keys
        ), f"dt leaked into head.state_dict: {[k for k in head_keys if 'dt' in k]}"

    def test_dt_appears_once_in_container_state_dict(self):
        from compressai.models._helpers.dictionary_context import (
            SharedDictionary,
            build_dictionary_mean_scale_head,
        )

        m, slice_ch, support_count = 32, 8, 2
        support_ch = 2 * m + slice_ch * support_count

        class _Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_dictionary = SharedDictionary(dict_num=8, dictionary_dim=64)
                self.heads = nn.ModuleDict(
                    {
                        f"y{k}": build_dictionary_mean_scale_head(
                            slice_ch=slice_ch,
                            support_ch=support_ch,
                            shared_dictionary=self.shared_dictionary,
                            dict_output_ch=m,
                            cross_attention_kwargs={"head_num": 4, "mlp_rate": 2},
                            widths=(16,),
                        )
                        for k in range(3)
                    }
                )

        container = _Container()
        dt_keys = [k for k in container.state_dict() if k.endswith(".dt")]
        assert dt_keys == [
            "shared_dictionary.dt"
        ], f"expected single shared_dictionary.dt path, got: {dt_keys}"


class TestOLP:
    @staticmethod
    def test_forward_shape_square():
        from compressai.models._helpers.auxt import OLP

        m = OLP(8, 8)
        out = m(torch.randn(2, 8))
        assert out.shape == (2, 8)

    @staticmethod
    def test_loss_returns_scalar_for_each_aspect_ratio():
        from compressai.models._helpers.auxt import OLP

        for in_dim, out_dim in [(8, 8), (16, 4), (4, 16)]:
            m = OLP(in_dim, out_dim)
            loss = m.loss()
            assert loss.dim() == 0, f"OLP({in_dim}, {out_dim}).loss() must be scalar"
            assert torch.isfinite(loss)

    @staticmethod
    def test_state_dict_round_trip():
        from compressai.models._helpers.auxt import OLP

        m = OLP(8, 8)
        m2 = OLP(8, 8)
        m2.load_state_dict(m.state_dict(), strict=True)
        x = torch.randn(2, 8)
        assert torch.allclose(m(x), m2(x))


class TestWLSiWLS:
    @staticmethod
    def test_wls_iwls_shapes_and_round_trip():
        pytest.importorskip("pytorch_wavelets")
        from compressai.models._helpers.auxt import WLS, iWLS

        wls = WLS(in_dim=3, out_dim=8)
        iwls = iWLS(in_dim=8, out_dim=3)
        x = torch.randn(2, 3, 16, 16)
        y = wls(x)
        # WLS halves spatial size (DWT) and produces out_dim channels.
        assert y.shape == (2, 8, 8, 8)
        z = iwls(y)
        assert z.shape == x.shape

        # state_dict round-trip on WLS.
        wls2 = WLS(in_dim=3, out_dim=8)
        wls2.load_state_dict(wls.state_dict(), strict=True)
        assert torch.allclose(wls(x), wls2(x))

    @staticmethod
    def test_aux_loss_returns_zero_when_no_olp_present():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import aux_loss

        # A toy model with no OLP submodules; aux_loss should return a 0-d
        # zero Tensor so callers can unconditionally add it to the objective.
        model = _nn.Sequential(_nn.Linear(8, 8))
        loss = aux_loss(model)
        assert loss.dim() == 0
        assert loss.item() == 0.0

    @staticmethod
    def test_aux_loss_aggregates_olp_modules():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import OLP, aux_loss

        class _Container(_nn.Module):
            def __init__(self):
                super().__init__()
                self.a = OLP(8, 8)
                self.b = OLP(16, 4)

        c = _Container()
        expected = c.a.loss() + c.b.loss()
        assert torch.allclose(aux_loss(c), expected)


class TestForwardWithAuxt:
    @staticmethod
    def test_collapses_to_transform_when_aux_layers_none():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import forward_with_auxt

        transform = _nn.Sequential(_nn.Conv2d(3, 4, 1), _nn.Conv2d(4, 5, 1))
        x = torch.randn(2, 3, 4, 4)
        with torch.no_grad():
            assert torch.allclose(
                forward_with_auxt(transform, None, (), x), transform(x)
            )

    @staticmethod
    def test_sums_auxt_at_merge_positions():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import forward_with_auxt

        def _identity_conv(ch):
            conv = _nn.Conv2d(ch, ch, 1, bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.eye(ch).view(ch, ch, 1, 1))
            return conv

        transform = _nn.Sequential(*(_identity_conv(3) for _ in range(4)))
        aux = _nn.ModuleList([_identity_conv(3), _identity_conv(3)])
        x = torch.randn(1, 3, 2, 2)
        out = forward_with_auxt(transform, aux, (1, 3), x)
        assert torch.allclose(out, 3 * x)

    @staticmethod
    def test_raises_when_merge_positions_underrun_aux_depth():
        import torch.nn as _nn

        from compressai.models._helpers.auxt import forward_with_auxt

        transform = _nn.Sequential(_nn.Conv2d(3, 3, 1), _nn.Conv2d(3, 3, 1))
        aux = _nn.ModuleList([_nn.Conv2d(3, 3, 1), _nn.Conv2d(3, 3, 1)])
        x = torch.randn(1, 3, 2, 2)
        with pytest.raises(RuntimeError, match="merge positions"):
            forward_with_auxt(transform, aux, (0,), x)


class TestAuxtStateDictHelpers:
    @staticmethod
    def test_has_auxt_state():
        from compressai.models._helpers.auxt import has_auxt_state

        assert has_auxt_state({"AuxT_enc.0.olp.linear.weight": torch.zeros(2)})
        assert has_auxt_state({"AuxT_dec.3.scaling_factors": torch.zeros(2)})
        assert not has_auxt_state({"g_a.0.weight": torch.zeros(2)})

    @staticmethod
    def test_is_auxt_wavelet_buffer_key():
        from compressai.models._helpers.auxt import is_auxt_wavelet_buffer_key

        assert is_auxt_wavelet_buffer_key("AuxT_enc.0.dwt.transform.h0_col")
        assert is_auxt_wavelet_buffer_key("AuxT_dec.0.idwt.inverse.g0_col")
        assert not is_auxt_wavelet_buffer_key("AuxT_enc.0.olp.linear.weight")
        assert not is_auxt_wavelet_buffer_key("g_a.0.weight")

    @staticmethod
    def test_is_auxt_upstream_wavelet_buffer_key():
        from compressai.models._helpers.auxt import (
            is_auxt_upstream_wavelet_buffer_key,
        )

        for suffix in ("w_ll", "w_lh", "w_hl", "w_hh"):
            assert is_auxt_upstream_wavelet_buffer_key(f"AuxT_enc.0.dwt.{suffix}")
        assert is_auxt_upstream_wavelet_buffer_key("AuxT_dec.0.idwt.filters")
        assert not is_auxt_upstream_wavelet_buffer_key(
            "AuxT_enc.0.dwt.transform.h0_col"
        )
        assert not is_auxt_upstream_wavelet_buffer_key("AuxT_enc.0.olp.linear.weight")

    @staticmethod
    def test_normalize_upstream_auxt_key_renames_pascal_olp():
        from compressai.models._helpers.auxt import normalize_upstream_auxt_key

        assert (
            normalize_upstream_auxt_key("AuxT_enc.0.OLP.linear.weight")
            == "AuxT_enc.0.olp.linear.weight"
        )
        assert (
            normalize_upstream_auxt_key("AuxT_dec.3.OLP.linear.bias")
            == "AuxT_dec.3.olp.linear.bias"
        )
        assert normalize_upstream_auxt_key("g_a.0.weight") is None
