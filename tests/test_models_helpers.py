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

import torch
import torch.nn as nn

from compressai.models._helpers.channel_context import (
    MeanScaleContextHead,
    build_mean_scale_head,
)


class TestMeanScaleContextHead:
    def test_forward_shape_concatenates_mean_and_scale(self):
        slice_ch, support_ch = 4, 12
        head = build_mean_scale_head(slice_ch, support_ch, widths=(8, 8))
        x = torch.randn(2, support_ch, 4, 4)
        out = head(x)
        assert out.shape == (2, 2 * slice_ch, 4, 4)

    def test_state_dict_paths_split_mean_and_scale(self):
        head = build_mean_scale_head(4, 12, widths=(8, 8))
        keys = set(head.state_dict().keys())
        assert any(k.startswith("mean_cc.") for k in keys)
        assert any(k.startswith("scale_cc.") for k in keys)
        # No support transforms by default -> no associated state.
        assert not any(k.startswith("mean_support_transform.") for k in keys)
        assert not any(k.startswith("scale_support_transform.") for k in keys)

    def test_support_transform_factory_wraps_inputs(self):
        # Use 1x1 conv that preserves channel count.
        def factory(c_in, c_out):
            return nn.Conv2d(c_in, c_out, 1)

        head = build_mean_scale_head(
            4, 12, widths=(8, 8), support_transform_factory=factory
        )
        keys = set(head.state_dict().keys())
        assert any(k.startswith("mean_support_transform.") for k in keys)
        assert any(k.startswith("scale_support_transform.") for k in keys)
        # mean and scale support transforms are independent instances (not shared).
        assert head.mean_support_transform is not head.scale_support_transform

    def test_direct_construction_round_trip(self):
        torch.manual_seed(0)
        mean_cc = nn.Conv2d(12, 4, 1)
        scale_cc = nn.Conv2d(12, 4, 1)
        head = MeanScaleContextHead(mean_cc=mean_cc, scale_cc=scale_cc)
        rebuilt = MeanScaleContextHead(
            mean_cc=nn.Conv2d(12, 4, 1), scale_cc=nn.Conv2d(12, 4, 1)
        )
        rebuilt.load_state_dict(head.state_dict())
        x = torch.randn(2, 12, 4, 4)
        with torch.no_grad():
            assert torch.allclose(head(x), rebuilt(x))

    def test_side_split_routes_means_to_mean_cc_and_scales_to_scale_cc(self):
        # side_split=8 means input is cat(latent_means(8), latent_scales(8), prev_y_hat(4));
        # mean_cc should see cat(latent_means(8), prev_y_hat(4)) = 12 channels;
        # scale_cc same width but reading latent_scales instead of latent_means.
        torch.manual_seed(0)
        head = build_mean_scale_head(
            slice_ch=4, support_ch=20, widths=(8,), side_split=8
        )
        # Sub-network input width = support_ch - side_split = 12.
        first_mean_conv = next(m for m in head.mean_cc if isinstance(m, nn.Conv2d))
        first_scale_conv = next(m for m in head.scale_cc if isinstance(m, nn.Conv2d))
        assert first_mean_conv.in_channels == 12
        assert first_scale_conv.in_channels == 12

        latent_means = torch.randn(2, 8, 4, 4)
        latent_scales = torch.randn(2, 8, 4, 4)
        prev_y_hat = torch.randn(2, 4, 4, 4)
        x = torch.cat([latent_means, latent_scales, prev_y_hat], dim=1)
        with torch.no_grad():
            head_out = head(x)
        assert head_out.shape == (2, 8, 4, 4)
        # Verify routing: mean_cc(cat(latent_means, prev_y_hat)) appears as
        # the second half of head_out (chunks=("scales","means")).
        with torch.no_grad():
            expected_mean = head.mean_cc(torch.cat([latent_means, prev_y_hat], dim=1))
            expected_scale = head.scale_cc(
                torch.cat([latent_scales, prev_y_hat], dim=1)
            )
            scale_out, mean_out = head_out.chunk(2, dim=1)
        assert torch.allclose(scale_out, expected_scale)
        assert torch.allclose(mean_out, expected_mean)


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
        # Output: cat([scale, mean]) → 2 * slice_ch
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
