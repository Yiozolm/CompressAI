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

from compressai.layers import is_freia_available, is_pytorch_wavelets_available
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelSliceLatentCodec,
    HierarchicalProgressiveLatentCodec,
    MLICPlusPlusLatentCodec,
    WeChARMLatentCodec,
)
from compressai.models import (
    CCAModel,
    CMIC,
    DCAE,
    FrequencyAwareTransFormer,
    HPCM,
    Informer,
    InvCompress,
    MambaIC,
    MambaVC,
    MLICPlusPlus,
    SAAF,
    ShiftLIC,
    TCM,
    TIC,
    TinyLIC,
    WACNN,
    WeConvene,
)
from compressai.models.google import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    MeanScaleHyperprior,
    ScaleHyperprior,
    get_scale_table,
)
from compressai.models.utils import (
    _update_registered_buffer,
    find_named_module,
    update_registered_buffers,
)
from compressai.models.vbr import ScaleHyperpriorVbr
from compressai.models.video.google import ScaleSpaceFlow


class DummyCompressionModel(CompressionModel):
    def __init__(self, entropy_bottleneck_channels):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)


class TestCompressionModel:
    def test_parameters(self):
        model = DummyCompressionModel(32)
        assert len(list(model.parameters())) == 15
        with pytest.raises(NotImplementedError):
            model(torch.rand(1))

    def test_init(self):
        class Model(DummyCompressionModel):
            def __init__(self):
                super().__init__(3)
                self.conv = nn.Conv2d(3, 3, 3)
                self.deconv = nn.ConvTranspose2d(3, 3, 3)
                self.original_conv = self.conv.weight
                self.original_deconv = self.deconv.weight

        model = Model()
        nn.init.kaiming_normal_(model.original_conv)
        nn.init.kaiming_normal_(model.original_deconv)

        assert torch.allclose(model.original_conv, model.conv.weight)
        assert torch.allclose(model.original_deconv, model.deconv.weight)


class TestModels:
    def test_factorized_prior(self):
        model = FactorizedPrior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

    def test_scale_hyperprior(self, tmpdir):
        model = ScaleHyperprior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperprior(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = ScaleHyperprior.from_state_dict(torch.load(filepath))
            assert model.N == loaded.N and model.M == loaded.M

    def test_scale_hyperprior_vbr(self, tmpdir):
        model = ScaleHyperpriorVbr(128, 192, vr_entbttlnck=True)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperpriorVbr(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = ScaleHyperpriorVbr.from_state_dict(torch.load(filepath))
            assert model.N == loaded.N and model.M == loaded.M

    def test_mean_scale_hyperprior(self):
        model = MeanScaleHyperprior(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

    def test_jarhp(self, tmpdir):
        model = JointAutoregressiveHierarchicalPriors(128, 192)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape

        y_likelihoods_shape = out["likelihoods"]["y"].shape
        assert y_likelihoods_shape[0] == x.shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x.shape[2] / 2**4
        assert y_likelihoods_shape[3] == x.shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"]["z"].shape
        assert z_likelihoods_shape[0] == x.shape[0]
        assert z_likelihoods_shape[1] == 128
        assert z_likelihoods_shape[2] == x.shape[2] / 2**6
        assert z_likelihoods_shape[3] == x.shape[3] / 2**6

        for sz in [(128, 128), (128, 192), (192, 128)]:
            model = JointAutoregressiveHierarchicalPriors(*sz)
            filepath = tmpdir.join("model.pth.rar").strpath
            torch.save(model.state_dict(), filepath)
            loaded = JointAutoregressiveHierarchicalPriors.from_state_dict(
                torch.load(filepath)
            )
            assert model.N == loaded.N and model.M == loaded.M

    def test_scale_space_flow(self):
        model = ScaleSpaceFlow()
        x = [torch.rand(1, 3, 128, 128), torch.rand(1, 3, 128, 128)]
        out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "keyframe" in out["likelihoods"][0]
        assert "y" in out["likelihoods"][0]["keyframe"]
        assert "z" in out["likelihoods"][0]["keyframe"]

        assert "motion" in out["likelihoods"][1]
        assert "y" in out["likelihoods"][1]["motion"]
        assert "z" in out["likelihoods"][1]["motion"]

        assert "residual" in out["likelihoods"][1]
        assert "y" in out["likelihoods"][1]["residual"]
        assert "z" in out["likelihoods"][1]["residual"]

        assert out["x_hat"][0].shape == x[0].shape
        assert out["x_hat"][1].shape == x[1].shape

        y_likelihoods_shape = out["likelihoods"][0]["keyframe"]["y"].shape
        assert y_likelihoods_shape[0] == x[0].shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x[0].shape[2] / 2**4
        assert y_likelihoods_shape[3] == x[0].shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"][0]["keyframe"]["z"].shape
        assert z_likelihoods_shape[0] == x[0].shape[0]
        assert z_likelihoods_shape[1] == 192
        assert z_likelihoods_shape[2] == x[0].shape[2] / 2**7  # (128x128 input)
        assert z_likelihoods_shape[3] == x[0].shape[3] / 2**7

        y_likelihoods_shape = out["likelihoods"][1]["motion"]["y"].shape
        assert y_likelihoods_shape[0] == x[1].shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x[1].shape[2] / 2**4
        assert y_likelihoods_shape[3] == x[1].shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"][1]["motion"]["z"].shape
        assert z_likelihoods_shape[0] == x[1].shape[0]
        assert z_likelihoods_shape[1] == 192
        assert z_likelihoods_shape[2] == x[1].shape[2] / 2**7  # (128x128 input)
        assert z_likelihoods_shape[3] == x[1].shape[3] / 2**7

        y_likelihoods_shape = out["likelihoods"][1]["residual"]["y"].shape
        assert y_likelihoods_shape[0] == x[1].shape[0]
        assert y_likelihoods_shape[1] == 192
        assert y_likelihoods_shape[2] == x[1].shape[2] / 2**4
        assert y_likelihoods_shape[3] == x[1].shape[3] / 2**4

        z_likelihoods_shape = out["likelihoods"][1]["residual"]["z"].shape
        assert z_likelihoods_shape[0] == x[1].shape[0]
        assert z_likelihoods_shape[1] == 192
        assert z_likelihoods_shape[2] == x[1].shape[2] / 2**7  # (128x128 input)
        assert z_likelihoods_shape[3] == x[1].shape[3] / 2**7

    def test_mlicpp(self):
        model = MLICPlusPlus(N=16, M=32, slice_num=4, context_window=3)
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        assert isinstance(model.latent_codec, MLICPlusPlusLatentCodec)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)

        loaded = MLICPlusPlus.from_state_dict(model.state_dict())
        assert loaded.N == 16
        assert loaded.M == 32
        assert loaded.slice_num == 4
        assert loaded.context_window == 3

        legacy_state_dict = {
            (
                key.removeprefix("latent_codec.")
                if key.startswith("latent_codec.")
                else key
            ): value
            for key, value in model.state_dict().items()
        }
        legacy_loaded = MLICPlusPlus.from_state_dict(legacy_state_dict)
        assert legacy_loaded.N == 16
        assert legacy_loaded.M == 32
        assert legacy_loaded.slice_num == 4
        assert legacy_loaded.context_window == 3

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 2
        assert len(compressed["strings"][0]) == x.size(0)
        assert len(compressed["strings"][1]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_cmic_missing_dependency(self):
        if is_pytorch_wavelets_available():
            pytest.skip("pytorch_wavelets is installed.")

        with pytest.raises(ModuleNotFoundError, match="pytorch_wavelets"):
            CMIC(N=16, M=32)

    @pytest.mark.skipif(
        not is_pytorch_wavelets_available(),
        reason="pytorch_wavelets is not installed",
    )
    def test_cmic(self):
        model = CMIC(
            N=16,
            M=32,
            groups=[8, 8, 8, 8],
            stage_dims=(16, 16, 24),
            stage_depths=(1, 1),
            num_heads=(4, 4),
            d_state=4,
            window_size=4,
            cluster_num=16,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)

        loaded = CMIC.from_state_dict(model.state_dict())
        assert loaded.N == 16
        assert loaded.M == 32
        assert loaded.groups == [8, 8, 8, 8]
        assert loaded.stage_dims == (16, 16, 24)
        assert loaded.stage_depths == (1, 1)
        assert loaded.num_heads == (4, 4)
        assert loaded.d_state == 4
        assert loaded.window_size == 4
        assert loaded.cluster_num == 16

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == len(model.groups) * 2 + 1
        assert len(compressed["strings"][-1]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_tcm(self):
        model = TCM(
            config=[1, 1, 1, 1, 1, 1],
            head_dim=[8, 8, 8, 8, 8, 8],
            N=16,
            M=32,
            hyper_channels=16,
            hyper_head_dim=8,
            num_slices=4,
            max_support_slices=2,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        assert isinstance(model.latent_codec, ChannelSliceLatentCodec)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)

        loaded = TCM.from_state_dict(model.state_dict())
        assert loaded.N == 16
        assert loaded.M == 32
        assert loaded.num_slices == 4
        assert loaded.max_support_slices == 2

        legacy_state_dict = {
            (
                key.removeprefix("latent_codec.")
                if key.startswith("latent_codec.")
                else key
            ): value
            for key, value in model.state_dict().items()
        }
        legacy_loaded = TCM.from_state_dict(legacy_state_dict)
        assert legacy_loaded.N == 16
        assert legacy_loaded.M == 32
        assert legacy_loaded.num_slices == 4
        assert legacy_loaded.max_support_slices == 2

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 2
        assert len(compressed["strings"][0]) == x.size(0)
        assert len(compressed["strings"][1]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_weconvene_missing_dependency(self):
        if is_pytorch_wavelets_available():
            pytest.skip("pytorch_wavelets is installed.")

        with pytest.raises(ModuleNotFoundError, match="pytorch_wavelets"):
            WeConvene(N=16, M=32)

    @pytest.mark.skipif(
        not is_pytorch_wavelets_available(),
        reason="pytorch_wavelets is not installed",
    )
    def test_weconvene(self):
        model = WeConvene(
            N=16,
            M=32,
            hyper_channels=16,
            num_slices=4,
            max_support_slices=2,
            support_window_size=4,
            support_head_dim=4,
            support_attention_dim=16,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        assert isinstance(model.latent_codec, WeChARMLatentCodec)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y_low" in out["likelihoods"]
        assert "y_high" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y_low"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["y_high"].shape == (1, 96, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)

        loaded = WeConvene.from_state_dict(model.state_dict())
        assert loaded.N == 16
        assert loaded.M == 32
        assert loaded.hyper_channels == 16
        assert loaded.num_slices == 4
        assert loaded.max_support_slices == 2
        assert loaded.support_window_size == 4
        assert loaded.support_head_dim == 4
        assert loaded.support_attention_dim == 16

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 3
        assert len(compressed["strings"][0]) == x.size(0)
        assert len(compressed["strings"][1]) == x.size(0)
        assert len(compressed["strings"][2]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_tcm_with_cca(self):
        model = TCM(
            config=[1, 1, 1, 1, 1, 1],
            head_dim=[8, 8, 8, 8, 8, 8],
            N=16,
            M=32,
            hyper_channels=16,
            hyper_head_dim=8,
            num_slices=4,
            max_support_slices=2,
            use_cca=True,
            cca_hidden_channels=32,
            cca_num_layers=2,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert "aux_likelihoods" in out
        assert out["aux_likelihoods"]["y_aux"].shape == (1, 32, 4, 4)
        assert out["aux_likelihoods"]["y_cca"].shape == (1, 32, 4, 4)

        loaded = TCM.from_state_dict(model.state_dict())
        assert loaded.use_cca

    @pytest.mark.skipif(
        not is_pytorch_wavelets_available(),
        reason="pytorch_wavelets is not installed",
    )
    def test_tcm_with_auxt(self):
        model = TCM(
            config=[1, 1, 1, 1, 1, 1],
            head_dim=[8, 8, 8, 8, 8, 8],
            N=16,
            M=32,
            hyper_channels=16,
            hyper_head_dim=8,
            num_slices=4,
            max_support_slices=2,
            use_auxt=True,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert model.use_auxt
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)
        assert model.ortho_loss().ndim == 0

        loaded = TCM.from_state_dict(model.state_dict())
        assert loaded.use_auxt

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert decoded["x_hat"].shape == x.shape

    def test_tcm_with_auxt_missing_dependency(self):
        if is_pytorch_wavelets_available():
            pytest.skip("pytorch_wavelets is installed.")

        with pytest.raises(ModuleNotFoundError, match="pytorch_wavelets"):
            TCM(
                config=[1, 1, 1, 1, 1, 1],
                head_dim=[8, 8, 8, 8, 8, 8],
                N=16,
                M=32,
                hyper_channels=16,
                hyper_head_dim=8,
                num_slices=4,
                max_support_slices=2,
                use_auxt=True,
            )

    def test_wacnn_with_cca(self):
        model = WACNN(
            N=16,
            M=32,
            num_slices=4,
            max_support_slices=2,
            use_cca=True,
            cca_hidden_channels=32,
            cca_num_layers=2,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert "aux_likelihoods" in out
        assert out["aux_likelihoods"]["y_aux"].shape == (1, 32, 4, 4)
        assert out["aux_likelihoods"]["y_cca"].shape == (1, 32, 4, 4)

        loaded = WACNN.from_state_dict(model.state_dict())
        assert loaded.use_cca

    def test_mambavc(self):
        model = MambaVC(
            depths=(1, 1, 1, 1),
            N=16,
            M=32,
            hyper_channels=16,
            num_slices=4,
            max_support_slices=2,
            window_size=4,
            support_head_dim=8,
            support_attention_dim=16,
            ssm_d_state=4,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        assert isinstance(model.latent_codec, ChannelSliceLatentCodec)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)

        loaded = MambaVC.from_state_dict(model.state_dict())
        assert loaded.N == 16
        assert loaded.M == 32
        assert loaded.hyper_channels == 16
        assert loaded.num_slices == 4
        assert loaded.max_support_slices == 2

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 2
        assert len(compressed["strings"][0]) == x.size(0)
        assert len(compressed["strings"][1]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_mambaic(self):
        model = MambaIC(
            depths=(1, 1, 1, 1),
            N=16,
            M=32,
            hyper_channels=16,
            num_slices=4,
            max_support_slices=2,
            context_depths=(1, 1, 1, 1),
            window_size=4,
            support_head_dim=8,
            context_head_dim=4,
            support_attention_dim=16,
            context_attention_dim=16,
            ssm_d_state=4,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)

        loaded = MambaIC.from_state_dict(model.state_dict())
        assert loaded.N == 16
        assert loaded.M == 32
        assert loaded.hyper_channels == 16
        assert loaded.num_slices == 4
        assert loaded.max_support_slices == 2

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 2
        assert len(compressed["strings"][0]) == model.num_slices
        assert len(compressed["strings"][1]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_dcae(self):
        model = DCAE(
            head_dim=[8, 8, 8, 8, 8, 8],
            N=16,
            M=32,
            hyper_channels=16,
            num_slices=4,
            max_support_slices=2,
            feature_dims=(16, 16, 32),
            block_num=(1, 1, 1),
            dict_num=16,
            dict_head_num=4,
            window_size=4,
            hyper_window_size=2,
            hyper_head_dim=8,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)

        loaded = DCAE.from_state_dict(model.state_dict())
        assert loaded.N == 16
        assert loaded.M == 32
        assert loaded.num_slices == 4
        assert loaded.max_support_slices == 2

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 2
        assert len(compressed["strings"][0]) == x.size(0)
        assert len(compressed["strings"][1]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_ftic(self):
        model = FrequencyAwareTransFormer(
            config=(1, 1, 1, 1, 1, 1),
            num_heads=(4, 4, 8, 8, 4, 4),
            feature_dims=(16, 24, 32),
            hyper_hidden_channels=32,
            hyper_channels=16,
            M=32,
            num_slices=4,
            hyper_num_heads=8,
            tca_depth=2,
            tca_ratio=2,
            tca_num_heads=16,
            window_size=4,
            fm_window_size=8,
            hyper_window_size=2,
            hyper_fm_window_size=4,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)

        loaded = FrequencyAwareTransFormer.from_state_dict(model.state_dict())
        assert loaded.M == 32
        assert loaded.num_slices == 4
        assert loaded.config == (1, 1, 1, 1, 1, 1)
        assert loaded.num_heads == (4, 4, 8, 8, 4, 4)
        assert loaded.tca_depth == 2
        assert loaded.tca_ratio == 2

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 2
        assert len(compressed["strings"][0]) == model.num_slices
        assert len(compressed["strings"][1]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_saaf(self):
        model = SAAF(
            head_dim=[8, 8, 8, 8, 8, 8],
            N=16,
            M=32,
            hyper_channels=16,
            num_slices=4,
            max_support_slices=2,
            feature_dims=(16, 16, 32),
            block_num=(1, 1, 1),
            dict_num=16,
            dict_head_num=4,
            window_size=4,
            hyper_window_size=2,
            hyper_head_dim=8,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert "diffusion_loss" in out
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)

        loaded = SAAF.from_state_dict(model.state_dict())
        assert loaded.N == 16
        assert loaded.M == 32
        assert loaded.num_slices == 4
        assert loaded.max_support_slices == 2

        model.train()
        ortho_loss = model.ortho_loss()
        assert ortho_loss.ndim == 0
        model.eval()

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 2
        assert len(compressed["strings"][0]) == x.size(0)
        assert len(compressed["strings"][1]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_cca(self):
        model = CCAModel(
            latent_channels=32,
            hyper_channels=16,
            slice_proportions=(1, 1, 1, 1),
            encoder_dims=(16, 16, 24),
            encoder_layers=(1, 1, 1),
            em_hidden_channels=24,
            em_num_layers=1,
            cca_training=True,
        )
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 16, 1, 1)
        assert out["aux_likelihoods"]["y_aux"].shape == (1, 32, 4, 4)
        assert out["aux_likelihoods"]["y_cca"].shape == (1, 32, 4, 4)

        loaded = CCAModel.from_state_dict(model.state_dict())
        assert loaded.M == 32
        assert loaded.N == 16
        assert loaded.num_slices == 4
        assert loaded.cca_training is True

        infer = CCAModel(
            latent_channels=32,
            hyper_channels=16,
            slice_proportions=(1, 1, 1, 1),
            encoder_dims=(16, 16, 24),
            encoder_layers=(1, 1, 1),
            em_hidden_channels=24,
            em_num_layers=1,
            cca_training=False,
        )
        with torch.no_grad():
            out_no_aux = infer(x)
        assert out_no_aux["aux_likelihoods"] is None

    def test_tinylic(self):
        # Default config (N=128, M=320). Use 256x256 — the upstream training
        # / inference size. Eval + no_grad keeps memory bounded.
        model = TinyLIC(N=128, M=320)
        model.eval()
        x = torch.rand(1, 3, 256, 256)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 320, 16, 16)
        assert out["likelihoods"]["z"].shape == (1, 192, 4, 4)

        # state_dict roundtrip via from_state_dict (compressai layout).
        loaded = TinyLIC.from_state_dict(model.state_dict())
        assert loaded.N == 128
        assert loaded.M == 320

        # Upstream-style state_dict (entropy keys at top level) must also load
        # via the load_state_dict pre-pass remap.
        upstream_sd = {}
        for k, v in model.state_dict().items():
            if k.startswith("latent_codec."):
                upstream_sd[k[len("latent_codec.") :]] = v
            else:
                upstream_sd[k] = v
        loaded_upstream = TinyLIC.from_state_dict(upstream_sd)
        assert loaded_upstream.N == 128
        assert loaded_upstream.M == 320
        for k, v in model.state_dict().items():
            assert torch.equal(v, loaded_upstream.state_dict()[k]), k

        # Bitstream roundtrip.
        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 2
        assert len(compressed["strings"][0]) == 1  # one packed y bytestring
        assert len(compressed["strings"][1]) == x.size(0)  # per-sample z
        assert decoded["x_hat"].shape == x.shape

    def test_tic(self):
        # Default config (N=128, M=192). Use 256x256 to give the deepest g_a5
        # stage (32x32) a window-size==8 attention map.
        model = TIC(N=128, M=192)
        model.eval()
        x = torch.rand(1, 3, 256, 256)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 192, 16, 16)
        assert out["likelihoods"]["z"].shape == (1, 128, 4, 4)

        # state_dict roundtrip via from_state_dict.
        loaded = TIC.from_state_dict(model.state_dict())
        assert loaded.N == 128
        assert loaded.M == 192

        # Smaller config + minimum valid input (128, divisible by 64) to keep
        # the autoregressive bitstream roundtrip fast.
        small = TIC(N=32, M=48)
        small.eval()
        small.update(force=True)
        xs = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            compressed = small.compress(xs)
            decoded = small.decompress(compressed["strings"], compressed["shape"])
        assert len(compressed["strings"]) == 2
        assert decoded["x_hat"].shape == xs.shape

    @pytest.mark.parametrize("variant", ["small", "middle", "large"])
    def test_shiftlic(self, variant):
        # All three variants share the encoder/hyper structure (N=192, M=320);
        # large additionally exercises the staged latent codec branch.
        model = ShiftLIC(variant=variant, N=192, M=320)
        model.eval()
        x = torch.rand(1, 3, 256, 256)

        with torch.no_grad():
            out = model(x)

        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 320, 16, 16)
        assert out["likelihoods"]["z"].shape == (1, 192, 4, 4)

        # state_dict roundtrip via from_state_dict (variant inferred).
        loaded = ShiftLIC.from_state_dict(model.state_dict())
        assert loaded.variant == variant
        assert loaded.N == 192 and loaded.M == 320

        # Upstream-style state_dict (codec keys at top level) only matters
        # for ``large``; small/middle don't use the codec.
        if variant == "large":
            upstream_sd = {}
            for k, v in model.state_dict().items():
                if k.startswith("latent_codec."):
                    upstream_sd[k[len("latent_codec.") :]] = v
                else:
                    upstream_sd[k] = v
            loaded_upstream = ShiftLIC.from_state_dict(upstream_sd)
            assert loaded_upstream.variant == "large"
            for k, v in model.state_dict().items():
                assert torch.equal(v, loaded_upstream.state_dict()[k]), k

    def test_invcompress_missing_dependency(self):
        if is_freia_available():
            pytest.skip("FrEIA is installed.")

        with pytest.raises(ModuleNotFoundError, match="FrEIA"):
            InvCompress(N=16)

    @pytest.mark.skipif(not is_freia_available(), reason="FrEIA is not installed")
    def test_invcompress(self):
        model = InvCompress(N=192)
        model.eval()
        x = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 192, 4, 4)
        assert out["likelihoods"]["z"].shape == (1, 192, 1, 1)

        loaded = InvCompress.from_state_dict(model.state_dict())
        assert loaded.N == 192
        assert loaded.M == 192

        model.update(force=True)
        with torch.no_grad():
            compressed = model.compress(x)
            decoded = model.decompress(compressed["strings"], compressed["shape"])

        assert len(compressed["strings"]) == 2
        assert len(compressed["strings"][0]) == x.size(0)
        assert len(compressed["strings"][1]) == x.size(0)
        assert decoded["x_hat"].shape == x.shape

    def test_hpcm_base(self):
        model = HPCM(
            N=32,
            M=32,
            g_a_depth=1,
            g_s_depth=1,
            y_prior_depth=1,
            use_attention=True,
            attn_window_s1=2,
            attn_window_s2=4,
            attn_window_s3=8,
            attn_num_heads=4,
        )
        model.eval()
        x = torch.rand(1, 3, 128, 128)

        assert isinstance(model.latent_codec, HierarchicalProgressiveLatentCodec)

        with torch.no_grad():
            out = model(x)

        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 32, 8, 8)
        assert out["likelihoods"]["z"].shape == (1, 32, 2, 2)

        loaded = HPCM.from_state_dict(
            model.state_dict(),
            attn_window_s1=2,
            attn_window_s2=4,
            attn_window_s3=8,
            attn_num_heads=4,
        )
        assert loaded.N == 32
        assert loaded.M == 32
        assert loaded.g_a_depth == 1
        assert loaded.g_s_depth == 1
        assert loaded.y_prior_depth == 1
        assert loaded.use_attention is True

        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])

    def test_hpcm_phi_no_attention(self):
        model = HPCM(
            N=32,
            M=32,
            g_a_depth=1,
            g_s_depth=1,
            y_prior_depth=1,
            use_attention=False,
        )
        model.eval()
        x = torch.rand(1, 3, 128, 128)

        with torch.no_grad():
            out = model(x)
        assert out["x_hat"].shape == x.shape
        assert "y" in out["likelihoods"]
        assert "z" in out["likelihoods"]

        loaded = HPCM.from_state_dict(model.state_dict())
        assert loaded.use_attention is False
        with torch.no_grad():
            out_loaded = loaded(x)
        assert torch.allclose(out["x_hat"], out_loaded["x_hat"])

    def test_informer(self):
        # Default config (N=192, M=192, num_global=8). 256x256 → y is 16x16.
        model = Informer(N=192, M=192, num_global=8)
        model.eval()
        x = torch.rand(1, 3, 256, 256)

        with torch.no_grad():
            out = model(x)

        assert "x_hat" in out
        assert "likelihoods" in out
        assert out["x_hat"].shape == x.shape
        assert out["likelihoods"]["y"].shape == (1, 192, 16, 16)
        assert out["likelihoods"]["l_z"].shape == (1, 12, 16, 16)
        assert out["likelihoods"]["g_z"].shape == (1, 192, 1, 1)

        # state_dict roundtrip via from_state_dict (N/M/num_global inferred).
        loaded = Informer.from_state_dict(model.state_dict())
        assert loaded.N == 192
        assert loaded.M == 192
        assert loaded.num_global == 8

        # Smaller config to keep the autoregressive bitstream roundtrip fast.
        small = Informer(N=64, M=64, num_global=4)
        small.eval()
        small.update(force=True)
        xs = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            compressed = small.compress(xs)
            decoded = small.decompress(compressed["strings"], compressed["shape"])
        assert len(compressed["strings"]) == 3  # y, local_z, global_z
        assert len(compressed["strings"][0]) == xs.size(0)
        assert len(compressed["strings"][1]) == xs.size(0)
        assert len(compressed["strings"][2]) == xs.size(0)
        assert decoded["x_hat"].shape == xs.shape


def test_scale_table_default():
    table = get_scale_table()
    assert SCALES_MIN == 0.11
    assert SCALES_MAX == 256
    assert SCALES_LEVELS == 64
    assert table[0] == SCALES_MIN
    assert table[-1] == SCALES_MAX
    assert len(table.size()) == 1
    assert table.size(0) == SCALES_LEVELS


def test_scale_table_custom():
    table = get_scale_table(0.02, 1337, 32)
    assert pytest.approx(table[0].item()) == 0.02
    assert pytest.approx(table[-1].item()) == 1337
    assert len(table.size()) == 1
    assert table.size(0) == 32


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 1)


def test_find_named_module():
    assert find_named_module(Foo(), "conv3") is None
    foo = Foo()
    found = find_named_module(foo, "conv1")
    assert found == foo.conv1


def test_update_registered_buffers():
    foo = Foo()
    with pytest.raises(ValueError):
        update_registered_buffers(foo, "conv1", ["qweight"], {})


def test_update_registered_buffer():
    foo = Foo()

    # non-registered buffer
    state_dict = foo.state_dict()
    state_dict["conv1.wweight"] = torch.rand(3)
    with pytest.raises(RuntimeError):
        _update_registered_buffer(
            foo.conv1, "wweight", "conv1.wweight", state_dict, policy="resize"
        )
    with pytest.raises(RuntimeError):
        _update_registered_buffer(
            foo.conv1, "wweight", "conv1.wweight", state_dict, policy="resize_if_empty"
        )
