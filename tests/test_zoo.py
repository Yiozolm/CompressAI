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

from compressai.layers import is_freia_available, is_pytorch_wavelets_available
from compressai.models import (
    CCAModel,
    CMIC,
    Cheng2020Anchor,
    Cheng2020Attention,
    DCAE,
    FactorizedPrior,
    FrequencyAwareTransFormer,
    HPCM,
    InvCompress,
    JointAutoregressiveHierarchicalPriors,
    MambaIC,
    MambaVC,
    MLICPlusPlus,
    MeanScaleHyperprior,
    SAAF,
    ScaleHyperprior,
    TCM,
    WeConvene,
)
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_factorized_relu,
    bmshj2018_hyperprior,
    cca,
    cheng2020_anchor,
    cheng2020_attn,
    candidate_model_architectures,
    cmic,
    dcae,
    ftic,
    hpcm,
    hpcm_base,
    hpcm_large,
    hpcm_phi,
    invcompress,
    mambaic,
    mambavc,
    mlicpp,
    mbt2018,
    mbt2018_mean,
    saaf,
    tcm,
    weconvene,
)
from compressai.zoo.image import _load_model


class TestLoadModel:
    def test_invalid(self):
        with pytest.raises(ValueError):
            _load_model("yolo", "mse", 1)

        with pytest.raises(ValueError):
            _load_model("mbt2018", "mse", 0)


class TestCandidateModels:
    def test_cmic_missing_dependency(self):
        if is_pytorch_wavelets_available():
            pytest.skip("pytorch_wavelets is installed.")

        assert candidate_model_architectures["cmic"] is None
        with pytest.raises(ModuleNotFoundError, match="pytorch_wavelets"):
            cmic()

    @pytest.mark.skipif(
        not is_pytorch_wavelets_available(),
        reason="pytorch_wavelets is not installed",
    )
    def test_cmic(self):
        net = cmic(
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

        assert isinstance(net, CMIC)
        assert candidate_model_architectures["cmic"] is CMIC

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            cmic(pretrained=True)

    def test_mlicpp(self):
        net = mlicpp(N=16, M=32, slice_num=4, context_window=3)

        assert isinstance(net, MLICPlusPlus)
        assert candidate_model_architectures["mlicpp"] is MLICPlusPlus

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            mlicpp(pretrained=True)

    def test_tcm(self):
        net = tcm(
            config=[1, 1, 1, 1, 1, 1],
            head_dim=[8, 8, 8, 8, 8, 8],
            N=16,
            M=32,
            hyper_channels=16,
            hyper_head_dim=8,
            num_slices=4,
            max_support_slices=2,
        )

        assert isinstance(net, TCM)
        assert candidate_model_architectures["lic-tcm"] is TCM
        assert candidate_model_architectures["tcm"] is TCM

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            tcm(pretrained=True)

    @pytest.mark.skipif(
        not is_pytorch_wavelets_available(),
        reason="pytorch_wavelets is not installed",
    )
    def test_tcm_with_auxt(self):
        net = tcm(
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

        assert isinstance(net, TCM)
        assert net.use_auxt
        assert candidate_model_architectures["lic-tcm"] is TCM

    def test_weconvene_missing_dependency(self):
        if is_pytorch_wavelets_available():
            pytest.skip("pytorch_wavelets is installed.")

        assert candidate_model_architectures["weconvene"] is None
        with pytest.raises(ModuleNotFoundError, match="pytorch_wavelets"):
            weconvene()

    @pytest.mark.skipif(
        not is_pytorch_wavelets_available(),
        reason="pytorch_wavelets is not installed",
    )
    def test_weconvene(self):
        net = weconvene(
            N=16,
            M=32,
            hyper_channels=16,
            num_slices=4,
            max_support_slices=2,
            support_window_size=4,
            support_head_dim=4,
            support_attention_dim=16,
        )

        assert isinstance(net, WeConvene)
        assert candidate_model_architectures["weconvene"] is WeConvene

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            weconvene(pretrained=True)

    def test_dcae(self):
        net = dcae(
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

        assert isinstance(net, DCAE)
        assert candidate_model_architectures["dcae"] is DCAE

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            dcae(pretrained=True)

    def test_ftic(self):
        net = ftic(
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

        assert isinstance(net, FrequencyAwareTransFormer)
        assert candidate_model_architectures["ftic"] is FrequencyAwareTransFormer

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            ftic(pretrained=True)

    def test_saaf(self):
        net = saaf(
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

        assert isinstance(net, SAAF)
        assert candidate_model_architectures["saaf"] is SAAF

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            saaf(pretrained=True)

    def test_cca(self):
        net = cca(
            latent_channels=32,
            hyper_channels=16,
            slice_proportions=(1, 1, 1, 1),
            encoder_dims=(16, 16, 24),
            encoder_layers=(1, 1, 1),
            em_hidden_channels=24,
            em_num_layers=1,
            cca_training=True,
        )

        assert isinstance(net, CCAModel)
        assert candidate_model_architectures["cca"] is CCAModel

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            cca(pretrained=True)

    def test_invcompress_missing_dependency(self):
        if is_freia_available():
            pytest.skip("FrEIA is installed.")

        assert candidate_model_architectures["invcompress"] is None
        with pytest.raises(ModuleNotFoundError, match="FrEIA"):
            invcompress()

    @pytest.mark.skipif(not is_freia_available(), reason="FrEIA is not installed")
    def test_invcompress(self):
        net = invcompress(N=192)

        assert isinstance(net, InvCompress)
        assert candidate_model_architectures["invcompress"] is InvCompress

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            invcompress(pretrained=True)

    def test_mambaic(self):
        net = mambaic(
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

        assert isinstance(net, MambaIC)
        assert candidate_model_architectures["mambaic"] is MambaIC

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            mambaic(pretrained=True)

    def test_mambavc(self):
        net = mambavc(
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

        assert isinstance(net, MambaVC)
        assert candidate_model_architectures["mambavc"] is MambaVC

        with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
            mambavc(pretrained=True)

    def test_hpcm(self):
        common_kwargs = dict(
            N=32,
            M=32,
            g_a_depth=1,
            g_s_depth=1,
            y_prior_depth=1,
            attn_window_s1=2,
            attn_window_s2=4,
            attn_window_s3=8,
            attn_num_heads=4,
        )
        for factory, expect_attention in (
            (hpcm, True),
            (hpcm_base, True),
            (hpcm_phi, False),
        ):
            kwargs = dict(common_kwargs)
            if not expect_attention:
                # PhiContext factory ignores attn_window_* kwargs because
                # use_attention=False; pass only the shape kwargs.
                for key in ("attn_window_s1", "attn_window_s2", "attn_window_s3", "attn_num_heads"):
                    kwargs.pop(key)
            net = factory(**kwargs)
            assert isinstance(net, HPCM)
            assert net.use_attention is expect_attention
            with pytest.raises(RuntimeError, match="Pre-trained model not yet available"):
                factory(pretrained=True)

        assert candidate_model_architectures["hpcm"] is HPCM
        assert candidate_model_architectures["hpcm-base"] is HPCM
        assert candidate_model_architectures["hpcm-large"] is HPCM
        assert candidate_model_architectures["hpcm-phi"] is HPCM


class TestBmshj2018Factorized:
    def test_params(self):
        for i in range(1, 6):
            net = bmshj2018_factorized(i, metric="mse", progress=False)
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(i, metric="mse", progress=False)
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            bmshj2018_factorized(-1)

        with pytest.raises(ValueError):
            bmshj2018_factorized(10)

        with pytest.raises(ValueError):
            bmshj2018_factorized(10, metric="ssim")

        with pytest.raises(ValueError):
            bmshj2018_factorized(1, metric="ssim")

    @pytest.mark.slow
    @pytest.mark.pretrained
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, metric):
        for i in range(1, 6):
            net = bmshj2018_factorized(
                i, metric=metric, pretrained=True, progress=False
            )
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized(
                i, metric=metric, pretrained=True, progress=False
            )
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestBmshj2018FactorizedReLU:
    def test_params(self):
        for i in range(1, 6):
            net = bmshj2018_factorized_relu(i, metric="mse", progress=False)
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_factorized_relu(i, metric="mse")
            assert isinstance(net, FactorizedPrior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            bmshj2018_factorized_relu(-1)

        with pytest.raises(ValueError):
            bmshj2018_factorized_relu(10)

        with pytest.raises(ValueError):
            bmshj2018_factorized_relu(10, metric="ssim")

        with pytest.raises(ValueError):
            bmshj2018_factorized_relu(1, metric="ssim")


class TestBmshj2018Hyperprior:
    def test_params(self):
        for i in range(1, 6):
            net = bmshj2018_hyperprior(i, metric="mse", progress=False)
            assert isinstance(net, ScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_hyperprior(i, metric="mse", progress=False)
            assert isinstance(net, ScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            bmshj2018_hyperprior(-1)

        with pytest.raises(ValueError):
            bmshj2018_hyperprior(10)

        with pytest.raises(ValueError):
            bmshj2018_hyperprior(10, metric="ssim")

        with pytest.raises(ValueError):
            bmshj2018_hyperprior(1, metric="ssim")

    @pytest.mark.slow
    @pytest.mark.pretrained
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, metric):
        # test we can load the correct models from the urls
        for i in range(1, 6):
            net = bmshj2018_hyperprior(
                i, metric=metric, pretrained=True, progress=False
            )
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(6, 9):
            net = bmshj2018_hyperprior(
                i, metric=metric, pretrained=True, progress=False
            )
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestMbt2018Mean:
    def test_parameters(self):
        for i in range(1, 5):
            net = mbt2018_mean(i, metric="mse", progress=False)
            assert isinstance(net, MeanScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018_mean(i, metric="mse", progress=False)
            assert isinstance(net, MeanScaleHyperprior)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            mbt2018_mean(-1)

        with pytest.raises(ValueError):
            mbt2018_mean(10)

        with pytest.raises(ValueError):
            mbt2018_mean(10, metric="ssim")

        with pytest.raises(ValueError):
            mbt2018_mean(1, metric="ssim")

    @pytest.mark.slow
    @pytest.mark.pretrained
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, metric):
        # test we can load the correct models from the urls
        for i in range(1, 5):
            net = mbt2018_mean(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.weight"].size(0) == 128
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018_mean(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestMbt2018:
    def test_ok(self):
        for i in range(1, 5):
            net = mbt2018(i, metric="mse", progress=False)
            assert isinstance(net, JointAutoregressiveHierarchicalPriors)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018(i, metric="mse", progress=False)
            assert isinstance(net, JointAutoregressiveHierarchicalPriors)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            mbt2018(-1)

        with pytest.raises(ValueError):
            mbt2018(10)

        with pytest.raises(ValueError):
            mbt2018(10, metric="ssim")

        with pytest.raises(ValueError):
            mbt2018(1, metric="ssim")

    @pytest.mark.slow
    @pytest.mark.pretrained
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, metric):
        # test we can load the correct models from the urls
        for i in range(1, 5):
            net = mbt2018(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 192

        for i in range(5, 9):
            net = mbt2018(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.weight"].size(0) == 192
            assert net.state_dict()["g_a.6.weight"].size(0) == 320


class TestCheng2020:
    @pytest.mark.parametrize(
        "func,cls",
        (
            (cheng2020_anchor, Cheng2020Anchor),
            (cheng2020_attn, Cheng2020Attention),
        ),
    )
    def test_anchor_ok(self, func, cls):
        for i in range(1, 4):
            net = func(i, metric="mse", progress=False)
            assert isinstance(net, cls)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 128

        for i in range(4, 7):
            net = func(i, metric="mse", progress=False)
            assert isinstance(net, cls)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 192

    @pytest.mark.slow
    @pytest.mark.pretrained
    @pytest.mark.parametrize("model_entrypoint", (cheng2020_anchor, cheng2020_attn))
    @pytest.mark.parametrize("metric", ("mse", "ms-ssim"))
    def test_pretrained(self, model_entrypoint, metric):
        for i in range(1, 4):
            net = model_entrypoint(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) == 128

        for i in range(4, 7):
            net = model_entrypoint(i, metric=metric, pretrained=True, progress=False)
            assert net.state_dict()["g_a.0.conv1.weight"].size(0) in (128, 192)
