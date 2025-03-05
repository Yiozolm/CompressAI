import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, conv3x3, subpel_conv3x3, Win_noShift_Attention, ConvTransBlock, \
    ResidualBlockWithStride, ResidualBlockUpsample, SWAtten, PatchEmbed, PatchSplit, PatchMerging, BasicLayer
from compressai.models import MeanScaleHyperprior
from compressai.models.utils import conv, deconv
from compressai.ops import quantize_ste
from compressai.registry import register_model
from .base import (
    CompressionModel,
)

__all__ = [
    "Minnen2020",
    "Minnen2020LRP",
    "WACNN",
    "SymmetricalTransFormer",
    "TCM",
]


@register_model("Minnen2020-channelwise")
class Minnen2020(MeanScaleHyperprior):
    r"""
    Channel-wise Context Model proposed in David Minnen&Saurabh Singh: `"Channel-wise Autoregressive Entropy Models for Learned Image Compression" <https://arxiv.org/abs/2007.08739>`_, ICIP 2020.
    *WITHOUT* LRP(Latent Residual Prediction) module.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
        slice (int): Channel Condition(CC) slices number
    """

    def __init__(self, N=192, M=320, slice=10, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, (N + M) // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv((N + M) // 2, N, stride=2, kernel_size=5),
        )
        # 320 -> 256 -> 192

        self.h_s = nn.Sequential(
            deconv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(N, (M + N) // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv((M + N) // 2, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
        )
        # 192 -> 256 -> 320
        self.slice = slice
        self.slice_size = M // self.slice  # Channel size for one slice. Note that M % slice should be zero
        self.y_size_list = [(i + 1) * self.slice_size for i in range(self.slice - 1)]
        self.y_size_list.append(M)  # [32, 64, 96, 128, 160, 192, 224, 256, 288, 320] if M = 320 and slice = 10
        EP_inputs = [i * self.slice_size for i in range(
            self.slice)]  # Input channel size for entropy parameters layer. [0, 32, 64, 96, 128, 160, 192, 224, 256, 288] if M = 320 and slice = 10
        self.EPlist = nn.ModuleList([])
        for y_size in EP_inputs:
            EP = nn.Sequential(
                conv(y_size + M, M - (N // 2), stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(M - (N // 2), (M + N) // 4, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv((M + N) // 4, M * 2 // 10, stride=1, kernel_size=3),
            )
            self.EPlist.append(EP)
        # Variable->224, 224->128, 128->32

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)
        list_sliced_y = []  # Stores each slice of y
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:, (self.slice_size * i):(self.slice_size * (i + 1)), :, :])
        list_sliced_y.append(y[:, self.slice_size * (self.slice - 1):, :, :])
        y_hat_cumul = torch.Tensor().to(y.device)  # Cumulative y_hat. Stores already encoded y_hat slice
        scales_hat_list = []
        means_hat_list = []
        for i in range(self.slice):
            if i == 0:
                gaussian_params = self.EPlist[0](
                    hyper_params
                )
            else:
                gaussian_params = self.EPlist[i](
                    torch.cat([hyper_params, y_hat_cumul], dim=1)
                )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            scales_hat_list.append(scales_hat)
            means_hat_list.append(means_hat)
            y_hat_sliced = self.gaussian_conditional.quantize(
                list_sliced_y[i], "noise" if self.training else "dequantize"
            )
            y_hat_cumul = torch.cat([y_hat_cumul, y_hat_sliced], dim=1)

        scales_all = torch.cat(scales_hat_list, dim=1)
        means_all = torch.cat(means_hat_list, dim=1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means=means_all)
        x_hat = self.g_s(y_hat_cumul)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        encoder = BufferedRansEncoder()
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        indexes_list = []
        symbols_list = []
        y_strings = []
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)

        list_sliced_y = []
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:, (self.slice_size * i):self.slice_size * (i + 1), :, :])
        list_sliced_y.append(y[:, self.slice_size * (self.slice - 1):, :, :])
        y_hat = torch.Tensor().to(x.device)
        for i in range(self.slice):
            y_sliced = list_sliced_y[i]  # size[1, M/S * i, H', W']
            if i == 0:
                gaussian_params = self.EPlist[0](
                    hyper_params
                )
            else:
                gaussian_params = self.EPlist[i](
                    torch.cat([hyper_params, y_hat], dim=1)
                )
            # gaussian_params = gaussian_params.squeeze(3).squeeze(2) #size ([1,256])
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)

            y_hat_sliced = self.gaussian_conditional.quantize(y_sliced, "symbols", means_hat)
            symbols_list.extend(y_hat_sliced.reshape(-1).tolist())
            indexes_list.extend(indexes.reshape(-1).tolist())
            y_hat_sliced = y_hat_sliced + means_hat

            y_hat = torch.cat([y_hat, y_hat_sliced], dim=1)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        y_string = encoder.flush()
        y_strings.append(y_string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyper_params = self.h_s(z_hat)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])

        y_hat = torch.Tensor().to(z_hat.device)
        for i in range(self.slice):
            if i == 0:
                gaussian_params = self.EPlist[0](hyper_params)
            else:
                gaussian_params = self.EPlist[i](
                    torch.cat([hyper_params, y_hat], dim=1)
                )
            scales_sliced, means_sliced = gaussian_params.chunk(2, 1)
            indexes_sliced = self.gaussian_conditional.build_indexes(scales_sliced)
            y_sliced_hat = decoder.decode_stream(
                indexes_sliced.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            y_sliced_hat = torch.Tensor(y_sliced_hat).reshape(scales_sliced.shape).to(scales_sliced.device)
            y_sliced_hat += means_sliced
            y_hat = torch.cat([y_hat, y_sliced_hat], dim=1)

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


@register_model("Minnen2020-channelwise-lrp")
class Minnen2020LRP(Minnen2020):
    r"""
    Channel-wise Context Model proposed in David Minnen&Saurabh Singh, Channel-wise Autoregressive Entropy Models for Learned Image Compression. ICIP 2020.
    *WITH* LRP(Latent Residual Prediction) module.

    This model also separates scale parameter prediction network and mean parameter prediction network to represent the original paper.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
        slice (int): Channel Condition(CC) slices number
    """

    def __init__(self, N=192, M=320, slice=10, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.slice = slice
        self.slice_size = M // self.slice
        y_size_list = [self.slice_size for _ in range(self.slice - 1)]
        y_size_list.append(M - self.slice_size * (self.slice - 1))
        # [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
        y_inputs = [i * self.slice_size for i in range(self.slice)]
        # [0, 32, 64, 96, 128, 160, 192, 224, 256, 288]
        LRP_inputs = [(i + 1) * self.slice_size for i in range(self.slice)]
        # [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
        self.LRPlist = nn.ModuleList([])
        self.scaleEPlist = nn.ModuleList([])
        self.meanEPlist = nn.ModuleList([])
        for y_cumul_size in y_inputs:
            scaleEP = nn.Sequential(
                conv(y_cumul_size + (M // 2), M - (N // 2), stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(M - (N // 2), (M + N) // 4, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv((M + N) // 4, M // self.slice, stride=1, kernel_size=3),
            )
            self.scaleEPlist.append(scaleEP)
            meanEP = nn.Sequential(
                conv(y_cumul_size + (M // 2), M - (N // 2), stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(M - (N // 2), (M + N) // 4, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv((M + N) // 4, M // self.slice, stride=1, kernel_size=3),
            )
            self.meanEPlist.append(meanEP)
        for y_cumul_size_alt in LRP_inputs:
            LRP = nn.Sequential(
                conv(y_cumul_size_alt + (M // 2), M - (N // 2), stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(M - (N // 2), (M + N) // 4, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv((M + N) // 4, M // self.slice, stride=1, kernel_size=3),
            )
            self.LRPlist.append(LRP)
            # The in/out channel for LRP layers are designed to agree the values shown in paper when N = 192 and M = 320.

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)
        hyper_scale, hyper_mean = hyper_params.chunk(2, 1)
        list_sliced_y = []
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:, (self.slice_size * i):self.slice_size * (i + 1), :, :])
        list_sliced_y.append(y[:, self.slice_size * (self.slice - 1):, :, :])
        y_hat = torch.Tensor().to(y.device)
        scales_hat_list = []
        means_hat_list = []
        for i in range(self.slice):
            if i == 0:
                scales_hat = self.scaleEPlist[0](
                    hyper_scale
                )
                means_hat = self.meanEPlist[0](
                    hyper_mean
                )
            else:
                scales_hat = self.scaleEPlist[i](
                    torch.cat([hyper_scale, y_hat], dim=1)
                )
                means_hat = self.meanEPlist[i](
                    torch.cat([hyper_mean, y_hat], dim=1)
                )
            scales_hat_list.append(scales_hat)
            means_hat_list.append(means_hat)
            y_hat_sliced = self.gaussian_conditional.quantize(
                list_sliced_y[i], "noise" if self.training else "dequantize"
            )
            LRP_param = self.LRPlist[i](
                torch.cat([y_hat, y_hat_sliced, hyper_mean], dim=1)
            )
            y_hat = torch.cat([y_hat, y_hat_sliced + LRP_param], dim=1)

        scales_all = torch.cat(scales_hat_list, dim=1)
        means_all = torch.cat(means_hat_list, dim=1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means=means_all)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        encoder = BufferedRansEncoder()
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        indexes_list = []
        symbols_list = []

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        hyper_scale, hyper_mean = hyper_params.chunk(2, 1)
        list_sliced_y = []
        for i in range(self.slice - 1):
            list_sliced_y.append(y[:, (self.slice_size * i):self.slice_size * (i + 1), :, :])
        list_sliced_y.append(y[:, self.slice_size * (self.slice - 1):, :, :])

        y_hat = torch.Tensor().to(y.device)
        for i in range(self.slice):
            y_sliced = list_sliced_y[i]  # size[1, M/S * i, H', W']
            if i == 0:
                scales_hat = self.scaleEPlist[0](
                    hyper_scale
                )
                means_hat = self.meanEPlist[0](
                    hyper_mean
                )
            else:
                scales_hat = self.scaleEPlist[i](
                    torch.cat([hyper_scale, y_hat], dim=1)
                )
                means_hat = self.meanEPlist[i](
                    torch.cat([hyper_mean, y_hat], dim=1)
                )
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_hat_sliced = self.gaussian_conditional.quantize(y_sliced, "symbols", means_hat)
            symbols_list.extend(y_hat_sliced.reshape(-1).tolist())
            indexes_list.extend(indexes.reshape(-1).tolist())
            y_hat_sliced = y_hat_sliced + means_hat

            # LRP configuration
            LRPparam = self.LRPlist[i](
                torch.cat([y_hat, y_hat_sliced, hyper_mean], dim=1)
            )
            y_hat_sliced += LRPparam
            y_hat = torch.cat([y_hat, y_hat_sliced], dim=1)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_strings = []
        y_string = encoder.flush()
        y_strings.append(y_string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyper_params = self.h_s(z_hat)
        hyper_scale, hyper_mean = hyper_params.chunk(2, 1)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])

        y_hat = torch.Tensor().to(z_hat.device)
        for i in range(self.slice):
            if i == 0:
                scales_sliced = self.scaleEPlist[0](
                    hyper_scale
                )
                means_sliced = self.meanEPlist[0](
                    hyper_mean
                )
            else:
                scales_sliced = self.scaleEPlist[i](
                    torch.cat([hyper_scale, y_hat], dim=1)
                )
                means_sliced = self.meanEPlist[i](
                    torch.cat([hyper_mean, y_hat], dim=1)
                )
            indexes_sliced = self.gaussian_conditional.build_indexes(scales_sliced)
            y_hat_sliced = decoder.decode_stream(
                indexes_sliced.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            y_hat_sliced = torch.Tensor(y_hat_sliced).reshape(scales_sliced.shape).to(scales_sliced.device)
            y_hat_sliced += means_sliced
            LRPparam = self.LRPlist[i](
                torch.cat([y_hat, y_hat_sliced, hyper_mean], dim=1)
            )

            y_hat_sliced += LRPparam
            y_hat = torch.cat([y_hat, y_hat_sliced], dim=1)

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


@register_model("Zou2022-CNN")
class WACNN(CompressionModel):
    r"""
    CNN Model
    Zou, Renjie, Chunfeng Song, and Zhaoxiang Zhang. “The Devil Is in the Details: Window-Based Attention for Image Compression.” In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17471–80. New Orleans, LA, USA: IEEE, 2022. https://doi.org/10.1109/CVPR52688.2022.01697.
    """
    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        self.num_slices = 10
        self.max_support_slices = 5

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i + 1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )

        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = quantize_ste(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}


@register_model("Zou2022-STF")
class SymmetricalTransFormer(CompressionModel):
    r"""
    Transformer Model
    Zou, Renjie, Chunfeng Song, and Zhaoxiang Zhang. “The Devil Is in the Details: Window-Based Attention for Image Compression.” In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17471–80. New Orleans, LA, USA: IEEE, 2022. https://doi.org/10.1109/CVPR52688.2022.01697.
    """
    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=2,
                 in_chans=3,
                 embed_dim=48,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=4,
                 num_slices=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.num_slices = num_slices
        self.max_support_slices = num_slices // 2
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=False)
            self.layers.append(layer)

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.syn_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (3 - i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchSplit if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=True)
            self.syn_layers.append(layer)

        self.end_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * patch_size ** 2, kernel_size=5, stride=1, padding=2),
            nn.PixelShuffle(patch_size),
            nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
        )

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.g_a = None
        self.g_s = None

        self.h_a = nn.Sequential(
            conv3x3(384, 384),
            nn.GELU(),
            conv3x3(384, 336),
            nn.GELU(),
            conv3x3(336, 288, stride=2),
            nn.GELU(),
            conv3x3(288, 240),
            nn.GELU(),
            conv3x3(240, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
        )
        self.h_scale_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(384 + 32 * min(i, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(384 + 32 * min(i, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(num_slices)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(384 + 32 * min(i + 1, 7), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(num_slices)
        )
        self.entropy_bottleneck = EntropyBottleneck(embed_dim * 4)
        self.gaussian_conditional = GaussianConditional(None)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x, Wh, Ww = layer(x, Wh, Ww)

        y = x
        C = self.embed_dim * 8
        y = y.view(-1, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = quantize_ste(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, Wh * Ww, C)
        for i in range(self.num_layers):
            layer = self.syn_layers[i]
            y_hat, Wh, Ww = layer(y_hat, Wh, Ww)

        x_hat = self.end_conv(y_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous())
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x, Wh, Ww = layer(x, Wh, Ww)
        y = x
        C = self.embed_dim * 8
        y = y.view(-1, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        Wh, Ww = y_shape
        C = self.embed_dim * 8

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, Wh * Ww, C)
        for i in range(self.num_layers):
            layer = self.syn_layers[i]
            y_hat, Wh, Ww = layer(y_hat, Wh, Ww)

        x_hat = self.end_conv(y_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous()).clamp_(0, 1)
        return {"x_hat": x_hat}


@register_model("Liu2022-TCM")
class TCM(CompressionModel):
    r"""
    Liu, Jinming, Heming Sun, and Jiro Katto. “Learned Image Compression with Mixed Transformer-CNN Architectures.” In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14388–97. Vancouver, BC, Canada: IEEE, 2023. https://doi.org/10.1109/CVPR52729.2023.01383.
    """

    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=128, M=320,
                 num_slices=5, max_support_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW')
                        for i in range(config[0])] + \
                       [ResidualBlockWithStride(2 * N, 2 * N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW')
                        for i in range(config[1])] + \
                       [ResidualBlockWithStride(2 * N, 2 * N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW')
                        for i in range(config[2])] + \
                       [conv3x3(2 * N, M, stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW')
                      for i in range(config[3])] + \
                     [ResidualBlockUpsample(2 * N, 2 * N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW')
                      for i in range(config[4])] + \
                     [ResidualBlockUpsample(2 * N, 2 * N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW')
                      for i in range(config[5])] + \
                     [subpel_conv3x3(2 * N, 3, 2)]

        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, 2 * N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)

        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, 2 * N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.ha_down1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i % 2 else 'SW')
                         for i in range(config[0])] + \
                        [conv3x3(2 * N, 192, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(320, 2 * N, 2)] + \
             self.ha_down1
        )

        self.hs_up1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i % 2 else 'SW')
                       for i in range(config[3])] + \
                      [subpel_conv3x3(2 * N, 320, 2)]

        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2 * N, 2)] + \
             self.hs_up1
        )

        self.hs_up2 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i % 2 else 'SW')
                       for i in range(config[3])] + \
                      [subpel_conv3x3(2 * N, 320, 2)]

        self.h_scale_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2 * N, 2)] + \
             self.hs_up2
        )

        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320 // self.num_slices) * min(i, 5)), (320 + (320 // self.num_slices) * min(i, 5)), 16,
                        self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320 // self.num_slices) * min(i, 5)), (320 + (320 // self.num_slices) * min(i, 5)), 16,
                        self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320 // self.num_slices) * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320 // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320 // self.num_slices) * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320 // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320 // self.num_slices) * min(i + 1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320 // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = quantize_ste(y_slice - mu) + mu
            # if self.training:
            #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
            # else:
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y}
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

