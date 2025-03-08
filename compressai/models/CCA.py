import math
import torch
import torch.nn as nn
from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste
from compressai.registry import register_model
from .base import (
    CompressionModel,
)

__all__ = [
    "CCA",
]


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv(in_ch, mid_ch, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv(mid_ch, mid_ch, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv(mid_ch, out_ch, kernel_size=1, stride=1)
        self.skip = conv(in_ch, out_ch, kernel_size=1, stride=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out + identity


class NAFBlock(nn.Module):
    def __init__(self, dim, inter_dim=None):
        super().__init__()

        self.dim = inter_dim if inter_dim is not None else dim

        dw_channel = self.dim << 1
        ffn_channel = self.dim << 1

        self.dwconv = nn.Sequential(
            nn.Conv2d(self.dim, dw_channel, 1),
            nn.Conv2d(dw_channel, dw_channel, 3, 1, padding=1, groups=dw_channel)
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.FFN = nn.Sequential(
            nn.Conv2d(self.dim, ffn_channel, 1),
            SimpleGate(),
            nn.Conv2d(ffn_channel >> 1, self.dim, 1)
        )

        self.norm1 = LayerNorm2d(self.dim)
        self.norm2 = LayerNorm2d(self.dim)
        self.conv1 = nn.Conv2d(dw_channel >> 1, self.dim, 1)

        self.beta = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)

        self.in_conv = conv(dim, inter_dim, kernel_size=1, stride=1) if inter_dim is not None else nn.Identity()
        self.out_conv = conv(inter_dim, dim, kernel_size=1, stride=1) if inter_dim is not None else nn.Identity()

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        x = self.norm1(x)

        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv1(x)

        out = identity + x * self.beta
        identity = out

        out = self.norm2(out)
        out = self.FFN(out)

        out = identity + out * self.gamma

        out = self.out_conv(out)
        return out


class blocks(nn.Module):
    def __init__(self, input_dim, output_dim, layers=4, inter_dim=128) -> None:
        super().__init__()

        self.layers = layers
        self.blocks = nn.ModuleList(NAFBlock(inter_dim) for _ in range(self.layers))

        self.in_conv = conv(input_dim, inter_dim, kernel_size=1, stride=1)
        self.out_conv = conv(inter_dim, output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.in_conv(x)
        identity = out
        for i in range(self.layers):
            out = self.blocks[i](out)
        out += identity
        out = self.out_conv(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_dim=3, bottleneck_dim=320, dim=[192, 224, 256], depth=3, layers=[4, 4, 4]):
        super().__init__()
        assert len(dim) == len(layers) == depth
        self.depth = depth
        self.dim = [in_dim] + dim + [bottleneck_dim]
        self.down = nn.ModuleList(
            conv(self.dim[i], self.dim[i + 1], kernel_size=5, stride=2) for i in range(depth + 1)
        )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                nn.Sequential(*([ResidualBottleneckBlock(dim[i], dim[i]) for _ in range(3)] +
                                [NAFBlock(dim[i]) for _ in range(layers[i])])
                              )
            )

    def forward(self, x):
        for i in range(self.depth):
            x = self.down[i](x)
            x = self.blocks[i](x)
        y = self.down[self.depth](x)
        return y


class Decoder(nn.Module):
    def __init__(self, out_dim=3, bottleneck_dim=320, dim=[192, 224, 256], depth=3, layers=[4, 4, 4]):
        super().__init__()
        assert len(dim) == len(layers) == depth
        self.depth = depth
        self.dim = [out_dim] + dim + [bottleneck_dim]
        self.up = nn.ModuleList(
            deconv(self.dim[i + 1], self.dim[i], kernel_size=5, stride=2) for i in reversed(range(depth + 1))
        )
        self.blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            self.blocks.append(
                nn.Sequential(*([NAFBlock(dim[i]) for _ in range(layers[i])] +
                                [ResidualBottleneckBlock(dim[i], dim[i]) for _ in range(3)])
                              )
            )

    def forward(self, x):
        for i in range(self.depth):
            x = self.up[i](x)
            x = self.blocks[i](x)
        y = self.up[self.depth](x)
        return y


class AuxEntropyModel(nn.Module):
    def __init__(
            self,
            M=320,
            prop_slices=[1, 1, 1, 1, 1],
            em_dim=224,
            em_layers=4,
    ):
        super().__init__()

        if len(prop_slices) == 1:
            self.num_slices = prop_slices[0]
            prop_slices = [1 for _ in range(self.num_slices)]
        else:
            self.num_slices = len(prop_slices)
        self.size_slices = list(math.floor(1.0 * M / sum(prop_slices) * prop) for prop in prop_slices)
        self.size_slices[self.num_slices - 1] += M - sum(self.size_slices)

        self.mean_NAF_transforms = nn.ModuleList(
            blocks(M + sum(self.size_slices[:(i - 1 if i - 1 > 0 else 0)]),
                   M + sum(self.size_slices[:(i - 1 if i - 1 > 0 else 0)]), layers=em_layers, inter_dim=em_dim) for i in
            range(self.num_slices)
        )

        self.scale_NAF_transforms = nn.ModuleList(
            blocks(M + sum(self.size_slices[:(i - 1 if i - 1 > 0 else 0)]),
                   M + sum(self.size_slices[:(i - 1 if i - 1 > 0 else 0)]), layers=em_layers, inter_dim=em_dim) for i in
            range(self.num_slices)
        )

        self.mean_cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + sum(self.size_slices[:(i - 1 if i - 1 > 0 else 0)]), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.size_slices[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.scale_cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + sum(self.size_slices[:(i - 1 if i - 1 > 0 else 0)]), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.size_slices[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        if self.num_slices > 2:
            self.lrp_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(M + self.size_slices[i] + sum(self.size_slices[:(i - 1 if i - 1 > 0 else 0)]), 224, stride=1,
                         kernel_size=3),
                    nn.GELU(),
                    conv(224, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, self.size_slices[i], stride=1, kernel_size=3),
                ) for i in range(self.num_slices - 2)
            )

        self.y_entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, y, latent_scales, latent_means):
        y_hat_slices = []
        y_likelihood = []

        _, y_aux_likelihoods = self.y_entropy_bottleneck(y)
        y_slices = y.split(self.size_slices, 1)

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = y_hat_slices[:(slice_index - 1)] if slice_index - 1 > 0 else []
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.mean_NAF_transforms[slice_index](mean_support)
            mu = self.mean_cc_transforms[slice_index](mean_support)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.scale_NAF_transforms[slice_index](scale_support)
            scale = self.scale_cc_transforms[slice_index](scale_support)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)

            if self.num_slices > 2 and slice_index < self.num_slices - 2:
                y_hat_slice = quantize_ste(y_slice - mu) + mu
                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp
                y_hat_slices.append(y_hat_slice)

        y_likelihoods = torch.cat(y_likelihood, dim=1)

        return y_aux_likelihoods, y_likelihoods


@register_model("Han2024-CCA")
class CCA(CompressionModel):
    r"""
    Han, Minghao, Shiyin Jiang, Shengxi Li, Xin Deng, Mai Xu, Ce Zhu和Shuhang Gu. "Causal Context Adjustment Loss for Learned Image Compression". in The Thirty-Eighth Annual Conference on Neural Information Processing Systems (NIPS). Vancouver CANADA, 2024.
    """

    def __init__(
            self,
            M=320,
            prop_slices=[1, 1, 1, 1, 1],
            ae_dim=[192, 224, 256],
            em_dim=224,
            ae_layers=[4, 4, 4],
            em_layers=4,
            cca_training=False
    ):
        super().__init__()

        if len(prop_slices) == 1:
            self.num_slices = prop_slices[0]
            prop_slices = [1 for _ in range(self.num_slices)]
        else:
            self.num_slices = len(prop_slices)
        self.size_slices = list(math.floor(1.0 * M / sum(prop_slices) * prop) for prop in prop_slices)
        self.size_slices[self.num_slices - 1] += M - sum(self.size_slices)
        self.cca_training = cca_training

        self.g_a = Encoder(3, M, ae_dim, 3, ae_layers)
        self.g_s = Decoder(3, M, ae_dim, 3, ae_layers)

        self.h_a = nn.Sequential(
            conv(M, ae_dim[2], stride=1, kernel_size=3),
            nn.GELU(),
            conv(ae_dim[2], ae_dim[2], stride=2, kernel_size=5),
            nn.GELU(),
            conv(ae_dim[2], 192, stride=2, kernel_size=5),
        )

        self.h_mean_s = nn.Sequential(
            deconv(192, ae_dim[2], stride=2, kernel_size=5),
            nn.GELU(),
            deconv(ae_dim[2], ae_dim[2], stride=2, kernel_size=5),
            nn.GELU(),
            deconv(ae_dim[2], M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(192, ae_dim[2], stride=2, kernel_size=5),
            nn.GELU(),
            deconv(ae_dim[2], ae_dim[2], stride=2, kernel_size=5),
            nn.GELU(),
            deconv(ae_dim[2], M, stride=1, kernel_size=3),
        )

        self.mean_NAF_transforms = nn.ModuleList(
            blocks(M + sum(self.size_slices[:i]), M + sum(self.size_slices[:i]), layers=em_layers, inter_dim=em_dim) for
            i in range(self.num_slices)
        )

        self.scale_NAF_transforms = nn.ModuleList(
            blocks(M + sum(self.size_slices[:i]), M + sum(self.size_slices[:i]), layers=em_layers, inter_dim=em_dim) for
            i in range(self.num_slices)
        )

        self.mean_cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + sum(self.size_slices[:i]), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.size_slices[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.scale_cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + sum(self.size_slices[:i]), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.size_slices[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + sum(self.size_slices[: i + 1]), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.size_slices[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        if cca_training:
            self.aux_entropymodel = AuxEntropyModel(M, prop_slices, em_dim, em_layers)

        self.z_entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.z_entropy_bottleneck(z)

        z_offset = self.z_entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.split(self.size_slices, 1)

        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            mean_support = torch.cat([latent_means] + y_hat_slices, dim=1)
            mean_support = self.mean_NAF_transforms[slice_index](mean_support)
            mu = self.mean_cc_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales] + y_hat_slices, dim=1)
            scale_support = self.scale_NAF_transforms[slice_index](scale_support)
            scale = self.scale_cc_transforms[slice_index](scale_support)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = quantize_ste(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        if self.cca_training:
            y_aux_likelihoods, y_cca_likelihoods = self.aux_entropymodel(y, latent_scales, latent_means)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "y": y,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "aux_likelihoods": {"y_aux": y_aux_likelihoods, "y_cca": y_cca_likelihoods} if self.cca_training else None,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)

        z = self.h_a(y)
        z_strings = self.z_entropy_bottleneck.compress(z)
        z_hat = self.z_entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.split(self.size_slices, 1)
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
            mean_support = torch.cat([latent_means] + y_hat_slices, dim=1)
            mean_support = self.mean_NAF_transforms[slice_index](mean_support)
            mu = self.mean_cc_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales] + y_hat_slices, dim=1)
            scale_support = self.scale_NAF_transforms[slice_index](scale_support)
            scale = self.scale_cc_transforms[slice_index](scale_support)

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
        z_hat = self.z_entropy_bottleneck.decompress(strings[1], shape)
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
            mean_support = torch.cat([latent_means] + y_hat_slices, dim=1)
            mean_support = self.mean_NAF_transforms[slice_index](mean_support)
            mu = self.mean_cc_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales] + y_hat_slices, dim=1)
            scale_support = self.scale_NAF_transforms[slice_index](scale_support)
            scale = self.scale_cc_transforms[slice_index](scale_support)

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
