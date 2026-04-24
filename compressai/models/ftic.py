from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GsnConditionalLocScaleShift
from compressai.layers.lic import TCAEntropyModel
from compressai.registry import register_model

from .base import CompressionModel
from .ftic_support import (
    FTICAnalysisTransform,
    FTICHyperAnalysisTransform,
    FTICHyperSynthesisTransform,
    FTICSynthesisTransform,
    infer_fm_window_size,
    infer_ftic_num_heads,
    infer_ftic_stage_depth,
    infer_num_slices,
    infer_tca_depth,
    infer_tca_ratio,
    infer_window_size,
)
from .utils import update_registered_buffers

__all__ = ["FrequencyAwareTransFormer"]


def ste_round(input_tensor: Tensor) -> Tensor:
    return torch.round(input_tensor) - input_tensor.detach() + input_tensor


def _split_drop_paths(
    drop_path_rate: float,
    config: Sequence[int],
) -> Tuple[Sequence[float], ...]:
    all_paths = torch.linspace(0.0, drop_path_rate, sum(config)).tolist()
    splits = []
    offset = 0
    for depth in config:
        splits.append(tuple(float(value) for value in all_paths[offset : offset + depth]))
        offset += depth
    return tuple(splits)


@register_model("ftic")
class FrequencyAwareTransFormer(CompressionModel):
    def __init__(
        self,
        config: Sequence[int] = (2, 2, 2, 2, 2, 2),
        num_heads: Sequence[int] = (8, 16, 32, 32, 16, 8),
        drop_path_rate: float = 0.0,
        feature_dims: Tuple[int, int, int] = (96, 144, 256),
        hyper_hidden_channels: int = 256,
        hyper_channels: int = 192,
        M: int = 320,
        num_slices: int = 5,
        num_scales: int = 256,
        num_means: int = 100,
        min_scale: float = 0.01,
        tail_mass: float = 2 ** (-8),
        window_size: int = 8,
        fm_window_size: int = 16,
        hyper_window_size: int = 2,
        hyper_fm_window_size: int = 4,
        hyper_num_heads: int = 32,
        tca_depth: int = 12,
        tca_ratio: int = 4,
        tca_window_size: int = 8,
        tca_num_heads: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if len(config) != 6:
            raise ValueError("config must provide six stage depths")
        if len(num_heads) != 6:
            raise ValueError("num_heads must provide six stage head counts")
        if M % num_slices != 0:
            raise ValueError("M must be divisible by num_slices")
        if M % tca_num_heads != 0:
            raise ValueError("M must be divisible by tca_num_heads")

        self.config = tuple(int(value) for value in config)
        self.num_heads = tuple(int(value) for value in num_heads)
        self.feature_dims = tuple(int(value) for value in feature_dims)
        self.hyper_hidden_channels = int(hyper_hidden_channels)
        self.hyper_channels = int(hyper_channels)
        self.M = int(M)
        self.num_slices = int(num_slices)
        self.num_scales = int(num_scales)
        self.num_means = int(num_means)
        self.min_scale = float(min_scale)
        self.tail_mass = float(tail_mass)
        self.window_size = int(window_size)
        self.fm_window_size = int(fm_window_size)
        self.hyper_window_size = int(hyper_window_size)
        self.hyper_fm_window_size = int(hyper_fm_window_size)
        self.hyper_num_heads = int(hyper_num_heads)
        self.tca_depth = int(tca_depth)
        self.tca_ratio = int(tca_ratio)
        self.tca_window_size = int(tca_window_size)
        self.tca_num_heads = int(tca_num_heads)

        drop_paths = _split_drop_paths(drop_path_rate, self.config)
        self.g_a = FTICAnalysisTransform(
            feature_dims=self.feature_dims,
            M=self.M,
            config=self.config,
            num_heads=self.num_heads,
            drop_paths=drop_paths,
            window_size=self.window_size,
            fm_window_size=self.fm_window_size,
        )
        self.g_s = FTICSynthesisTransform(
            feature_dims=self.feature_dims,
            M=self.M,
            config=self.config,
            num_heads=self.num_heads,
            drop_paths=tuple(reversed(drop_paths)),
            window_size=self.window_size,
            fm_window_size=self.fm_window_size,
        )
        self.h_a = FTICHyperAnalysisTransform(
            M=self.M,
            hyper_hidden_channels=self.hyper_hidden_channels,
            hyper_channels=self.hyper_channels,
            depth=self.config[0],
            num_heads=self.hyper_num_heads,
            window_size=self.hyper_window_size,
            fm_window_size=self.hyper_fm_window_size,
        )
        self.h_mean_s = FTICHyperSynthesisTransform(
            M=self.M,
            hyper_hidden_channels=self.hyper_hidden_channels,
            hyper_channels=self.hyper_channels,
            depth=self.config[3],
            num_heads=self.hyper_num_heads,
            window_size=self.hyper_window_size,
            fm_window_size=self.hyper_fm_window_size,
        )
        self.h_scale_s = FTICHyperSynthesisTransform(
            M=self.M,
            hyper_hidden_channels=self.hyper_hidden_channels,
            hyper_channels=self.hyper_channels,
            depth=self.config[3],
            num_heads=self.hyper_num_heads,
            window_size=self.hyper_window_size,
            fm_window_size=self.hyper_fm_window_size,
        )
        self.tca = TCAEntropyModel(
            dim=self.M,
            depth=self.tca_depth,
            ratio=self.tca_ratio,
            slices=self.num_slices,
            window_size=self.tca_window_size,
            num_heads=self.tca_num_heads,
        )
        self.entropy_bottleneck = EntropyBottleneck(self.hyper_channels)
        self.gaussian_conditional = GsnConditionalLocScaleShift(
            num_scales=self.num_scales,
            num_means=self.num_means,
            min_scale=self.min_scale,
            tail_mass=self.tail_mass,
        )

    def _hyper(self, z_hat: Tensor) -> Tensor:
        return torch.cat((self.h_mean_s(z_hat), self.h_scale_s(z_hat)), dim=1)

    def forward(self, x: Tensor) -> Dict[str, Dict[str, Tensor] | Tensor]:
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        hyper = self._hyper(z_hat)
        y_hat = ste_round(y)
        means, scales, lrp = self.tca(hyper, y_hat)
        _, y_likelihoods = self.gaussian_conditional(y, scales, means)
        y_hat = y_hat + 0.5 * torch.tanh(lrp)
        return {
            "x_hat": self.g_s(y_hat),
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: Tensor) -> Dict[str, object]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper = self._hyper(z_hat)

        y_strings = []
        y_hat_coded = torch.round(y)
        lrp_coded = torch.zeros_like(y)
        channels_per_slice = self.M // self.num_slices
        means, scales, lrps = self.tca(hyper, y_hat_coded)

        for slice_index in range(self.num_slices):
            start = slice_index * channels_per_slice
            end = (slice_index + 1) * channels_per_slice
            mu = means[:, start:end]
            scale = scales[:, start:end]
            lrp = lrps[:, start:end]
            y_slice = y[:, start:end]
            y_hat_slice, slice_strings = self.gaussian_conditional.compress(y_slice, scale, mu)
            y_hat_coded[:, start:end] = y_hat_slice
            lrp_coded[:, start:end] = lrp
            y_strings.append(slice_strings)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "y_hat": y_hat_coded + 0.5 * torch.tanh(lrp_coded),
        }

    def decompress(
        self,
        strings: Sequence[Sequence[bytes] | Sequence[Sequence[bytes]]],
        shape: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        if len(strings) != 2:
            raise ValueError("strings must contain [y_strings, z_strings]")

        y_strings = strings[0]
        z_strings = strings[1]
        if len(y_strings) != self.num_slices:
            raise ValueError("y_strings must contain one entry per slice")

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper = self._hyper(z_hat)
        y_hat_coded = torch.zeros(
            z_hat.size(0),
            self.M,
            z_hat.size(2) * 4,
            z_hat.size(3) * 4,
            device=z_hat.device,
        )
        lrp_coded = torch.zeros_like(y_hat_coded)
        channels_per_slice = self.M // self.num_slices

        for slice_index in range(self.num_slices):
            means, scales, lrps = self.tca(hyper, y_hat_coded)
            start = slice_index * channels_per_slice
            end = (slice_index + 1) * channels_per_slice
            mu = means[:, start:end]
            scale = scales[:, start:end]
            lrp = lrps[:, start:end]
            y_hat_slice = self.gaussian_conditional.decompress(y_strings[slice_index], scale, mu)
            y_hat_coded[:, start:end] = y_hat_slice
            lrp_coded[:, start:end] = lrp

        y_hat = y_hat_coded + 0.5 * torch.tanh(lrp_coded)
        return {"x_hat": self.g_s(y_hat).clamp_(0, 1)}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "FrequencyAwareTransFormer":
        dim0 = state_dict["g_a.input_block.conv1.weight"].size(0)
        dim1 = state_dict["g_a.stage1.tail.conv1.weight"].size(0)
        dim2 = state_dict["g_a.stage2.tail.conv1.weight"].size(0)
        M = state_dict["g_a.stage3.tail.weight"].size(0)
        config = (
            infer_ftic_stage_depth(state_dict, "g_a.stage1.blocks."),
            infer_ftic_stage_depth(state_dict, "g_a.stage2.blocks."),
            infer_ftic_stage_depth(state_dict, "g_a.stage3.blocks."),
            infer_ftic_stage_depth(state_dict, "g_s.stage1.blocks."),
            infer_ftic_stage_depth(state_dict, "g_s.stage2.blocks."),
            infer_ftic_stage_depth(state_dict, "g_s.stage3.blocks."),
        )
        num_heads = (
            infer_ftic_num_heads(state_dict, "g_a.stage1"),
            infer_ftic_num_heads(state_dict, "g_a.stage2"),
            infer_ftic_num_heads(state_dict, "g_a.stage3"),
            infer_ftic_num_heads(state_dict, "g_s.stage1"),
            infer_ftic_num_heads(state_dict, "g_s.stage2"),
            infer_ftic_num_heads(state_dict, "g_s.stage3"),
        )
        net = cls(
            config=config,
            num_heads=num_heads,
            feature_dims=(dim0, dim1, dim2),
            hyper_hidden_channels=state_dict["h_a.input_block.conv1.weight"].size(0),
            hyper_channels=state_dict["entropy_bottleneck.quantiles"].size(0),
            M=M,
            num_slices=infer_num_slices(state_dict, M),
            num_scales=state_dict["gaussian_conditional.scale_table"].numel(),
            num_means=state_dict["gaussian_conditional._prior_mean"].size(0)
            if state_dict["gaussian_conditional._prior_mean"].numel() > 0
            else 100,
            min_scale=float(state_dict["gaussian_conditional.scale_bound"].item()),
            window_size=infer_window_size(state_dict, "g_a.stage1"),
            fm_window_size=infer_fm_window_size(state_dict, "g_a.stage1"),
            hyper_window_size=infer_window_size(state_dict, "h_a.stage"),
            hyper_fm_window_size=infer_fm_window_size(state_dict, "h_a.stage"),
            hyper_num_heads=infer_ftic_num_heads(state_dict, "h_a.stage"),
            tca_depth=infer_tca_depth(state_dict),
            tca_ratio=infer_tca_ratio(state_dict, M),
        )
        if "gaussian_conditional._prior_mean" in state_dict:
            update_registered_buffers(
                net.gaussian_conditional,
                "gaussian_conditional",
                ["_prior_mean", "_prior_scale"],
                state_dict,
            )
        net.load_state_dict(state_dict)
        return net
