from __future__ import annotations

import math

import torch
import torch.nn as nn

from pytorch_msssim import ms_ssim

from compressai.registry import register_criterion


@register_criterion("CCARateDistortionLoss")
class CCARateDistortionLoss(nn.Module):
    def __init__(
        self,
        lmbda: float = 0.01,
        metric: str = "mse",
        return_type: str = "all",
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")

        self.lmbda = float(lmbda)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.return_type = return_type

    def forward(self, output, target):
        if "aux_likelihoods" not in output or output["aux_likelihoods"] is None:
            raise KeyError("output must contain aux_likelihoods for CCARateDistortionLoss")

        aux_likelihoods = output["aux_likelihoods"]
        if "y_aux" not in aux_likelihoods or "y_cca" not in aux_likelihoods:
            raise KeyError("aux_likelihoods must contain y_aux and y_cca")

        batch_size, _, height, width = target.size()
        num_pixels = batch_size * height * width
        out = {}

        out["cca_loss"] = (
            torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2))
            - torch.log(aux_likelihoods["y_cca"]).sum() / (-math.log(2))
        ) / num_pixels
        out["aux2_loss"] = torch.sum(
            aux_likelihoods["y_cca"] * torch.log(aux_likelihoods["y_aux"])
        ) / (-math.log(2) * num_pixels)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = (
            self.lmbda * distortion
            + self.beta * out["bpp_loss"]
            + self.alpha * out["cca_loss"]
            + out["aux2_loss"]
        )
        if self.return_type == "all":
            return out
        return out[self.return_type]
