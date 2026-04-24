import math

import pytest
import torch

from compressai.losses import CCARateDistortionLoss


class TestCCARateDistortionLoss:
    def test_forward(self):
        criterion = CCARateDistortionLoss(
            lmbda=0.01,
            metric="mse",
            alpha=0.5,
            beta=1.25,
        )
        target = torch.rand(1, 3, 8, 8)
        x_hat = target.clone()
        output = {
            "x_hat": x_hat,
            "likelihoods": {
                "y": torch.full((1, 4, 2, 2), 0.75),
                "z": torch.full((1, 2, 1, 1), 0.8),
            },
            "aux_likelihoods": {
                "y_aux": torch.full((1, 4, 2, 2), 0.6),
                "y_cca": torch.full((1, 4, 2, 2), 0.5),
            },
        }

        out = criterion(output, target)

        num_pixels = target.size(0) * target.size(2) * target.size(3)
        expected_bpp = sum(
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in output["likelihoods"].values()
        )
        expected_cca = (
            torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2))
            - torch.log(output["aux_likelihoods"]["y_cca"]).sum() / (-math.log(2))
        ) / num_pixels
        expected_aux2 = torch.sum(
            output["aux_likelihoods"]["y_cca"]
            * torch.log(output["aux_likelihoods"]["y_aux"])
        ) / (-math.log(2) * num_pixels)

        assert torch.isclose(out["mse_loss"], torch.tensor(0.0))
        assert torch.isclose(out["bpp_loss"], expected_bpp)
        assert torch.isclose(out["cca_loss"], expected_cca)
        assert torch.isclose(out["aux2_loss"], expected_aux2)

    def test_missing_aux_likelihoods(self):
        criterion = CCARateDistortionLoss()
        target = torch.rand(1, 3, 8, 8)
        output = {
            "x_hat": target,
            "likelihoods": {"y": torch.full((1, 4, 2, 2), 0.75)},
        }

        with pytest.raises(KeyError):
            criterion(output, target)
