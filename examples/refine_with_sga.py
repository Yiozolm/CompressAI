"""Refine a single image's latent ``y`` and side information ``z`` with SGA.

Loads an MLIC-family checkpoint and runs the Stochastic Gumbel Annealing
inference-time optimization from Yang et al. (NeurIPS 2020) on the latent
representation, reproducing the MLICv2+ recipe described in
arXiv:2504.19119 §3.5. Reports BPP / PSNR before and after refinement.

Notes:
    * Works out of the box for ``mlic`` / ``mlic+`` / ``mlicpp`` (pure SGA).
    * For ``mlicv2``, GSC stays active during refinement; meaningful gradient
      on ``y`` requires the GSC head to have been trained (i.e. a real ckpt,
      not fresh init).

Example::

    python examples/refine_with_sga.py \\
        --variant mlicpp \\
        --checkpoint candidate/MLIC/mlicpp_mse_q5_2960000.pth.tar \\
        --image kodak/kodim01.png \\
        --quality-lambda 0.025 \\
        --total-iter 2000 \\
        --lr 5e-3
"""

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

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Dict, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from compressai.models.base import CompressionModel
from compressai.models.mlic import MLIC, MLICPlus, MLICPlusPlus, MLICv2
from compressai.ops import SGAQuantizer

_MODELS: Dict[str, Type[CompressionModel]] = {
    "mlic": MLIC,
    "mlic+": MLICPlus,
    "mlicpp": MLICPlusPlus,
    "mlicv2": MLICv2,
}


def _psnr(mse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(torch.tensor(1.0) / mse)


def _bpp(out: Dict, num_pixels: int) -> torch.Tensor:
    y_lik = out["likelihoods"]["y"]
    z_lik = out["likelihoods"]["z"]
    return (-torch.log2(y_lik).sum() - torch.log2(z_lik).sum()) / num_pixels


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return x


def _load_model(
    variant: str, checkpoint: Path, device: torch.device
) -> CompressionModel:
    cls = _MODELS[variant]
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = cls.from_state_dict(state).to(device).eval()
    return model


def _eval_pass(
    model: CompressionModel,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> Tuple[float, float]:
    with torch.no_grad():
        out = model.refine_forward(y, z)
    bpp = _bpp(out, x.shape[2] * x.shape[3]).item()
    mse = F.mse_loss(x, out["x_hat"]).item()
    psnr = _psnr(torch.tensor(mse)).item()
    return bpp, psnr


def refine(
    model: CompressionModel,
    x: torch.Tensor,
    *,
    quality_lambda: float,
    total_iter: int,
    lr: float,
) -> Dict[str, float]:
    sga = SGAQuantizer()
    model.set_sga_mode(sga)

    with torch.no_grad():
        y_init, z_init = model.refine_extract(x)

    bpp_init, psnr_init = _eval_pass(model, x, y_init, z_init)

    y = nn.Parameter(y_init.clone())
    z = nn.Parameter(z_init.clone())
    opt = torch.optim.Adam([y, z], lr=lr)

    num_pixels = x.shape[2] * x.shape[3]
    for it in range(total_iter):
        sga.set_iter(it, total_iter)
        opt.zero_grad()
        out = model.refine_forward(y, z)
        bpp = _bpp(out, num_pixels)
        mse = F.mse_loss(x, out["x_hat"])
        loss = bpp + quality_lambda * mse * 255**2
        loss.backward()
        opt.step()

    sga.set_iter(None, None)
    bpp_post, psnr_post = _eval_pass(model, x, y, z)
    model.set_sga_mode(None)

    return {
        "bpp_init": bpp_init,
        "psnr_init": psnr_init,
        "bpp_post": bpp_post,
        "psnr_post": psnr_post,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(_MODELS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--quality-lambda", type=float, default=0.025)
    parser.add_argument("--total-iter", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = _load_model(args.variant, args.checkpoint, device)
    x = _load_image(args.image, device)

    result = refine(
        model,
        x,
        quality_lambda=args.quality_lambda,
        total_iter=args.total_iter,
        lr=args.lr,
    )
    print(
        "init  bpp={bpp_init:.4f}  psnr={psnr_init:.2f}\n"
        "post  bpp={bpp_post:.4f}  psnr={psnr_post:.2f}".format(**result)
    )


if __name__ == "__main__":
    main()
