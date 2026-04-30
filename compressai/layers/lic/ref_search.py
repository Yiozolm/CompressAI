# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
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

"""Global reference search + masked-conv-on-unfolded-refs blocks.

Used by the Reference-Based AR image-compression model
(`Qian et al., ICLR 2021 <https://arxiv.org/abs/2010.08321>`_).

* :class:`SearchTransfer` — for each spatial location of ``y`` finds the
  most-similar already-decoded location (cosine similarity over a masked k×k
  patch) and gathers both its features and an associated probability.
* :class:`Conv2dUnfold` — applies a masked k×k convolution on the gathered
  *unfolded* references (returned by ``SearchTransfer``); equivalent to
  unfold → matmul → fold but operating directly on the gathered patches.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

__all__ = ["SearchTransfer", "Conv2dUnfold"]


class Conv2dUnfold(nn.Conv2d):
    """Masked k×k convolution acting on already-unfolded references.

    The trainable ``weight`` and the ``mask`` buffer share the same shape as a
    standard :class:`nn.Conv2d`. The ``mask`` zeroes out the contribution of
    the centre and below-centre kernel positions (mask type "A"), so the
    masked conv only "sees" causal neighbours of each reference patch.
    """

    def __init__(self, mask: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Buffer name 'mask' matches upstream so checkpoint keys load 1:1.
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.is_mask = bool(mask)
        if mask:
            self.mask.fill_(1)
            self.mask[:, :, kH // 2, kW // 2 + 1 :] = 0
            self.mask[:, :, kH // 2 + 1 :] = 0

    def forward(self, x_unfold: Tensor, h: int, w: int) -> Tensor:  # type: ignore[override]
        if self.is_mask:
            self.weight.data *= self.mask
        out_unfold = (
            x_unfold.transpose(1, 2)
            .matmul(self.weight.view(self.weight.size(0), -1).t())
            .transpose(1, 2)
        )
        return F.fold(out_unfold, (h, w), (1, 1))

    def forward_origin(self, x: Tensor) -> Tensor:
        """Standard masked-conv forward (operates on dense feature maps)."""
        if self.is_mask:
            self.weight.data *= self.mask
        return super().forward(x)


class SearchTransfer(nn.Module):
    """Causal global-reference search + transfer.

    For each spatial position the module finds the index of the most-similar
    already-decoded location based on cosine similarity over masked k×k
    patches, then gathers the corresponding patch and a per-position
    probability tensor. Returns ``(S, U, ref_unfold, R_arg)`` where:

    * ``S`` — similarity score map ``(N, 1, H, W)``;
    * ``U`` — gathered probability ``(N, 1, H, W)``;
    * ``ref_unfold`` — gathered k×k patch unfolded ``(N, C * k * k, H * W)``,
      ready to be consumed by :class:`Conv2dUnfold`;
    * ``R_arg`` — argmax indices ``(N, H * W)``.
    """

    def __init__(self, channels: int, k: int = 3, split: int = 1) -> None:
        super().__init__()
        # Mask Type "A": zero out centre + lower-half so the search only
        # references causal neighbours of each reference patch.
        mask = torch.ones((channels // split, k, k))
        mask[:, k // 2, k // 2 :] = 0
        mask[:, k // 2 + 1 :, :] = 0
        mask_unfold = F.unfold(mask.unsqueeze(0), kernel_size=(k, k), padding=0)
        # Stored as a non-trainable Parameter (matches upstream key
        # `search.mask_unfold` so converted checkpoints load by name).
        self.mask_unfold = nn.Parameter(mask_unfold, requires_grad=False)
        self.k = int(k)
        self.split = int(split)

    def forward(
        self, y_hat: Tensor, y_prob: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        k = self.k
        n, c, h, w = y_hat.shape

        unfold = F.unfold(y_hat, kernel_size=(k, k), padding=k // 2) * self.mask_unfold
        unfold = F.normalize(unfold, dim=1)  # (N, C*k*k, H*W)
        unfold_T = unfold.permute(0, 2, 1)  # (N, H*W, C*k*k)
        R = torch.bmm(unfold_T, unfold)  # (N, H*W, H*W)

        # Training: bidirectional reference (drop diagonal). Eval: causal only.
        if self.training:
            R = torch.triu(R, diagonal=1) + torch.tril(R, diagonal=-1)
        else:
            R = torch.triu(R, diagonal=1)

        R_star, R_star_arg = torch.max(R, dim=1)  # (N, H*W)

        y_hat_unfold = F.unfold(y_hat, kernel_size=(k, k), padding=k // 2)
        ref_unfold = self._batch_index_select(y_hat_unfold, 2, R_star_arg)
        unfold_prob = F.unfold(y_prob, kernel_size=(1, 1), padding=0)
        U_unfold = self._batch_index_select(unfold_prob, 2, R_star_arg)

        S = R_star.view(n, 1, h, w)
        U = F.fold(U_unfold, output_size=(h, w), kernel_size=(1, 1), padding=0)

        # First pixel has no causal reference; tag as zero/identity.
        if not self.training:
            S[:, :, 0, 0] = 1e-8
            U[:, :, 0, 0] = 1e-8
            ref_unfold[:, :, 0] = 0.0
            R_star_arg[:, 0] = -1

        S = torch.clamp(S, min=1e-8, max=1.0)
        U = torch.clamp(U, min=1e-8, max=1.0)
        return S, U, ref_unfold, R_star_arg

    @staticmethod
    def _batch_index_select(input_: Tensor, dim: int, index: Tensor) -> Tensor:
        """Per-batch ``torch.gather`` along ``dim``."""
        views = [input_.size(0)] + [
            1 if i != dim else -1 for i in range(1, len(input_.size()))
        ]
        expanse = list(input_.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input_, dim, index)
