from __future__ import annotations

"""Position-only RealNVP coupling layers for molecular coordinates.

Each layer transforms a subset of coordinates conditioned on the remaining
subset.  A simple MLP produces scale/shift factors.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .mlp import MLP
from .base_flow import FlowModule

__all__ = ["PositionCouplingLayer", "create_alternating_masks"]


def create_alternating_masks(num_atoms: int) -> list[Tensor]:
    """Return a list of boolean masks of shape [dim] alternating per layer."""
    masks = []
    dim = num_atoms * 3
    base_mask = torch.tensor([((i // 3) % 2) for i in range(dim)], dtype=torch.bool)
    for i in range(8):
        masks.append(base_mask.clone())
        base_mask = ~base_mask  # flip
    return masks


class PositionCouplingLayer(FlowModule):
    """RealNVP-like coupling layer acting on Cartesian coords.

    Notes
    -----
    *We ignore graph adjacency and atom types for now.*  This can be upgraded to
    graph-aware or attention-based networks later.
    """

    def __init__(
        self,
        mask: Tensor,  # bool tensor of shape [dim]
        hidden_dim: int = 256,
        num_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)  # [dim]
        dim = mask.numel()
        in_dim = (~mask).sum().item()  # conditioning part size
        out_dim = mask.sum().item() * 2  # scale + shift for transformed part
        self.net = MLP(in_dim, out_dim, hidden_layer_dims=[hidden_dim] * num_hidden)

    def forward(
        self,
        coords: Tensor,  # [B, N, 3]
        atom_types: Tensor,
        adj_list: Tensor,
        reverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        B, N, _ = coords.shape
        x = coords.reshape(B, -1)  # [B, dim]
        x_a = x[:, ~self.mask]  # conditioning
        params = self.net(x_a)  # [B, out_dim]
        s, t = params.chunk(2, dim=-1)
        # Match Timewarp behaviour: small initial log-scale (~0) so flow starts near identity
        s = torch.tanh(s) * 0.05  # scale factor shrinks range to ±0.05
        log_scale = s  # log s
        scale = torch.exp(log_scale)

        y = x.clone()
        if reverse:
            # inverse: (y - t)/scale
            y_b = (y[:, self.mask] - t) / scale
            y[:, self.mask] = y_b
            log_det = -log_scale.sum(dim=-1)
        else:
            y_b = y[:, self.mask] * scale + t
            y[:, self.mask] = y_b
            log_det = log_scale.sum(dim=-1)

        y_coords = y.reshape(B, N, 3)
        return y_coords, log_det 