from __future__ import annotations

"""Minimal base classes for conditional flows (positions-only).

We only provide a lightweight `SequentialFlow` that chains coupling layers and
accumulates log-determinant terms.  Velocity handling, caches, and advanced
features from Timewarp are stripped for clarity.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class FlowModule(nn.Module):
    """Interface for coupling layers compatible with `SequentialFlow`."""

    def forward(
        self,
        coords: Tensor,  # [B, N, 3]
        atom_types: Tensor,  # [B, N]
        adj_list: Tensor,  # [E, 2]
        reverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Return (new_coords, log_det).
        log_det has shape [B].
        """
        raise NotImplementedError


class SequentialFlow(nn.Module):
    """Chain of coupling layers supporting forward/reverse mode and log-det sums."""

    def __init__(self, layers: List[FlowModule]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        coords: Tensor,  # [B, N, 3]
        atom_types: Tensor,
        adj_list: Tensor,
        reverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        log_det = torch.zeros(coords.shape[0], device=coords.device)
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers:
            coords, ld = layer(coords=coords, atom_types=atom_types, adj_list=adj_list, reverse=reverse)
            log_det = log_det + ld
        return coords, log_det 