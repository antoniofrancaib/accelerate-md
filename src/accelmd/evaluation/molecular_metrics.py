from __future__ import annotations

"""Basic molecular validity metrics used during training/evaluation."""

from typing import List, Tuple

import torch
from torch import Tensor


def bond_length_violations(coords: Tensor, bonds: List[Tuple[int, int]], tol: float = 0.1) -> Tensor:
    """Return fraction of bonds with length outside 0.1 nm ± tol.

    Parameters
    ----------
    coords
        Tensor of shape `[B, N, 3]` (nm).
    bonds
        List of (i,j) atom index pairs.
    tol
        Allowed deviation from 0.1 nm (~1 Å).
    """
    B = coords.shape[0]
    device = coords.device
    mask = torch.zeros(B, dtype=torch.bool, device=device)
    for i, j in bonds:
        dist = (coords[:, i] - coords[:, j]).norm(dim=-1)
        mask |= (dist < 0.1 - tol) | (dist > 0.1 + tol)
    return mask.float().mean() 