from __future__ import annotations

"""Physics-informed flow mapping coordinates between adjacent PT temperatures.

For v0 this class provides deterministic transforms and log-determinant
calculations; the likelihood interface will be filled once the trainer logic is
in place.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..targets import build_target
from .base_flow import SequentialFlow
from .coupling_layers import PositionCouplingLayer, create_alternating_masks

__all__ = ["PTSwapFlow"]


class PTSwapFlow(nn.Module):
    """Deterministic invertible flow between two temperatures.

    Parameters
    ----------
    num_atoms
        Number of atoms (22 for ALDP → 66-D Cartesian space).
    num_layers
        Number of coupling layers (alternating masks).
    hidden_dim
        Hidden width of the MLP inside each coupling layer.
    source_temperature
        Lower temperature of the pair.
    target_temperature
        Higher temperature of the pair.
    target_name
        Name of the target Boltzmann base.
    target_kwargs
        Additional keyword arguments for building the Boltzmann bases.
    device
        Compute device for internal tensors.
    """

    def __init__(
        self,
        num_atoms: int = 22,
        num_layers: int = 8,
        hidden_dim: int = 256,
        source_temperature: float = 1.0,
        target_temperature: float = 1.5,
        target_name: str = "aldp",
        target_kwargs: dict | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.dim = num_atoms * 3
        self.device = torch.device(device)

        # Build coupling layers with alternating masks
        masks = create_alternating_masks(num_atoms)[:num_layers]
        layers: List[PositionCouplingLayer] = []
        for mask in masks:
            layers.append(
                PositionCouplingLayer(mask=mask, hidden_dim=hidden_dim)
            )
        self.flow = SequentialFlow(layers).to(self.device)

        # Boltzmann priors for each temperature (for likelihood later)
        target_kwargs = target_kwargs or {}
        self.base_low = build_target(target_name, temperature=source_temperature, device="cpu", **target_kwargs)
        self.base_high = build_target(target_name, temperature=target_temperature, device="cpu", **target_kwargs)

    # ------------------------------------------------------------------
    # Core forward / reverse operations
    # ------------------------------------------------------------------
    def transform(self, coords: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        """Apply the flow and return (coords_out, log_det).

        Accepts coordinate tensors in any of the following shapes
        and canonicalises them to `[B, N, 3]` before passing them to
        the underlying flow:
            • `[B, N, 3]`  – standard batch.
            • `[B, 1, N, 3]` – batch where a singleton dummy dimension is
              present (e.g. coming from a collate_fn that stacked samples
              of shape `[1, N, 3]`).
            • `[N, 3]` – single sample without batch dimension.
        """
        # Handle single sample without batch dim
        if coords.ndim == 2:  # [N, 3]
            coords = coords.unsqueeze(0)

        # Handle DataLoader outputs of shape [B, 1, N, 3]
        if coords.ndim == 4 and coords.shape[1] == 1:
            coords = coords.squeeze(1)  # -> [B, N, 3]

        if coords.ndim != 3:
            raise ValueError(
                "coords must have shape [B, N, 3] after canonicalisation; "
                f"got {coords.shape} instead."
            )

        return self.flow(coords.to(self.device), None, None, reverse=reverse)

    # Aliases
    def forward(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        return self.transform(coords, reverse=False)

    def inverse(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        return self.transform(coords, reverse=True)

    # ------------------------------------------------------------------
    # Placeholder likelihood – to be replaced with proper physics-aware one.
    # ------------------------------------------------------------------
    def log_likelihood(
        self,
        x_coords: Tensor,  # source temperature
        y_coords: Tensor,  # target temperature
        reverse: bool = False,
    ) -> Tensor:
        """Return log p(y|x) using change-of-variables and Boltzmann prior.

        For the *forward* direction (low → high) we assume the flow `f` maps
        coordinates sampled at the lower temperature to the corresponding
        higher-temperature configuration.  The change-of-variables formula is

            log p(y | x) = log p_low(f⁻¹(y)) + log|det J_{f⁻¹}(y)|,

        where `J_{f⁻¹}` is the Jacobian of the inverse map.

        For the *reverse* direction (high → low) we instead use the forward
        map `f` (because it *is* the inverse for the reverse conditioning)
        together with the Boltzmann prior at the *high* temperature.
        """
        if reverse:
            # high → low               (y is low-T coords)
            base = self.base_high  # prior corresponds to *high* temperature
            # Forward map takes low-T coords to high-T space.
            x_recon, log_det = self.forward(y_coords)  # log_det = log|det J_f|
        else:
            # low → high               (y is high-T coords)
            base = self.base_low
            # Inverse map takes high-T coords back to low-T space.
            x_recon, log_det = self.inverse(y_coords)  # log_det = log|det J_{f⁻¹}|

        # Flatten coordinates for Boltzmann prior evaluation
        B, N, _ = x_recon.shape
        x_flat = x_recon.reshape(B, -1)

        log_p_base = base.log_prob(x_flat)  # [B]

        return log_p_base + log_det

    # ------------------------------------------------------------------
    def sample_proposal(self, source_coords: Tensor, direction: str = "forward") -> Tensor:
        """Generate deterministic proposal given source coordinates."""
        reverse = direction != "forward"  # forward = low→high => normal flow
        y, _ = self.transform(source_coords, reverse=reverse)
        return y 