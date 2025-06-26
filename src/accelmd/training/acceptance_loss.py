from __future__ import annotations

"""Acceptance-loss implementation (position–only) inspired by Timewarp.

The loss minimises the Metropolis exponent that governs swap acceptance
between two temperatures when using a deterministic, invertible flow.

    L = ⟨ clamp(Δ, min=0) ⟩
    where
        Δ = -log α  = (β_low U(y_low) + β_high U(y_high)
                       -β_low U(x_low) - β_high U(x_high)
                       - log|det J_f| - log|det J_{f⁻¹}|)

`clamp=True` limits Δ≤0 so perfectly accepted moves contribute zero loss.

No velocities, no entropy term for v0.
"""

from typing import Dict, Tuple
import torch
from torch import Tensor

from ..physics.openmm_bridge import compute_potential_energy

__all__ = ["acceptance_loss"]


def _flatten(coords: Tensor) -> Tensor:
    """Return `[B,66]` from `[B,N,3]` or already-flat tensor."""
    if coords.ndim >= 3:
        return coords.reshape(coords.shape[0], -1)
    return coords


# pylint: disable=too-many-locals
def acceptance_loss(
    model,
    batch: Dict[str, Tensor],
    beta_low: float,
    beta_high: float,
    clamp: bool = True,
    energy_threshold: float | None = None,
) -> Tensor:
    """Compute mean acceptance-oriented loss for one mini-batch.

    Parameters
    ----------
    model
        Flow model exposing `.forward()` (low→high) and `.inverse()` (high→low).
    batch
        Dict with keys `source_coords`, `target_coords` (shapes `[B,N,3]`).
    beta_low, beta_high
        1/(k_B T) for the two replicas (dimensionless).
    clamp
        Whether to clamp the Metropolis exponent so that Δ<0 (α>1) yields 0 loss.
    energy_threshold
        Optional energy threshold for clipping.
    """
    device = next(model.parameters()).device
    x_low = batch["source_coords"].to(device)
    x_high = batch["target_coords"].to(device)

    # Flow transforms
    y_high, log_det_f = model.forward(x_low)   # low → high
    y_low, log_det_inv = model.inverse(x_high) # high → low

    # Energies (kJ/mol) – differentiable via autograd bridge
    U_x_low = compute_potential_energy(_flatten(x_low))      # [B]
    U_x_high = compute_potential_energy(_flatten(x_high))
    U_y_low = compute_potential_energy(_flatten(y_low))
    U_y_high = compute_potential_energy(_flatten(y_high))

    # Metropolis exponent (negative log α)
    neg_log_alpha = (
        beta_low * U_y_low + beta_high * U_y_high
        - beta_low * U_x_low - beta_high * U_x_high
        - log_det_f - log_det_inv
    )

    if clamp:
        neg_log_alpha = torch.clamp(neg_log_alpha, min=0.0)

    # Optional energy-based masking to drop pathological samples
    if energy_threshold is not None and energy_threshold > 0:
        keep_mask = (
            (U_x_low < energy_threshold)
            & (U_x_high < energy_threshold)
            & (U_y_low < energy_threshold)
            & (U_y_high < energy_threshold)
        )
        if keep_mask.any():
            neg_log_alpha = neg_log_alpha[keep_mask]
        else:
            return torch.tensor(1e4, device=device)

    loss = neg_log_alpha.mean()
    return loss 