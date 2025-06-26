from __future__ import annotations

"""Loss functions for PT swap flow training.

For now we implement a *placeholder* bidirectional negative log-likelihood that
falls back to a dummy value until `PTSwapFlow.log_likelihood` is fully
implemented.
"""

import torch
from torch import Tensor

__all__ = ["bidirectional_nll"]


def bidirectional_nll(model, batch, energy_threshold: float | None = None) -> Tensor:
    """Compute bidirectional NLL loss.

    If `model.log_likelihood` raises `NotImplementedError`, returns a zero loss
    tensor so training code can run smoke tests without the full physics-aware
    likelihood.
    """
    # Per-sample losses (no mean yet)
    loss_forward = -model.log_likelihood(
        x_coords=batch["source_coords"],
        y_coords=batch["target_coords"],
        reverse=False,
    )  # [B]
    loss_reverse = -model.log_likelihood(
        x_coords=batch["target_coords"],
        y_coords=batch["source_coords"],
        reverse=True,
    )  # [B]

    loss_per_sample = loss_forward + loss_reverse  # [B]

    if energy_threshold is not None and energy_threshold > 0:
        # Compute potential energies (kJ/mol) via model bases
        def _energy(coords_flat, base):
            # coords_flat: [B, dim]
            logp = base.log_prob(coords_flat)  # −β U
            return -logp / base.beta  # U in kJ/mol

        B = batch["source_coords"].shape[0]
        x_flat = batch["source_coords"].reshape(B, -1)
        y_flat = batch["target_coords"].reshape(B, -1)

        energy_x = _energy(x_flat, model.base_low)  # [B]
        energy_y = _energy(y_flat, model.base_high)  # [B]

        keep_mask = (energy_x < energy_threshold) & (energy_y < energy_threshold)

        if keep_mask.any():
            loss_per_sample = loss_per_sample[keep_mask]
        else:
            # All samples bad – return large constant
            return torch.tensor(1e4, device=loss_per_sample.device)

    loss = loss_per_sample.mean()
    return loss 