from __future__ import annotations

"""Loss functions for PT swap flow training.

For now we implement a *placeholder* bidirectional negative log-likelihood that
falls back to a dummy value until `PTSwapFlow.log_likelihood` is fully
implemented.
"""

import torch
from torch import Tensor

__all__ = ["bidirectional_nll"]


def bidirectional_nll(
    model,
    batch,
    energy_threshold: float | None = None,
    return_components: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Compute bidirectional NLL loss.

    Parameters
    ----------
    model
        Flow model exposing ``log_likelihood``.
    batch
        Mini-batch dict from ``PTTemperaturePairDataset``.
    energy_threshold
        If set, samples whose *potential* energy (kJ/mol) exceeds this value
        for either temperature are discarded from the loss (as in Timewarp's
        energy gating).
    return_components
        If ``True`` the function returns a triple ``(total, forward, reverse)``
        where *forward* = NLL for low→high and *reverse* = NLL for high→low.
        Otherwise only the scalar total loss is returned (previous behaviour –
        keeps unit-tests untouched).
    """

    # ----------------------------------------------------------
    # 1. Per-sample forward / reverse NLL
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 2. Optional energy gating (filter out unphysical frames)
    # ----------------------------------------------------------
    if energy_threshold is not None and energy_threshold > 0:
        def _energy(coords_flat, base):
            logp = base.log_prob(coords_flat)  # −β U
            return -logp / base.beta          # U in kJ/mol

        B = batch["source_coords"].shape[0]
        x_flat = batch["source_coords"].reshape(B, -1)
        y_flat = batch["target_coords"].reshape(B, -1)

        energy_x = _energy(x_flat, model.base_low)
        energy_y = _energy(y_flat, model.base_high)

        keep_mask = (energy_x < energy_threshold) & (energy_y < energy_threshold)
        if keep_mask.any():
            loss_per_sample = loss_per_sample[keep_mask]
        else:
            return torch.tensor(1e4, device=loss_per_sample.device)

    loss_total = loss_per_sample.mean()

    if return_components:
        return loss_total, loss_forward.mean(), loss_reverse.mean()
    return loss_total 