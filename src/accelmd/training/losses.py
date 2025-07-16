from __future__ import annotations

"""Loss functions for PT swap flow training.

Includes both negative log-likelihood and acceptance-loss implementations.
"""

import torch
from torch import Tensor
from typing import Dict

__all__ = ["bidirectional_nll", "acceptance_loss"]


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

    # Check if this is a graph flow that needs molecular data
    from ..flows.pt_swap_graph_flow import PTSwapGraphFlow
    molecular_kwargs = {}
    if isinstance(model, PTSwapGraphFlow):
        # Only pass molecular data to graph flows
        if "atom_types" in batch:
            molecular_kwargs["atom_types"] = batch["atom_types"]
        if "adj_list" in batch:
            molecular_kwargs["adj_list"] = batch["adj_list"]
        if "edge_batch_idx" in batch:
            molecular_kwargs["edge_batch_idx"] = batch["edge_batch_idx"]
        if "masked_elements" in batch:
            molecular_kwargs["masked_elements"] = batch["masked_elements"]

    # ----------------------------------------------------------
    # 1. Per-sample forward / reverse NLL
    # ----------------------------------------------------------
    loss_forward = -model.log_likelihood(
        x_coords=batch["source_coords"],
        y_coords=batch["target_coords"],
        reverse=False,
        **molecular_kwargs,  # Only passed for graph flows
    )  # [B]
    loss_reverse = -model.log_likelihood(
        x_coords=batch["target_coords"],
        y_coords=batch["source_coords"],
        reverse=True,
        **molecular_kwargs,  # Only passed for graph flows
    )  # [B]

    loss_per_sample = loss_forward + loss_reverse

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
        # Ensure mask is on same device as loss_per_sample
        keep_mask = keep_mask.to(loss_per_sample.device)
        if keep_mask.any():
            loss_per_sample = loss_per_sample[keep_mask]
        else:
            return torch.tensor(1e4, device=loss_per_sample.device)

    loss_total = loss_per_sample.mean()

    if return_components:
        return loss_total, loss_forward.mean(), loss_reverse.mean()
    return loss_total


def _flatten(coords: Tensor) -> Tensor:
    """Return `[B,66]` from `[B,N,3]` or already-flat tensor."""
    if coords.ndim >= 3:
        return coords.reshape(coords.shape[0], -1)
    return coords


def acceptance_loss(
    model,
    batch: Dict[str, Tensor],
    beta_low: float,
    beta_high: float,
    clamp: bool = True,
    energy_threshold: float | None = None,
) -> Tensor:
    """Compute mean acceptance-oriented loss for one mini-batch.
    
    Acceptance-loss implementation (position–only) inspired by Timewarp.
    The loss minimises the Metropolis exponent that governs swap acceptance
    between two temperatures when using a deterministic, invertible flow.

        L = ⟨ clamp(Δ, min=0) ⟩
        where
            Δ = -log α  = (β_low U(y_low) + β_high U(y_high)
                           -β_low U(x_low) - β_high U(x_high)
                           - log|det J_f| - log|det J_{f⁻¹}|)

    `clamp=True` limits Δ≤0 so perfectly accepted moves contribute zero loss.

    No velocities, no entropy term for v0.

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
    from .openmm_bridge import compute_potential_energy  # Import from training folder
    
    device = next(model.parameters()).device
    x_low = batch["source_coords"].to(device)
    x_high = batch["target_coords"].to(device)

    # Check if this is a graph flow that needs molecular data
    from ..flows.pt_swap_graph_flow import PTSwapGraphFlow
    if isinstance(model, PTSwapGraphFlow):
        molecular_kwargs = {}
        if "atom_types" in batch:
            molecular_kwargs["atom_types"] = batch["atom_types"].to(device)
        if "adj_list" in batch:
            molecular_kwargs["adj_list"] = batch["adj_list"].to(device)
        if "edge_batch_idx" in batch:
            molecular_kwargs["edge_batch_idx"] = batch["edge_batch_idx"].to(device)
        if "masked_elements" in batch:
            molecular_kwargs["masked_elements"] = batch["masked_elements"].to(device)
        
        # Flow transforms for graph model
        y_high, log_det_f = model.forward(x_low, **molecular_kwargs)   # low → high
        y_low, log_det_inv = model.inverse(x_high, **molecular_kwargs) # high → low
    else:
        # Flow transforms for simple model
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
        # Ensure mask is on same device as neg_log_alpha
        keep_mask = keep_mask.to(neg_log_alpha.device)
        if keep_mask.any():
            neg_log_alpha = neg_log_alpha[keep_mask]
        else:
            return torch.tensor(1e4, device=device)

    loss = neg_log_alpha.mean()
    return loss 