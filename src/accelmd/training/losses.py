from __future__ import annotations

"""Loss functions for PT swap flow training.

Includes both negative log-likelihood and acceptance-loss implementations.
"""

import torch
from torch import Tensor
from typing import Dict

from ..flows.pt_swap_graph_flow import PTSwapGraphFlow

__all__ = ["bidirectional_nll", "acceptance_loss", "augmented_bidirectional_nll", "augmented_acceptance_loss"]

# -----------------------------------------------------------------------------
# Simple in-memory cache for peptide-specific Boltzmann targets.  Without this
# multi-peptide training repeatedly reparses identical PDB files and eventually
# triggers OpenMM parser errors.  The cache key is “<PEP_CODE>_<TEMP_K>”.
# -----------------------------------------------------------------------------

_TARGET_CACHE: dict[str, object] = {}


def bidirectional_nll(
    model,
    batch,
    energy_threshold: float | None = None,
    return_components: bool = False,
    current_epoch: int | None = None,
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

    # Check if this is a flow that needs molecular data
    from ..flows.pt_swap_graph_flow import PTSwapGraphFlow
    from ..flows.pt_swap_transformer_flow import PTSwapTransformerFlow
    molecular_kwargs = {}
    if isinstance(model, (PTSwapGraphFlow, PTSwapTransformerFlow)):
        # Pass molecular data to graph and transformer flows
        if "atom_types" in batch:
            molecular_kwargs["atom_types"] = batch["atom_types"]
        if "adj_list" in batch:
            molecular_kwargs["adj_list"] = batch["adj_list"]
        if "edge_batch_idx" in batch:
            molecular_kwargs["edge_batch_idx"] = batch["edge_batch_idx"]
        if "masked_elements" in batch:
            molecular_kwargs["masked_elements"] = batch["masked_elements"]
        # Pass peptide information for multi-peptide mode
        if "peptide_names" in batch:
            molecular_kwargs["peptide_names"] = batch["peptide_names"]

    # ----------------------------------------------------------
    # 1. Per-sample forward / reverse NLL
    # ----------------------------------------------------------
    
    # Add current_epoch to molecular_kwargs if model supports it
    if current_epoch is not None and isinstance(model, PTSwapGraphFlow):
        molecular_kwargs["current_epoch"] = current_epoch
    
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
    # Skip energy gating for multi-peptide batches (peptide_names indicates mixed batch)
    if energy_threshold is not None and energy_threshold > 0 and "peptide_names" not in batch:
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
        Flow model with forward/inverse and base targets.
    batch
        Mini-batch dict from PTTemperaturePairDataset.
    beta_low, beta_high
        Inverse temperatures of two replicas.
    clamp
        Apply max(0, Δ) clamp to loss.
    energy_threshold
        Discard samples with potential energy > threshold.
    """
    x_low = batch["source_coords"]
    x_high = batch["target_coords"]

    # Check if this is a molecular flow that needs structural data
    from ..flows.pt_swap_graph_flow import PTSwapGraphFlow
    from ..flows.pt_swap_transformer_flow import PTSwapTransformerFlow
    
    molecular_kwargs = {}
    if isinstance(model, (PTSwapGraphFlow, PTSwapTransformerFlow)):
        # Pass molecular data to graph and transformer flows
        if "atom_types" in batch:
            molecular_kwargs["atom_types"] = batch["atom_types"]
        if "adj_list" in batch:
            molecular_kwargs["adj_list"] = batch["adj_list"]
        if "edge_batch_idx" in batch:
            molecular_kwargs["edge_batch_idx"] = batch["edge_batch_idx"]
        if "masked_elements" in batch:
            molecular_kwargs["masked_elements"] = batch["masked_elements"]

        # Flow transforms with molecular data
        y_high, log_det_f = model.forward(x_low, **molecular_kwargs)   # low → high
        y_low, log_det_inv = model.inverse(x_high, **molecular_kwargs) # high → low
    else:
        # Flow transforms for simple model
        y_high, log_det_f = model.forward(x_low)   # low → high
        y_low, log_det_inv = model.inverse(x_high) # high → low

    # FIXED: Handle multi-peptide mode for energy evaluation
    def _get_energy_multi_peptide(coords_flat, beta, temperature):
        """Get potential energy with multi-peptide support."""
        if "peptide_names" in batch and len(set(batch["peptide_names"])) == 1:
            # Uniform batch - all samples are the same peptide type
            peptide_name = batch["peptide_names"][0]
            
            # Create peptide-specific target
            from ..targets import build_target
            cache_key = f"{peptide_name.upper()}_{int(round(temperature))}"
            if cache_key in _TARGET_CACHE:
                base_target = _TARGET_CACHE[cache_key]
            else:
                if peptide_name.upper() == "AX":
                    base_target = build_target(
                        "aldp",
                        temperature=temperature,
                        device="cpu",
                    )
                else:
                    pdb_path = f"datasets/pt_dipeptides/{peptide_name}/ref.pdb"
                    base_target = build_target(
                        "dipeptide",
                        temperature=temperature,
                        device="cpu",
                        pdb_path=pdb_path,
                        env="implicit",
                    )
                _TARGET_CACHE[cache_key] = base_target
            
            log_prob = base_target.log_prob(coords_flat)  # -β U
            return -log_prob / beta  # U in kJ/mol
        else:
            # Fall back to model's base targets (single-peptide mode or mixed batch)
            base_target = model.base_low if abs(beta - model.base_low.beta) < 1e-6 else model.base_high
            log_prob = base_target.log_prob(coords_flat)  # -β U
            return -log_prob / beta  # U in kJ/mol

    # Energies (kJ/mol) – use peptide-specific targets for multi-peptide mode
    U_x_low = _get_energy_multi_peptide(_flatten(x_low), beta_low, 1.0 / (beta_low * 8.314e-3))      # [B]
    U_x_high = _get_energy_multi_peptide(_flatten(x_high), beta_high, 1.0 / (beta_high * 8.314e-3))
    U_y_low = _get_energy_multi_peptide(_flatten(y_low), beta_low, 1.0 / (beta_low * 8.314e-3))
    U_y_high = _get_energy_multi_peptide(_flatten(y_high), beta_high, 1.0 / (beta_high * 8.314e-3))

    # Metropolis exponent (negative log α)
    delta = (
        beta_low * U_y_low
        + beta_high * U_y_high
        - beta_low * U_x_low
        - beta_high * U_x_high
        - log_det_f
        - log_det_inv
    )

    # Apply energy gating if requested
    if energy_threshold is not None:
        mask = (U_x_low <= energy_threshold) & (U_x_high <= energy_threshold)
        if not mask.any():
            return torch.tensor(0.0, device=delta.device)
        delta = delta[mask]

    # Clamp and return
    if clamp:
        delta = torch.clamp(delta, min=0.0)
    return delta.mean()


def augmented_bidirectional_nll(
    model,
    batch,
    energy_threshold: float | None = None,
    return_components: bool = False,
    current_epoch: int | None = None,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Compute bidirectional NLL loss for transformer in augmented (q,v) space.
    
    For transformer flows that operate on positions + toy velocities, this computes
    the NLL using the full augmented Boltzmann density:
    p_β(q,v) ∝ exp[-β U(q) - 0.5‖v‖²]
    
    where velocities are sampled from N(0,I) and kinetic energy is temperature-independent.
    """
    from ..flows.pt_swap_transformer_flow import PTSwapTransformerFlow
    
    if not isinstance(model, PTSwapTransformerFlow):
        raise ValueError("augmented_bidirectional_nll only supports PTSwapTransformerFlow")
    
    # Extract positions and velocities
    q_low = batch["source_coords"]  # [B, N, 3]
    q_high = batch["target_coords"]  # [B, N, 3]
    
    # Sample toy velocities N(0,I) for each batch
    v_low = torch.randn_like(q_low)   # [B, N, 3]
    v_high = torch.randn_like(q_high) # [B, N, 3]
    
    # Molecular data for transformer
    molecular_kwargs = {}
    if "atom_types" in batch:
        molecular_kwargs["atom_types"] = batch["atom_types"]
    if "masked_elements" in batch:
        molecular_kwargs["masked_elements"] = batch["masked_elements"]
    # Note: PTSwapTransformerFlow doesn't accept peptide_names parameter
    # if "peptide_names" in batch:
    #     molecular_kwargs["peptide_names"] = batch["peptide_names"]
    
    # Forward transform: (q_low, v_low) → (q_high', v_high')
    qv_low = torch.cat([q_low, v_low], dim=1)  # [B, 2N, 3]
    qv_high_prime, log_det_f = model.forward(
        coordinates=qv_low,
        reverse=False,
        return_log_det=True,
        **molecular_kwargs
    )
    q_high_prime = qv_high_prime[:, :q_low.shape[1], :]  # [B, N, 3]
    v_high_prime = qv_high_prime[:, q_low.shape[1]:, :]  # [B, N, 3]
    
    # Inverse transform: (q_high, v_high) → (q_low', v_low')
    qv_high = torch.cat([q_high, v_high], dim=1)  # [B, 2N, 3]
    qv_low_prime, log_det_inv = model.forward(
        coordinates=qv_high,
        reverse=True,
        return_log_det=True,
        **molecular_kwargs
    )
    q_low_prime = qv_low_prime[:, :q_high.shape[1], :]   # [B, N, 3]
    v_low_prime = qv_low_prime[:, q_high.shape[1]:, :]   # [B, N, 3]
    
    # Get temperatures
    beta_low = model.source_target.beta
    beta_high = model.target_target.beta
    
    # Helper to get potential energy with multi-peptide support
    def _get_potential_energy(coords, base_target):
        coords_flat = coords.reshape(coords.shape[0], -1)
        if "peptide_names" in batch and len(set(batch["peptide_names"])) == 1:
            # Uniform batch - use peptide-specific target
            peptide_name = batch["peptide_names"][0]
            from ..targets import build_target
            temperature = 1.0 / (base_target.beta * 8.314e-3)
            cache_key = f"{peptide_name.upper()}_{int(round(temperature))}"
            if cache_key in _TARGET_CACHE:
                target = _TARGET_CACHE[cache_key]
            else:
                if peptide_name.upper() == "AX":
                    target = build_target("aldp", temperature=temperature, device="cpu")
                else:
                    pdb_path = f"datasets/pt_dipeptides/{peptide_name}/ref.pdb"
                    target = build_target("dipeptide", temperature=temperature, device="cpu",
                                        pdb_path=pdb_path, env="implicit")
                _TARGET_CACHE[cache_key] = target
            log_prob = target.log_prob(coords_flat)
        else:
            log_prob = base_target.log_prob(coords_flat)
        return -log_prob / base_target.beta  # U in kJ/mol
    
    # Potential energies
    U_q_high_prime = _get_potential_energy(q_high_prime, model.target_target)
    U_q_low_prime = _get_potential_energy(q_low_prime, model.source_target)
    
    # Kinetic energies (0.5 * ||v||²)
    K_v_high_prime = 0.5 * (v_high_prime**2).sum(dim=(-1, -2))  # [B]
    K_v_low_prime = 0.5 * (v_low_prime**2).sum(dim=(-1, -2))    # [B]
    
    # Forward loss: β_high * (U + K) - log|det J_f|
    loss_forward = beta_high * U_q_high_prime + K_v_high_prime - log_det_f
    
    # Reverse loss: β_low * (U + K) - log|det J_inv|
    loss_reverse = beta_low * U_q_low_prime + K_v_low_prime - log_det_inv
    
    loss_per_sample = loss_forward + loss_reverse
    
    # Optional energy gating (only on potential energy, not kinetic)
    if energy_threshold is not None and energy_threshold > 0 and "peptide_names" not in batch:
        U_q_low = _get_potential_energy(q_low, model.source_target)
        U_q_high = _get_potential_energy(q_high, model.target_target)
        keep_mask = (U_q_low < energy_threshold) & (U_q_high < energy_threshold)
        keep_mask = keep_mask.to(loss_per_sample.device)
        if keep_mask.any():
            loss_per_sample = loss_per_sample[keep_mask]
        else:
            return torch.tensor(1e4, device=loss_per_sample.device)
    
    loss_total = loss_per_sample.mean()
    
    if return_components:
        return loss_total, loss_forward.mean(), loss_reverse.mean()
    return loss_total


def augmented_acceptance_loss(
    model,
    batch: Dict[str, Tensor],
    beta_low: float,
    beta_high: float,
    clamp: bool = True,
    energy_threshold: float | None = None,
) -> Tensor:
    """Compute acceptance loss for transformer in augmented (q,v) space.
    
    Includes both potential energy U(q) and kinetic energy 0.5‖v‖² in the
    Metropolis exponent to prevent velocity-scaling shortcuts.
    """
    from ..flows.pt_swap_transformer_flow import PTSwapTransformerFlow
    
    if not isinstance(model, PTSwapTransformerFlow):
        raise ValueError("augmented_acceptance_loss only supports PTSwapTransformerFlow")
    
    # Extract positions
    q_low = batch["source_coords"]   # [B, N, 3]
    q_high = batch["target_coords"]  # [B, N, 3]
    
    # Sample toy velocities N(0,I)
    v_low = torch.randn_like(q_low)   # [B, N, 3]
    v_high = torch.randn_like(q_high) # [B, N, 3]
    
    # Molecular data
    molecular_kwargs = {}
    if "atom_types" in batch:
        molecular_kwargs["atom_types"] = batch["atom_types"]
    if "masked_elements" in batch:
        molecular_kwargs["masked_elements"] = batch["masked_elements"]
    
    # Flow transforms
    qv_low = torch.cat([q_low, v_low], dim=1)    # [B, 2N, 3]
    qv_high = torch.cat([q_high, v_high], dim=1) # [B, 2N, 3]
    
    qv_high_prime, log_det_f = model.forward(qv_low, **molecular_kwargs)     # low → high
    qv_low_prime, log_det_inv = model.forward(qv_high, reverse=True, **molecular_kwargs) # high → low
    
    # Split back into positions and velocities
    q_high_prime = qv_high_prime[:, :q_low.shape[1], :]
    v_high_prime = qv_high_prime[:, q_low.shape[1]:, :]
    q_low_prime = qv_low_prime[:, :q_high.shape[1], :]
    v_low_prime = qv_low_prime[:, q_high.shape[1]:, :]
    
    # Helper for potential energy with multi-peptide support
    def _get_potential_energy(coords, beta, temperature):
        coords_flat = coords.reshape(coords.shape[0], -1)
        if "peptide_names" in batch and len(set(batch["peptide_names"])) == 1:
            peptide_name = batch["peptide_names"][0]
            from ..targets import build_target
            cache_key = f"{peptide_name.upper()}_{int(round(temperature))}"
            if cache_key in _TARGET_CACHE:
                target = _TARGET_CACHE[cache_key]
            else:
                if peptide_name.upper() == "AX":
                    target = build_target("aldp", temperature=temperature, device="cpu")
                else:
                    pdb_path = f"datasets/pt_dipeptides/{peptide_name}/ref.pdb"
                    target = build_target("dipeptide", temperature=temperature, device="cpu",
                                        pdb_path=pdb_path, env="implicit")
                _TARGET_CACHE[cache_key] = target
            log_prob = target.log_prob(coords_flat)
            return -log_prob / beta
        else:
            base_target = model.base_low if abs(beta - model.base_low.beta) < 1e-6 else model.base_high
            log_prob = base_target.log_prob(coords_flat)
            return -log_prob / beta
    
    # Potential energies
    U_q_low = _get_potential_energy(q_low, beta_low, 1.0 / (beta_low * 8.314e-3))
    U_q_high = _get_potential_energy(q_high, beta_high, 1.0 / (beta_high * 8.314e-3))
    U_q_low_prime = _get_potential_energy(q_low_prime, beta_low, 1.0 / (beta_low * 8.314e-3))
    U_q_high_prime = _get_potential_energy(q_high_prime, beta_high, 1.0 / (beta_high * 8.314e-3))
    
    # Kinetic energies
    K_v_low = 0.5 * (v_low**2).sum(dim=(-1, -2))
    K_v_high = 0.5 * (v_high**2).sum(dim=(-1, -2))
    K_v_low_prime = 0.5 * (v_low_prime**2).sum(dim=(-1, -2))
    K_v_high_prime = 0.5 * (v_high_prime**2).sum(dim=(-1, -2))
    
    # Metropolis exponent (negative log α) including both U and K
    delta = (
        beta_low * U_q_low_prime + beta_high * U_q_high_prime
        - beta_low * U_q_low - beta_high * U_q_high
        + K_v_low_prime + K_v_high_prime - K_v_low - K_v_high
        - log_det_f - log_det_inv
    )
    
    # Apply energy gating if requested (only on potential energy)
    if energy_threshold is not None:
        mask = (U_q_low <= energy_threshold) & (U_q_high <= energy_threshold)
        if not mask.any():
            return torch.tensor(0.0, device=delta.device)
        delta = delta[mask]
    
    # Clamp and return
    if clamp:
        delta = torch.clamp(delta, min=0.0)
    return delta.mean() 