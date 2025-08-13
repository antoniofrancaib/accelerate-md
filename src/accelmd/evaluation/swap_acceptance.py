from __future__ import annotations

"""Utilities for computing swap‐acceptance rates for PT temperature pairs.

This module provides two convenience functions:

    1. ``naive_acceptance`` – Metropolis acceptance when simply exchanging the
       coordinates of two replicas (no flow).

    2. ``flow_acceptance`` – Acceptance when a deterministic, invertible flow
       is used to *morph* the coordinates before exchange as proposed in
       Boltzmann Generators.

Both helpers iterate over a ``DataLoader`` that yields the dictionary created
by :class:`accelmd.data.PTTemperaturePairDataset`.  The energy is evaluated via
:class:`accelmd.targets.AldpBoltzmann` instances corresponding to the two
replica temperatures.  The functions return the mean acceptance    rate as a
Python ``float``.

The implementation keeps all heavy computations on *CPU* because the
underlying OpenMM context in :pyclass:`AldpBoltzmann` is CPU‐only.  The flow
model may live on GPU; tensors are moved as needed.
"""

from typing import Iterable, Any

import torch
from torch.utils.data import DataLoader

from ..flows import PTSwapFlow, PTSwapGraphFlow

__all__ = [
    "naive_acceptance",
    "flow_acceptance",
]


def _energy(target: Any, coords: torch.Tensor) -> torch.Tensor:
    """Return *potential* energy (kJ/mol) for a batch of coordinates."""
    log_p = target.log_prob(coords)  # −β U
    return (-log_p) / target.beta  # strip β factor -> U


@torch.no_grad()
def naive_acceptance(
    loader: DataLoader,
    base_low: Any,
    base_high: Any,
    max_batches: int | None = None,
) -> float:
    """Estimate naïve swap‐acceptance rate over the given ``loader``.

    Parameters
    ----------
    loader
        ``DataLoader`` yielding batched samples with keys ``source_coords`` and
        ``target_coords`` (shapes ``[B,N,3]``).
    base_low, base_high
        Boltzmann targets at the two replica temperatures.
    max_batches
        If given, stop after this many batches to bound run time.
    """
    acc_sum = 0.0
    n_samples = 0

    beta_low = base_low.beta
    beta_high = base_high.beta

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        x_low = batch["source_coords"].view(batch["source_coords"].shape[0], -1)
        x_high = batch["target_coords"].view(batch["target_coords"].shape[0], -1)

        # Energies (kJ/mol)
        U_low = _energy(base_low, x_low)
        U_high = _energy(base_high, x_high)

        # Δ = (β_low - β_high) * (U_low - U_high)
        log_acc = (beta_low - beta_high) * (U_low - U_high)
        acc = torch.minimum(torch.ones_like(log_acc), log_acc.exp())

        acc_sum += acc.mean().item() * len(acc)
        n_samples += len(acc)

    return acc_sum / max(n_samples, 1)


@torch.no_grad()
def flow_acceptance(
    loader: DataLoader,
    model: PTSwapFlow | PTSwapGraphFlow,  # Note: PTSwapTransformerFlow will be added when imported
    base_low: Any,
    base_high: Any,
    device: str = "cpu",
    max_batches: int | None = None,
) -> float:
    """Estimate swap‐acceptance when using ``model`` as proposal.

    The deterministic proposal follows the Boltzmann‐Generator recipe:

    ``y_high = f(x_low)`` and ``y_low = f⁻¹(x_high)``

    Acceptance ratio:

    ``α = exp( -β_low U(y_low) - β_high U(y_high) + β_low U(x_low) + β_high U(x_high)
               + log|det J_f(x_low)| + log|det J_{f⁻¹}(x_high)| )``
    """
    model_device = torch.device(device)
    model.eval()

    acc_sum = 0.0
    n_samples = 0

    beta_low = base_low.beta
    beta_high = base_high.beta

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        x_low = batch["source_coords"].to(model_device)  # [B,N,3]
        x_high = batch["target_coords"].to(model_device)

        # Forward / inverse transforms + log‐dets
        from ..flows.pt_swap_transformer_flow import PTSwapTransformerFlow
        if isinstance(model, (PTSwapGraphFlow, PTSwapTransformerFlow)):
            # Graph/transformer-conditioned flow requires additional molecular data
            atom_types = batch["atom_types"].to(model_device)
            adj_list = batch.get("adj_list")
            edge_batch_idx = batch.get("edge_batch_idx")
            masked_elements = batch.get("masked_elements")
            
            if adj_list is not None:
                adj_list = adj_list.to(model_device)
            if edge_batch_idx is not None:
                edge_batch_idx = edge_batch_idx.to(model_device)
            if masked_elements is not None:
                masked_elements = masked_elements.to(model_device)
            
            # Graph/transformer flow uses different interface
            if isinstance(model, PTSwapTransformerFlow):
                # For transformer, we need to provide velocities but only count position log-dets
                # Sample zero velocities (we only care about position transport)
                v_low = torch.zeros_like(x_low)   # [B, N, 3]
                v_high = torch.zeros_like(x_high) # [B, N, 3]
                
                # Concatenate positions and velocities
                qv_low = torch.cat([x_low, v_low], dim=1)    # [B, 2N, 3]
                qv_high = torch.cat([x_high, v_high], dim=1) # [B, 2N, 3]
                
                # Forward: position-only evaluation
                qv_high_full, _, log_det_f_pos, _ = model.forward(
                    coordinates=qv_low,
                    atom_types=atom_types,
                    masked_elements=masked_elements,
                    reverse=False,
                    return_log_det=True,
                    return_separate_log_dets=True
                )
                y_high = qv_high_full[:, :x_low.shape[1], :]  # Extract positions only
                
                # Inverse: position-only evaluation  
                qv_low_full, _, log_det_inv_pos, _ = model.forward(
                    coordinates=qv_high,
                    atom_types=atom_types,
                    masked_elements=masked_elements,
                    reverse=True,
                    return_log_det=True,
                    return_separate_log_dets=True
                )
                y_low = qv_low_full[:, :x_high.shape[1], :]  # Extract positions only
                
                # Use only position log-dets for proper evaluation
                log_det_f = log_det_f_pos
                log_det_inv = log_det_inv_pos
                
            else:
                # Graph flow uses different interface
                y_high, log_det_f = model.forward(
                    coordinates=x_low,
                    atom_types=atom_types,
                    adj_list=adj_list,
                    edge_batch_idx=edge_batch_idx,
                    masked_elements=masked_elements,
                    reverse=False
                )  # low → high
                
                y_low, log_det_inv = model.forward(
                    coordinates=x_high,
                    atom_types=atom_types,
                    adj_list=adj_list,
                    edge_batch_idx=edge_batch_idx,
                    masked_elements=masked_elements,
                    reverse=True
                )  # high → low
            
        else:
            # Simple flow (PTSwapFlow)
            y_high, log_det_f = model.forward(x_low)           # low → high
            y_low, log_det_inv = model.inverse(x_high)         # high → low

        # Energies evaluated on CPU (OpenMM)
        y_high_flat = y_high.view(y_high.shape[0], -1).cpu()
        y_low_flat = y_low.view(y_low.shape[0], -1).cpu()
        x_low_flat = x_low.view(x_low.shape[0], -1).cpu()
        x_high_flat = x_high.view(x_high.shape[0], -1).cpu()

        U_y_low = _energy(base_low, y_low_flat)
        U_y_high = _energy(base_high, y_high_flat)
        U_x_low = _energy(base_low, x_low_flat)
        U_x_high = _energy(base_high, x_high_flat)

        log_acc = (
            -beta_low * U_y_low
            - beta_high * U_y_high
            + beta_low * U_x_low
            + beta_high * U_x_high
            + log_det_f.cpu()
            + log_det_inv.cpu()
        )
        acc = torch.minimum(torch.ones_like(log_acc), log_acc.exp())

        acc_sum += acc.mean().item() * len(acc)
        n_samples += len(acc)

    return acc_sum / max(n_samples, 1) 