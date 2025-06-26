from __future__ import annotations

"""Differentiable OpenMM energy wrapper tailor-made for Alanine Dipeptide.

The classic `simtk.openmm` API is *not* differentiable.  We expose a tiny
`torch.autograd.Function` that evaluates potential energy **and forces** via
OpenMM and propagates the forces back to upstream coordinates.

The implementation is purposely stripped down compared to Timewarp's
`OpenMMBridge` – we support only

    * one system (Alanine Dipeptide in vacuum),
    * positions in *nanometres* (shape `[B, 66]` or `[B, 22, 3]`),
    * no velocities, path probabilities, or multi-processing.

This keeps the dependency footprint small and avoids pulling in `bgflow`.
"""

from pathlib import Path
from functools import lru_cache
from typing import Tuple

import torch
from torch import Tensor

# OpenMM imports – we still rely on the legacy simtk namespace for now.
from simtk import openmm as mm  # type: ignore
from simtk import unit  # type: ignore
from simtk.openmm import app  # type: ignore

__all__ = [
    "compute_potential_energy",
]


# ----------------------------------------------------------------------------
# Helper – build a reusable OpenMM Context (CPU by default)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_aldp_context(platform_name: str = "Reference") -> Tuple[mm.Context, int]:
    """Return (context, n_atoms) for Alanine Dipeptide vacuum system."""
    # Build system once via OpenMMTools test system helper.
    try:
        from openmmtools import testsystems  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("openmmtools is required for Alanine Dipeptide target.") from e

    testsys = testsystems.AlanineDipeptideVacuum(constraints=None)
    system = testsys.system
    topology = testsys.topology
    positions = testsys.positions

    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions)

    return context, system.getNumParticles()


# ----------------------------------------------------------------------------
# autograd-aware potential energy (kJ/mol)
# ----------------------------------------------------------------------------
class _OpenMMPotentialEnergyFunction(torch.autograd.Function):
    """Autograd bridge for potential energy.

    Forward  : returns *potential* energy U(x) in kJ/mol.
    Backward : dU/dx = −F where F are the OpenMM forces (kJ/mol/nm).
    """

    @staticmethod
    def forward(ctx, coords_nm: Tensor) -> Tensor:  # type: ignore[override]
        # coords_nm shape: [B, 66] or [B, 22, 3]
        ctx.original_ndim = coords_nm.ndim
        if coords_nm.ndim == 3:  # [B, N, 3]
            B, N, _ = coords_nm.shape
            flat = coords_nm.view(B, -1)
        else:
            B, D = coords_nm.shape
            N = D // 3
            flat = coords_nm
        if flat.shape[1] != 66:
            raise ValueError("Expected 66-D coordinates for ALDP.")

        context, n_atoms = _get_aldp_context()
        if n_atoms * 3 != 66:
            raise RuntimeError("Unexpected atom count in ALDP system.")

        energies = torch.zeros(B, dtype=torch.double)
        forces_out = torch.zeros_like(flat, dtype=torch.double)

        for i in range(B):
            xyz = flat[i].detach().double().cpu().numpy().reshape(n_atoms, 3)
            context.setPositions(xyz * unit.nanometer)
            state = context.getState(getEnergy=True, getForces=True)
            e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            forces_quantity = state.getForces(asNumpy=True)
            f_kj_per_nm = forces_quantity.value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )  # ndarray (N,3)
            energies[i] = e_kj
            forces_out[i] = torch.tensor(f_kj_per_nm, dtype=torch.double).reshape(-1)

        # Save for backward (needs same device as input)
        ctx.save_for_backward(-forces_out.to(coords_nm.device))  # −F = dU/dx
        return energies.to(coords_nm.dtype).to(coords_nm.device)

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        deriv, = ctx.saved_tensors  # [B, 66]
        # grad_out [B] – broadcast over coords dim
        grad_coords = grad_out.unsqueeze(-1) * deriv  # [B,66]

        if ctx.original_ndim == 3:
            B = grad_coords.shape[0]
            grad_coords = grad_coords.view(B, -1, 3)
        return grad_coords


# Convenience wrapper -------------------------------------------------------------------------

def compute_potential_energy(coords: Tensor) -> Tensor:
    """Return potential energy **U(x)** (kJ/mol) with autograd support.

    Accepts `[B,66]` or `[B,22,3]` coordinates in **nanometres**.
    """
    return _OpenMMPotentialEnergyFunction.apply(coords) 