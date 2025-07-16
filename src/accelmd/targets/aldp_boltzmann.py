# NOTE: This file fully replaces the prior skeleton.
"""ALDP Boltzmann target distribution (✅ initial functional version).

Computes the un-normalised Boltzmann log-probability of an Alanine Dipeptide
(ALDP) configuration in vacuo using OpenMM.  The implementation is intentionally
simple and CPU-only for now – we create one `openmm.Context` and reuse it for
all energy evaluations.

The public API matches the expectations of the training loop:
    >>> target = AldpBoltzmann(temperature=300.0)
    >>> coords = torch.randn(4, 66) * 0.1  # nm, centre-ed around origin
    >>> logp = target.log_prob(coords)
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

# OpenMM stack
from simtk import openmm as mm  # type: ignore
from simtk import unit  # type: ignore
from simtk.openmm import app  # type: ignore
from openmmtools import testsystems  # type: ignore

from . import register_target

__all__ = ["AldpBoltzmann"]

# Boltzmann constant in kJ/(mol*K)
K_B_KJMOL = 0.00831446261815324


@register_target("aldp")
class AldpBoltzmann:
    """Alanine Dipeptide Boltzmann distribution in vacuum (Cartesian coordinates)."""

    def __init__(
        self,
        temperature: float = 300.0,
        device: str = "cpu",
        energy_cut: float | None = None,
        energy_max: float | None = None,
        **kwargs  # Accept and ignore extra parameters for compatibility
    ) -> None:
        self.T = float(temperature)
        self.beta = 1.0 / (K_B_KJMOL * self.T)  # dimensionless 1/(k_B T)
        self.device = torch.device(device)

        # ------------------------------------------------------------------
        # Build OpenMM system once and keep a reusable Context.
        # ------------------------------------------------------------------
        testsys = testsystems.AlanineDipeptideVacuum(constraints=None)
        system = testsys.system
        topology = testsys.topology
        
        # Get number of atoms dynamically from the system
        self.n_atoms = system.getNumParticles()
        self.dim = self.n_atoms * 3

        # Use a dummy integrator – we only need energies.
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)

        # Use the constantly available Reference platform for portability.
        platform = mm.Platform.getPlatformByName("Reference")
        self.context = mm.Context(system, integrator, platform)

        # We will allocate a fresh `Quantity` per sample; this avoids tricky
        # in-place unit conversions that OpenMM dislikes when mixing raw
        # NumPy arrays with `unit.Quantity` objects.

        # Optional energy regularisation parameters (mimic BoltzGen)
        self.energy_cut = energy_cut  # kJ/mol where regularisation starts
        self.energy_max = energy_max  # clamp threshold

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_prob(self, coords: Tensor) -> Tensor:
        """Return un-normalised log-probability `−β U(x)`.

        Parameters
        ----------
        coords
            Tensor of shape `[..., 66]` holding Cartesian coordinates **in
            nanometres**.  If unsure about units, inspect the dataset (see
            `.cursor/rules/pt_swap_workflow.mdc`).
        """
        # ------------------------------------------------------------------
        # Differentiable potential energy via `compute_potential_energy`.
        # ------------------------------------------------------------------
        # The autograd bridge returns forces in the backward pass so gradients
        # flow through the Boltzmann log-density – crucial for an NLL loss.

        from ..training.openmm_bridge import compute_potential_energy  # local import to avoid heavy dependency at module level

        coords_flat = coords.reshape(-1, self.dim)  # keep gradient tracking & device
        energies = compute_potential_energy(coords_flat).to(torch.double)  # kJ/mol, differentiable

        # Optional energy regularisation (identical to previous behaviour)
        if self.energy_max is not None:
            energies = torch.minimum(energies, torch.tensor(self.energy_max, dtype=energies.dtype, device=energies.device))

        if self.energy_cut is not None:
            mask = energies > self.energy_cut
            if mask.any():
                diff = energies[mask] - self.energy_cut
                energies = energies.clone()  # avoid in-place op that breaks autograd for shared storage
                energies[mask] = self.energy_cut + diff.log1p()

        log_p = -self.beta * energies  # −βU

        return log_p.reshape(coords.shape[:-1])

    # Alias
    __call__ = log_prob

    # ------------------------------------------------------------------
    # Misc helpers (optional)
    # ------------------------------------------------------------------
    def to(self, device: str | torch.device):
        """Interface shim so callers can do `target.to('cuda')` even though the
        heavy OpenMM context remains on CPU. Only tensor attributes are moved.
        """
        self.device = torch.device(device)
        return self 