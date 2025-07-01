from __future__ import annotations

"""Generic Cartesian Boltzmann target for any small peptide given a PDB + OpenMM force field.

This mirrors :class:`AldpBoltzmann` but does *not* hard-code any residue
layout.  The class is CPU-only for the OpenMM context but returns PyTorch
log-densities (−βU) on whichever device the caller specifies.

Example
-------
>>> target = build_target(
...     'dipeptide',
...     pdb_path='data/timewarp/2AA-1-big/train/AA-traj-state0.pdb',
...     forcefield_files=['amber14-all.xml', 'amber14/tip3p.xml'],
...     temperature=300.0,
... )
>>> x = torch.randn(4, target.dim) * 0.1  # nm
>>> logp = target.log_prob(x)
"""

from pathlib import Path
from typing import List

import torch
from torch import Tensor

# OpenMM stack ----------------------------------------------------------------
from simtk import openmm as mm  # type: ignore
from simtk import unit  # type: ignore
from simtk.openmm import app  # type: ignore

# BoltzGen provides autograd-aware Boltzmann distributions
import boltzgen as bg  # type: ignore

from . import register_target

__all__ = ["DipeptidePotentialCart"]

# Boltzmann constant (kJ/mol/K)
K_B_KJMOL = 0.00831446261815324


@register_target("dipeptide")
class DipeptidePotentialCart:
    """Cartesian Boltzmann distribution for an arbitrary peptide provided as PDB."""

    def __init__(
        self,
        pdb_path: str | Path,
        forcefield_files: List[str] | None = None,
        *,
        temperature: float = 300.0,
        env: str = "vacuum",
        energy_cut: float | None = 1e8,
        energy_max: float | None = 1e20,
        n_threads: int = 1,
        device: str | torch.device = "cpu",
        platform_name: str = "Reference",
    ) -> None:
        self.device = torch.device(device)

        pdb_path = Path(pdb_path)
        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)

        # ------------------------------------------------------------------
        # 1) Build OpenMM System from PDB + force field
        # ------------------------------------------------------------------
        pdb = app.PDBFile(str(pdb_path))

        if forcefield_files is None:
            # Sensible defaults per environment
            if env.lower() == "implicit":
                # General protein parameters + OBC implicit solvent parameters
                forcefield_files = ["amber14-all.xml", "implicit/obc2.xml"]
            else:  # vacuum or explicit (TIP3P) – we only need the base protein FF
                forcefield_files = ["amber14-all.xml"]
        ff = app.ForceField(*forcefield_files)

        env = env.lower()
        if env == "vacuum":
            system = ff.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=None,
            )
        elif env == "implicit":
            system = ff.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=None,
            )
        else:
            raise ValueError("env must be 'vacuum' or 'implicit'.")

        # ------------------------------------------------------------------
        # 2) Create a single OpenMM Context (Reference platform) for energies
        # ------------------------------------------------------------------
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
        except Exception:
            platform = mm.Platform.getPlatformByName("Reference")  # graceful fallback
        self.context = mm.Context(system, integrator, platform)
        # Set initial positions
        self.context.setPositions(pdb.positions)
        # Energy minimisation – provides a low-energy reference frame
        mm.LocalEnergyMinimizer.minimize(self.context)

        # ------------------------------------------------------------------
        # 3) Autograd-aware Boltzmann distribution via BoltzGen
        # ------------------------------------------------------------------
        self.temperature = float(temperature)
        self.beta = 1.0 / (K_B_KJMOL * self.temperature)

        if n_threads > 1:
            self.dist = bg.distributions.BoltzmannParallel(
                system,
                self.temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                n_threads=n_threads,
            )
        else:
            self.dist = bg.distributions.Boltzmann(
                self.context,
                self.temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
            )

        # Basic shape info --------------------------------------------------
        self.n_atoms = pdb.topology.getNumAtoms()
        self.dim = self.n_atoms * 3
        self.topology = pdb.topology

    # ------------------------------------------------------------------
    # Public API compatible with AldpBoltzmann (autograd-enabled)
    # ------------------------------------------------------------------
    def log_prob(self, coords: Tensor) -> Tensor:  # −βU
        orig_shape = coords.shape[:-1]
        flat = coords.view(-1, self.dim)

        # BoltzGen operates on CPU tensors; gradients are propagated through
        # a custom autograd.Function, so moving to CPU is safe.
        flat_cpu = flat.to("cpu")
        logp_cpu = self.dist.log_prob(flat_cpu)  # −βU
        return logp_cpu.to(coords.device).view(*orig_shape)

    __call__ = log_prob

    # Convenience – potential energy (kJ/mol)
    def potential_energy(self, coords: Tensor) -> Tensor:
        logp = self.log_prob(coords)  # −βU
        return (-logp) / self.beta 