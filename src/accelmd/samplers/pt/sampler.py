from __future__ import annotations

"""Generic MCMC samplers and Parallel Tempering implementation.

This file is largely a direct copy of the legacy `main/samplers/sampler.py`
with import paths adjusted for the new `src.accelmd` package layout and some
light typing / linting fixes.
"""

from typing import Callable, Optional, Tuple, Union, List

import copy
import numpy as np
import torch
from torch.distributions import Normal  # noqa: F401  # kept for future extensions

__all__ = [
    "MCMCSampler",
    "LangevinDynamics",
    "ParallelTempering",
]

Tensor = torch.Tensor


class MCMCSampler:
    """Abstract base class for (potentially batched) MCMC samplers."""

    def __init__(
        self,
        x: Tensor,
        energy_func: Callable[[Tensor], Tensor],
        step_size: Union[float, Tensor],
        *,
        mh: bool = True,
        device: str | torch.device = "cpu",
        point_estimator: bool = False,
    ) -> None:
        self.x: Tensor = x.to(device)
        self.step_size: Union[float, Tensor] = step_size if isinstance(step_size, float) else step_size.to(device)
        self.energy_func = energy_func
        self.mh = mh
        self.device = torch.device(device)
        self.point_estimator = point_estimator

    # ------------------------------------------------------------------
    def sample(self) -> Tuple[Tensor, Optional[Tensor]]:  # pragma: no cover
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Langevin Dynamics with optional Metropolis-Hastings correction
# -----------------------------------------------------------------------------


class LangevinDynamics(MCMCSampler):
    """Overdamped Langevin dynamics sampler (with optional MH correction)."""

    def __init__(
        self,
        x: Tensor,
        energy_func: Callable[[Tensor], Tensor],
        step_size: Union[float, Tensor],
        *,
        mh: bool = True,
        device: str | torch.device = "cpu",
        point_estimator: bool = False,
    ) -> None:
        super().__init__(x, energy_func, step_size, mh=mh, device=device, point_estimator=point_estimator)

        # Lazily initialised state for MH path. If energy_func delivers
        # non-differentiable outputs (no grad_fn) we will switch to a
        # gradient-free random-walk proposal inside `_single_step`.
        self.f_x: Optional[Tensor] = None
        self.grad_x: Optional[Tensor] = None
        if self.mh:
            x_c = self.x.detach().requires_grad_(True)
            f_xc = self.energy_func(x_c)
            if f_xc.requires_grad:
                grad_xc = torch.autograd.grad(f_xc.sum(), x_c, create_graph=False)[0]
                self.f_x = f_xc.detach()
                self.grad_x = grad_xc.detach()
            else:
                # Energy not differentiable -> fall back to non-MH random walk
                self.mh = False  # disable MH path
                self.f_x = None
                self.grad_x = None

    # ------------------------------------------------------------------
    def _single_step(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Perform one Langevin proposal (+ optional MH accept/reject)."""
        if self.point_estimator:
            x_c = self.x.detach().requires_grad_(True)
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c)[0]
            x_p = x_c - self.step_size * grad_xc
            self.x = x_p.detach()
            return self.x.clone(), None

        if not self.mh:
            # Gradient may or may not be available; if not, use pure Gaussian RW.
            x_c = self.x.detach().requires_grad_(True)
            f_xc = self.energy_func(x_c)
            if f_xc.requires_grad:
                grad_xc = torch.autograd.grad(f_xc.sum(), x_c)[0]
            else:
                grad_xc = torch.zeros_like(x_c)
            noise = torch.randn_like(x_c, device=self.device)
            x_p = x_c - self.step_size * grad_xc + torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device)) * noise
            self.x = x_p.detach()
            return self.x.clone(), f_xc.detach()

        # MH-corrected path ------------------------------------------------
        x_c = self.x.detach()
        f_xc = self.f_x.detach()
        grad_xc = self.grad_x.detach()

        noise = torch.randn_like(x_c, device=self.device)
        x_p = x_c - self.step_size * grad_xc + torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device)) * noise

        x_p = x_p.detach().requires_grad_(True)
        f_xp = self.energy_func(x_p)
        grad_xp = torch.autograd.grad(f_xp.sum(), x_p)[0]

        if isinstance(self.step_size, float):
            denom = 4 * self.step_size
        else:
            denom = 4 * self.step_size.squeeze(-1)

        log_joint_2 = -f_xc - torch.norm(x_p - x_c + self.step_size * grad_xc, dim=-1) ** 2 / denom
        log_joint_1 = -f_xp - torch.norm(x_c - x_p + self.step_size * grad_xp, dim=-1) ** 2 / denom

        log_accept = log_joint_1 - log_joint_2
        accept_prob = torch.minimum(torch.ones_like(log_accept), log_accept.exp())
        accept = (torch.rand_like(log_accept) <= accept_prob).unsqueeze(-1)

        self.x = torch.where(accept, x_p.detach(), self.x)
        self.f_x = torch.where(accept.squeeze(-1), f_xp.detach(), self.f_x)
        self.grad_x = torch.where(accept, grad_xp.detach(), self.grad_x)

        return self.x.clone(), accept_prob.detach()

    # ------------------------------------------------------------------
    def sample(self):  # type: ignore[override]
        return self._single_step()


# -----------------------------------------------------------------------------
# Parallel Tempering (Replica Exchange) with per-temperature batches
# -----------------------------------------------------------------------------


class ParallelTempering(LangevinDynamics):
    """Replica‐exchange Langevin sampler.

    Notes
    -----
    *Each replica may run multiple chains*; the input tensor `x` therefore has
    shape `[n_temp, n_chains, dim]`.
    """

    def __init__(
        self,
        x: Tensor,  # [n_temp, n_chains, dim]
        energy_func: Callable[[Tensor], Tensor],
        step_size: Union[float, Tensor],
        *,
        swap_interval: int,
        temperatures: Tensor,  # [n_temp]
        mh: bool = True,
        device: str | torch.device = "cpu",
        point_estimator: bool = False,
        log_history: bool = False,
    ) -> None:
        n_temp, n_chains, dim = x.shape
        self.num_temperatures = n_temp
        self.swap_interval = int(swap_interval)
        self.counter = 0
        self.temperatures = temperatures.to(device)

        # Flatten replicas & chains for parent class
        super().__init__(
            x=x.reshape(-1, dim).to(device),
            energy_func=lambda y: energy_func(y) / self.temperatures.repeat_interleave(n_chains),
            step_size=step_size,
            mh=mh,
            device=device,
            point_estimator=point_estimator,
        )

        self.base_energy = energy_func  # unscaled
        self.swap_rate = 0.0
        self.swap_rates: List[float] = []
        
        # History logging for round-trip analysis
        self.log_history = log_history
        self.history: List[Tensor] = [] if log_history else []
        
        # Track which walker is at which temperature (for round-trip analysis)
        if log_history:
            # Initially, walker i is at temperature i (identity mapping)
            # temp_assignment[walker_id] = temperature_index
            total_walkers = n_temp * n_chains
            self.temp_assignment = torch.arange(n_temp).repeat_interleave(n_chains).to(device)
            self.history.append(self.temp_assignment.clone())

    # ------------------------------------------------------------------
    # Replica-exchange utilities
    # ------------------------------------------------------------------
    def _attempt_swap(self, idx_a: int, idx_b: int) -> float:
        temp_a, temp_b = self.temperatures[idx_a], self.temperatures[idx_b]

        chains_per_temp = self.x.shape[0] // self.num_temperatures
        slice_a = slice(idx_a * chains_per_temp, (idx_a + 1) * chains_per_temp)
        slice_b = slice(idx_b * chains_per_temp, (idx_b + 1) * chains_per_temp)

        chain_a = self.x[slice_a]
        chain_b = self.x[slice_b]

        energy_a = self.base_energy(chain_a)
        energy_b = self.base_energy(chain_b)

        log_accept = (1.0 / temp_a - 1.0 / temp_b) * (energy_b - energy_a)
        accept_prob = torch.minimum(torch.ones_like(log_accept), log_accept.exp())
        accept = (torch.rand_like(log_accept) <= accept_prob).unsqueeze(-1)

        # Swap chains in-place where accepted
        self.x[slice_a] = torch.where(accept, chain_b, chain_a)
        self.x[slice_b] = torch.where(accept, chain_a, chain_b)
        
        # Update temperature assignments for history tracking
        if self.log_history:
            # For accepted swaps, exchange the temperature assignments
            accept_flat = accept.squeeze(-1)  # [n_chains]
            temp_a_old = self.temp_assignment[slice_a].clone()
            temp_b_old = self.temp_assignment[slice_b].clone()
            self.temp_assignment[slice_a] = torch.where(accept_flat, temp_b_old, temp_a_old)
            self.temp_assignment[slice_b] = torch.where(accept_flat, temp_a_old, temp_b_old)

        return accept_prob.mean().item()

    def _swap_replicas(self):
        rates = []
        for i in range(self.num_temperatures - 1, 0, -1):
            rates.append(self._attempt_swap(i, i - 1))
        self.swap_rate = float(np.mean(rates)) if rates else 0.0
        self.swap_rates = rates
        
        # Record current temperature assignments for round-trip analysis
        if self.log_history:
            self.history.append(self.temp_assignment.clone())

    # ------------------------------------------------------------------
    def sample(self):  # type: ignore[override]
        samples, acc = self._single_step()
        self.counter += 1
        if self.counter % self.swap_interval == 0:
            self._swap_replicas()
        # reshape back to [temp, chains, dim] for caller
        chains_per_temp = samples.shape[0] // self.num_temperatures
        samples = samples.view(self.num_temperatures, chains_per_temp, -1)
        return samples, acc 