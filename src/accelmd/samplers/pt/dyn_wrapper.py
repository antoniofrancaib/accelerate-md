from __future__ import annotations

"""Dynamic step-size adaptation wrapper for MCMC / PT samplers.

Ported from `main/samplers/dyn_mcmc_warp.py` and slimmed to essentials.
"""

from typing import Optional

import torch

from .sampler import MCMCSampler

__all__ = ["DynSamplerWrapper"]


class DynSamplerWrapper:
    """Wrap an :class:`MCMCSampler` and adapt its step size on-the-fly."""

    def __init__(
        self,
        sampler: MCMCSampler,
        *,
        per_temp: bool = False,
        total_n_temp: Optional[int] = None,
        target_acceptance_rate: float = 0.6,
        alpha: float = 0.25,
    ) -> None:
        self.sampler = sampler
        self.target_acceptance_rate = float(target_acceptance_rate)
        self.alpha = float(alpha)
        self.per_temp = per_temp
        self.total_n_temp = total_n_temp
        if per_temp and total_n_temp is None:
            raise ValueError("total_n_temp must be given when per_temp=True")

    # ------------------------------------------------------------------
    def sample(self):
        """Return `(samples, acc_metric, step_size)` just like the legacy code."""
        new_samples, acc = self.sampler.sample()

        if acc is None:
            return new_samples, torch.tensor(float("nan")), self.sampler.step_size

        if self.per_temp:
            assert isinstance(self.sampler.step_size, torch.Tensor)
            acc_temp = acc.view(self.total_n_temp, -1).mean(dim=1)
            cur_step = self.sampler.step_size.squeeze(-1).view(self.total_n_temp, -1).mean(dim=1)
            # Increase / decrease per temperature
            mask_up = acc_temp > self.target_acceptance_rate
            mask_dn = ~mask_up
            cur_step[mask_up] *= (1 + self.alpha)
            cur_step[mask_dn] *= (1 - self.alpha)
            # broadcast back to all chains
            self.sampler.step_size = cur_step.repeat_interleave(self.sampler.step_size.shape[0] // self.total_n_temp).unsqueeze(-1)
            return new_samples, acc_temp, self.sampler.step_size

        # Global (temperature-agnostic) version ---------------------------------
        acc_scalar = acc.mean().item()
        if acc_scalar > self.target_acceptance_rate:
            self.sampler.step_size *= (1 + self.alpha)
        else:
            self.sampler.step_size *= (1 - self.alpha)
        return new_samples, torch.tensor(acc_scalar), self.sampler.step_size 