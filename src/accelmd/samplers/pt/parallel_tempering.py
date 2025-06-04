import torch
import numpy as np
from typing import Callable, Union, Optional

from src.accelmd.samplers.mcmc.langevin import LangevinDynamics
from src.accelmd.kernels.local.base import LocalKernel
from src.accelmd.kernels.swap.base import SwapKernel


class ParallelTempering(LangevinDynamics):
    def __init__(self,
                 x: torch.Tensor,
                 energy_func: Callable,
                 step_size: Union[float, torch.Tensor],
                 swap_interval: int,
                 temperatures: torch.Tensor,
                 mh: bool = True,
                 device: str = 'cpu',
                 point_estimator: bool = False,
                 local_kernel: Optional[LocalKernel] = None,
                 swap_kernel: Optional[SwapKernel] = None):
        """Parallel Tempering sampler with pluggable kernel interfaces.
        
        Args:
            x: Initial configurations [num_temperatures, N, dim]
            energy_func: Energy function
            step_size: Step size for local moves
            swap_interval: Interval between swap attempts
            temperatures: Temperature schedule [num_temperatures]
            mh: Whether to use Metropolis-Hastings correction
            device: Device to run on
            point_estimator: Whether to use point estimator mode
            local_kernel: Local kernel for intra-replica moves (optional)
            swap_kernel: Swap kernel for inter-replica moves (optional)
        """
        super(ParallelTempering, self).__init__(
            x=x.reshape(-1, x.shape[-1]),
            energy_func=lambda samples: energy_func(samples) / temperatures.repeat_interleave(x.shape[1]),
            step_size=step_size,
            mh=mh,
            device=device,
            point_estimator=point_estimator
        )
        self.base_energy = energy_func
        assert len(x.shape) == 3 and len(temperatures.shape) == 1 and x.shape[0] == len(temperatures)
        self.temperatures = temperatures
        self.num_temperatures = x.shape[0]
        self.swap_rate = 0.
        self.swap_rates = []
        self.swap_interval = swap_interval
        self.counter = 0
        
        # NEW: Pluggable kernel interfaces
        self.local_kernel = local_kernel
        self.swap_kernel = swap_kernel
        self.use_new_kernels = (local_kernel is not None) and (swap_kernel is not None)
        
        if self.use_new_kernels:
            print(f"Using new kernel interfaces: local={type(local_kernel).__name__}, swap={type(swap_kernel).__name__}")
        else:
            print("Using legacy kernel implementation (fallback)")

    def sample_per_temp(self):
        """Sample within each temperature replica."""
        if self.use_new_kernels:
            return self._sample_per_temp_new()
        else:
            # Fallback to legacy implementation
            new_samples, acc = super(ParallelTempering, self).sample()
            return new_samples, acc
    
    def _sample_per_temp_new(self):
        """Sample using new LocalKernel interface."""
        self.x = self.x.reshape(self.num_temperatures, -1, self.x.shape[-1])
        all_acc_rates = []
        
        for temp_idx in range(self.num_temperatures):
            temp = self.temperatures[temp_idx]
            beta = 1.0 / temp
            x_temp = self.x[temp_idx]  # [N, dim]
            
            # Use local kernel for this temperature
            x_new, log_qratio = self.local_kernel.step(x_temp, beta, self.base_energy)
            
            # Compute acceptance probabilities
            with torch.no_grad():
                # Energy at current and proposed states
                E_old = self.base_energy(x_temp)
                E_new = self.base_energy(x_new)
                
                # Log acceptance ratio (including proposal ratio)
                log_alpha = -beta * (E_new - E_old) + log_qratio
                
                # Accept/reject
                accept_prob = torch.clamp(torch.exp(log_alpha), 0, 1)
                is_accept = torch.rand_like(accept_prob) < accept_prob
                is_accept = is_accept.unsqueeze(-1)

                # Update configurations
                self.x[temp_idx] = torch.where(is_accept, x_new, x_temp)
                all_acc_rates.append(accept_prob.mean().item())
        
        # Reshape back to flat form for compatibility
        self.x = self.x.reshape(-1, self.x.shape[-1])
        
        # Update cached energies and gradients for compatibility
        x_flat = self.x.clone().requires_grad_()
        f_x = self.energy_func(x_flat)
        grad_x = torch.autograd.grad(f_x.sum(), x_flat, create_graph=False)[0]
        self.f_x = f_x.detach()
        self.grad_x = grad_x.detach()
        
        return self.x.clone().detach(), torch.tensor(all_acc_rates).mean()
    
    def get_chain(self, chain_index):
        """Get chain data for a specific temperature index."""
        chain_samples = self.x.view(self.num_temperatures, -1, self.x.shape[-1])[chain_index].clone().detach()
        chain_energy = self.f_x.view(self.num_temperatures, -1)[chain_index].clone().detach()
        chain_scores = self.grad_x.view(self.num_temperatures, -1, self.x.shape[-1])[chain_index].clone().detach()
        return chain_samples, chain_energy, chain_scores
    
    def add_chain(self, chain_index, samples, energies, scores):
        """Add chain data for a specific temperature index."""
        self.x.view(self.num_temperatures, -1, self.x.shape[-1])[chain_index] = samples.clone().detach()
        self.f_x.view(self.num_temperatures, -1)[chain_index] = energies.clone().detach()
        self.grad_x.view(self.num_temperatures, -1, self.x.shape[-1])[chain_index] = scores.clone().detach()
    
    def attempt_swap(self, chain_a_index, chain_b_index):
        """Attempt swap between two neighboring temperature chains."""
        if self.use_new_kernels:
            return self._attempt_swap_new(chain_a_index, chain_b_index)
        else:
            # Fallback to legacy implementation
            return self._attempt_swap_legacy(chain_a_index, chain_b_index)
    
    def _attempt_swap_new(self, chain_a_index, chain_b_index):
        """Attempt swap using new SwapKernel interface."""
        temp_a, temp_b = self.temperatures[chain_a_index], self.temperatures[chain_b_index]
        beta_a, beta_b = 1.0 / temp_a, 1.0 / temp_b
        
        # Get current configurations
        chain_a, chain_a_energy, chain_a_score = self.get_chain(chain_a_index)
        chain_b, chain_b_energy, chain_b_score = self.get_chain(chain_b_index)
        
        # Ensure proper ordering (lo < hi temperature)
        if temp_a < temp_b:
            x_lo, x_hi = chain_a, chain_b
            h_lo, h_hi = chain_a_energy, chain_b_energy
            lo_idx, hi_idx = chain_a_index, chain_b_index
            beta_lo, beta_hi = beta_a, beta_b
        else:
            x_lo, x_hi = chain_b, chain_a
            h_lo, h_hi = chain_b_energy, chain_a_energy
            lo_idx, hi_idx = chain_b_index, chain_a_index
            beta_lo, beta_hi = beta_b, beta_a
        
        # Use swap kernel to propose new configurations
        y_lo, y_hi, log_qratio = self.swap_kernel.propose(x_lo, x_hi, h_lo, h_hi)
        
        # Compute energies of proposed configurations
        with torch.no_grad():
            E_y_lo = self.base_energy(y_lo)
            E_y_hi = self.base_energy(y_hi)
            E_x_lo = self.base_energy(x_lo)
            E_x_hi = self.base_energy(x_hi)
            
            # Log acceptance ratio for the swap
            # Detailed balance: α = min(1, exp(ΔE + log_qratio))
            # where ΔE = (β_lo - β_hi) * (E_y_hi - E_y_lo - E_x_hi + E_x_lo)
            delta_E = (beta_lo - beta_hi) * (E_y_hi - E_y_lo - E_x_hi + E_x_lo)
            log_alpha = delta_E + log_qratio
            
            # Accept/reject swap
            accept_prob = torch.clamp(torch.exp(log_alpha), 0, 1)
            is_accept = torch.rand_like(accept_prob) < accept_prob
            is_accept = is_accept.unsqueeze(-1)
            
            # Update configurations if accepted
            if temp_a < temp_b:  # chain_a is low-T, chain_b is high-T
                new_chain_a = torch.where(is_accept, y_lo, chain_a)
                new_chain_b = torch.where(is_accept, y_hi, chain_b)
                new_energy_a = torch.where(is_accept.squeeze(-1), E_y_lo, chain_a_energy)
                new_energy_b = torch.where(is_accept.squeeze(-1), E_y_hi, chain_b_energy)
            else:  # chain_b is low-T, chain_a is high-T  
                new_chain_a = torch.where(is_accept, y_hi, chain_a)
                new_chain_b = torch.where(is_accept, y_lo, chain_b)
                new_energy_a = torch.where(is_accept.squeeze(-1), E_y_hi, chain_a_energy)
                new_energy_b = torch.where(is_accept.squeeze(-1), E_y_lo, chain_b_energy)
            
            # For gradients, use simpler approach similar to legacy implementation
            # Scale existing gradients by temperature ratio if swap is accepted
            if temp_a < temp_b:  # chain_a is low-T, chain_b is high-T
                new_chain_a_score = torch.where(is_accept, chain_b_score * temp_b / temp_a, chain_a_score)
                new_chain_b_score = torch.where(is_accept, chain_a_score * temp_a / temp_b, chain_b_score)
            else:  # chain_b is low-T, chain_a is high-T  
                new_chain_a_score = torch.where(is_accept, chain_b_score * temp_b / temp_a, chain_a_score)
                new_chain_b_score = torch.where(is_accept, chain_a_score * temp_a / temp_b, chain_b_score)
            
            # Update chain data
            self.add_chain(chain_a_index, new_chain_a, new_energy_a / temp_a, new_chain_a_score)
            self.add_chain(chain_b_index, new_chain_b, new_energy_b / temp_b, new_chain_b_score)
            
            return accept_prob.mean().item()
    
    def _attempt_swap_legacy(self, chain_a_index, chain_b_index):
        """Legacy swap implementation (original code)."""
        temp_a, temp_b = self.temperatures[chain_a_index], self.temperatures[chain_b_index]
        chain_a, chain_a_energy, chain_a_score = self.get_chain(chain_a_index)
        chain_b, chain_b_energy, chain_b_score = self.get_chain(chain_b_index)
        log_prob_a = -chain_a_energy * temp_a  # this is untempered log prob
        log_prob_b = -chain_b_energy * temp_b  # this is untempered log prob
        
        log_acceptance_ratio = (1./temp_a - 1./temp_b) * (log_prob_b - log_prob_a)
        is_accept = torch.rand_like(log_acceptance_ratio, device=self.device).log() < log_acceptance_ratio
        is_accept = is_accept.unsqueeze(-1)

        new_chain_a = torch.where(is_accept, chain_b.detach().clone(), chain_a.detach().clone())
        new_chain_b = torch.where(is_accept, chain_a.detach().clone(), chain_b.detach().clone())

        new_chain_a_score = torch.where(is_accept, chain_b_score.detach().clone() * temp_b / temp_a, chain_a_score.detach().clone())
        new_chain_b_score = torch.where(is_accept, chain_a_score.detach().clone() * temp_a / temp_b, chain_b_score.detach().clone())

        new_chain_a_energy = torch.where(is_accept.squeeze(-1), chain_b_energy.detach().clone() * temp_b / temp_a, chain_a_energy.detach().clone())
        new_chain_b_energy = torch.where(is_accept.squeeze(-1), chain_a_energy.detach().clone() * temp_a / temp_b, chain_b_energy.detach().clone())

        self.add_chain(chain_a_index, new_chain_a, new_chain_a_energy, new_chain_a_score)
        self.add_chain(chain_b_index, new_chain_b, new_chain_b_energy, new_chain_b_score)

        return is_accept.float().mean().item()

    def swap_samples(self):
        """Perform swaps between neighboring temperature chains."""
        swap_rates = []
        for i in range(self.num_temperatures - 1, 0, -1):
            swap_rate = self.attempt_swap(i, i - 1)
            swap_rates.append(swap_rate)
        self.swap_rate = np.mean(swap_rates)
        self.swap_rates = swap_rates
    
    def sample(self):
        """Perform one sampling step (local moves + periodic swaps)."""
        _, acc = self.sample_per_temp()
        self.counter += 1
        if self.counter % self.swap_interval == 0:
            self.swap_samples()
        return self.x.clone().detach().reshape(self.num_temperatures, -1, self.x.shape[-1]), acc