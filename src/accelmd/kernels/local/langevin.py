import torch
import copy
from typing import Tuple, Callable, Union

from .base import LocalKernel


class Langevin(LocalKernel):
    """Langevin dynamics local kernel for MCMC sampling.
    
    This wraps the existing LangevinDynamics implementation into the
    LocalKernel interface for use in parallel tempering.
    """
    
    def __init__(
        self, 
        step_size: Union[float, torch.Tensor] = 1e-4,
        mh: bool = True,
        device: str = 'cpu',
    ):
        """Initialize Langevin kernel.
        
        Args:
            step_size: Step size for Langevin dynamics
            mh: Whether to use Metropolis-Hastings correction
            device: Device to run on ('cpu' or 'cuda')
        """
        self.step_size = step_size
        self.mh = mh
        self.device = device
        
        # Cache for gradient and energy computations
        self._x_cache = None
        self._f_x_cache = None
        self._grad_x_cache = None
    
    def step(
        self,
        x: torch.Tensor,
        beta: float,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one Langevin dynamics step.
        
        Args:
            x: Current configuration [batch_size, dim]
            beta: Inverse temperature (1/T)
            energy_fn: Energy function
            
        Returns:
            x_new: New configuration [batch_size, dim]
            log_qratio: Log proposal ratio [batch_size] (0 for symmetric proposals)
        """
        x = x.to(self.device)
        batch_size = x.shape[0]
        
        if not self.mh:
            # Pure Langevin (no MH correction)
            x_c = x.detach()
            x_c.requires_grad = True
            f_xc = energy_fn(x_c) * beta  # Scale by inverse temperature
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c, create_graph=False)[0]
            
            # Langevin update: x' = x - ε∇U + √(2ε)ξ
            noise = torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device)) * torch.randn_like(x_c)
            x_new = x_c - self.step_size * grad_xc + noise
            
            # Symmetric proposal, so log_qratio = 0
            log_qratio = torch.zeros(batch_size, device=self.device)
            
            return x_new.detach(), log_qratio
        
        else:
            # Metropolis-adjusted Langevin algorithm (MALA)
            x_c = x.detach()
            
            # Current energy and gradient
            x_c.requires_grad = True
            f_xc = energy_fn(x_c) * beta
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c, create_graph=False)[0]
            
            # Propose new state
            noise = torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device)) * torch.randn_like(x_c)
            x_p = x_c - self.step_size * grad_xc + noise
            
            # Energy and gradient at proposed state
            x_p = x_p.detach()
            x_p.requires_grad = True
            f_xp = energy_fn(x_p) * beta
            grad_xp = torch.autograd.grad(f_xp.sum(), x_p, create_graph=False)[0]
            
            # Compute MH acceptance probability
            if isinstance(self.step_size, float):
                # Proposal log-probabilities (reverse moves)
                log_q_forward = -torch.norm(x_p - x_c + self.step_size * grad_xc, dim=-1)**2 / (4 * self.step_size)
                log_q_backward = -torch.norm(x_c - x_p + self.step_size * grad_xp, dim=-1)**2 / (4 * self.step_size)
            else:
                # Handle tensor step sizes
                log_q_forward = -torch.norm(x_p - x_c + self.step_size * grad_xc, dim=-1)**2 / (4 * self.step_size.squeeze(-1))
                log_q_backward = -torch.norm(x_c - x_p + self.step_size * grad_xp, dim=-1)**2 / (4 * self.step_size.squeeze(-1))
            
            # Log acceptance ratio: log α = log p(x') + log q(x'→x) - log p(x) - log q(x→x')
            log_accept_ratio = -f_xp + log_q_backward + f_xc - log_q_forward
            
            # Accept/reject
            is_accept = torch.rand_like(log_accept_ratio).log() <= log_accept_ratio
            is_accept = is_accept.unsqueeze(-1)
            
            x_new = torch.where(is_accept, x_p.detach(), x_c.detach())
            
            # For detailed balance, log_qratio should be 0 for accepted moves
            # (since MH correction accounts for asymmetry)
            log_qratio = torch.zeros(batch_size, device=self.device)
            
            return x_new, log_qratio 