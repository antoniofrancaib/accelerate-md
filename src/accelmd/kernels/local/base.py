from abc import ABC, abstractmethod
from typing import Tuple, Callable
import torch


class LocalKernel(ABC):
    """Abstract base class for intra-replica proposal kernels.
    
    A LocalKernel operates on a single replica at a specific temperature,
    proposing new configurations within that replica.
    """
    
    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        beta: float,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propose a new configuration for a single replica.
        
        Args:
            x: Current configuration(s) [batch_size, dim]
            beta: Inverse temperature (1/T)
            energy_fn: Function that computes energy given configuration
            
        Returns:
            x_new: Proposed configuration(s) [batch_size, dim]
            log_qratio: Log proposal ratio log(q(x'|x)/q(x|x')) [batch_size]
                       (for symmetric proposals this is typically 0)
        """
        pass 