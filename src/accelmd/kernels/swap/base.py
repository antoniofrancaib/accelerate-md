from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch


class SwapKernel(ABC):
    """Abstract base class for inter-replica swap proposal kernels.
    
    A SwapKernel proposes swaps between neighboring replicas at different 
    temperatures in parallel tempering.
    """
    
    @abstractmethod
    def propose(
        self,
        x_lo: torch.Tensor,
        x_hi: torch.Tensor,
        h_lo: Optional[torch.Tensor] = None,
        h_hi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propose a swap between two neighboring replicas.
        
        Args:
            x_lo: Configuration(s) at lower temperature [batch_size, dim]
            x_hi: Configuration(s) at higher temperature [batch_size, dim]
            h_lo: Optional auxiliary state for low-T replica [batch_size, ...]
            h_hi: Optional auxiliary state for high-T replica [batch_size, ...]
            
        Returns:
            y_lo: Proposed configuration for low-T replica [batch_size, dim]
            y_hi: Proposed configuration for high-T replica [batch_size, dim]
            log_qratio: Log proposal ratio log(q(y_lo,y_hi|x_lo,x_hi)/q(x_lo,x_hi|y_lo,y_hi)) [batch_size]
        """
        pass 