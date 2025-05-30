import torch
from typing import Tuple, Optional

from .base import SwapKernel


class VanillaSwap(SwapKernel):
    """Standard parallel tempering swap kernel.
    
    This implements the classic PT swap proposal where configurations
    are simply exchanged between neighboring replicas.
    """
    
    def __init__(self):
        """Initialize vanilla swap kernel."""
        pass
    
    def propose(
        self,
        x_lo: torch.Tensor,
        x_hi: torch.Tensor,
        h_lo: Optional[torch.Tensor] = None,
        h_hi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propose a standard swap between neighboring replicas.
        
        In vanilla PT, the proposal is simply to exchange configurations:
        y_lo = x_hi, y_hi = x_lo
        
        Args:
            x_lo: Configuration(s) at lower temperature [batch_size, dim]
            x_hi: Configuration(s) at higher temperature [batch_size, dim]
            h_lo: Ignored (auxiliary state not used in vanilla swap)
            h_hi: Ignored (auxiliary state not used in vanilla swap)
            
        Returns:
            y_lo: Proposed low-T configuration (= x_hi) [batch_size, dim]
            y_hi: Proposed high-T configuration (= x_lo) [batch_size, dim]
            log_qratio: Log proposal ratio (= 0 for symmetric proposal) [batch_size]
        """
        batch_size = x_lo.shape[0]
        device = x_lo.device
        
        # Standard swap: exchange configurations
        y_lo = x_hi.clone()
        y_hi = x_lo.clone()
        
        # Symmetric proposal, so log_qratio = 0
        log_qratio = torch.zeros(batch_size, device=device)
        
        return y_lo, y_hi, log_qratio 