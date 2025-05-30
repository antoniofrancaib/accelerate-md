import torch
from typing import Tuple, Optional, Union
from pathlib import Path
import copy

from .base import SwapKernel
from src.accelmd.models import MODEL_REGISTRY


class TarFlowSwap(SwapKernel):
    """TarFlow-based swap kernel for parallel tempering.
    
    This uses a trained TarFlow normalizing flow to propose more sophisticated
    swaps between neighboring replicas.
    """
    
    def __init__(
        self,
        flow_checkpoint: Union[str, Path],
        model_config: Optional[dict] = None,
        device: str = 'cpu',
    ):
        """Initialize TarFlow swap kernel.
        
        Args:
            flow_checkpoint: Path to trained TarFlow model checkpoint
            model_config: Model configuration (if None, infer from checkpoint)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        self.flow_checkpoint = Path(flow_checkpoint)
        self.model_config = model_config
        self.flow = None
        self._load_flow()
    
    def _load_flow(self):
        """Load the trained TarFlow model."""
        if not self.flow_checkpoint.exists():
            raise FileNotFoundError(f"Flow checkpoint not found: {self.flow_checkpoint}")
        
        # Load checkpoint
        checkpoint = torch.load(self.flow_checkpoint, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # New format with embedded config
            state_dict = checkpoint["model_state_dict"]
            if self.model_config is None and "model_config" in checkpoint:
                self.model_config = checkpoint["model_config"]
        else:
            # Legacy format - just state dict
            state_dict = checkpoint
        
        # Extract model config if not provided
        if self.model_config is None:
            # Try to infer from checkpoint or use defaults
            self.model_config = {
                "dim": 2,  # Will be overridden when we know the actual dimension
                "channels": 512,
                "num_blocks": 8,
                "layers_per_block": 8,
                "head_dim": 64,
            }
        
        # Create model
        self.flow = MODEL_REGISTRY["tarflow"](self.model_config).to(self.device)
        
        # Load state dict
        self.flow.load_state_dict(state_dict)
        self.flow.eval()
    
    def propose(
        self,
        x_lo: torch.Tensor,
        x_hi: torch.Tensor,
        h_lo: Optional[torch.Tensor] = None,
        h_hi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propose a flow-based swap between neighboring replicas using TarFlow.
        
        The proposal uses the trained TarFlow to map configurations between
        temperature scales:
        - y_lo = flow.inverse(x_hi) (map high-T to low-T space)
        - y_hi = flow.forward(x_lo)  (map low-T to high-T space)
        
        Args:
            x_lo: Configuration(s) at lower temperature [batch_size, dim]
            x_hi: Configuration(s) at higher temperature [batch_size, dim]
            h_lo: Optional auxiliary state for low-T replica [batch_size, ...]
            h_hi: Optional auxiliary state for high-T replica [batch_size, ...]
            
        Returns:
            y_lo: Proposed low-T configuration [batch_size, dim]
            y_hi: Proposed high-T configuration [batch_size, dim]
            log_qratio: Log proposal ratio [batch_size]
        """
        if self.flow is None:
            raise RuntimeError("Flow not loaded. Call _load_flow() first.")
        
        x_lo = x_lo.to(self.device)
        x_hi = x_hi.to(self.device)
        batch_size = x_lo.shape[0]
        
        with torch.no_grad():
            # Map high-T configuration to low-T space
            y_lo, logdet_hi_to_lo = self.flow.inverse(x_hi)
            
            # Map low-T configuration to high-T space  
            y_hi, logdet_lo_to_hi = self.flow.forward(x_lo)
            
            # Compute log proposal ratio
            # For bidirectional flow: log q(y_lo,y_hi|x_lo,x_hi) / q(x_lo,x_hi|y_lo,y_hi)
            # This includes the Jacobian determinants from the flow transformations
            log_qratio = logdet_hi_to_lo + logdet_lo_to_hi
        
        return y_lo, y_hi, log_qratio
    
    def update_checkpoint(self, new_checkpoint: Union[str, Path]):
        """Update the flow checkpoint and reload the model.
        
        Args:
            new_checkpoint: Path to new checkpoint file
        """
        self.flow_checkpoint = Path(new_checkpoint)
        self._load_flow() 