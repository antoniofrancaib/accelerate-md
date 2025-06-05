import torch
from typing import Tuple, Optional, Union
from pathlib import Path
import copy

from .base import SwapKernel
from src.accelmd.models import MODEL_REGISTRY


class RealNVPSwap(SwapKernel):
    """RealNVP flow-based swap kernel for parallel tempering.
    
    This uses a trained RealNVP normalizing flow to propose more sophisticated
    swaps between neighboring replicas.
    """
    
    def __init__(
        self,
        flow_checkpoint: Union[str, Path],
        model_config: Optional[dict] = None,
        device: str = 'cpu',
    ):
        """Initialize RealNVP swap kernel.
        
        Args:
            flow_checkpoint: Path to trained RealNVP model checkpoint
            model_config: Model configuration (if None, infer from checkpoint)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        self.flow_checkpoint = Path(flow_checkpoint)
        self.model_config = model_config
        self.flow = None
        self._load_flow()
    
    def _load_flow(self):
        """Load the trained RealNVP flow model."""
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
            # Infer from checkpoint state dict
            self.model_config = self._infer_model_config(state_dict)
        
        # Create model
        self.flow = MODEL_REGISTRY["realnvp"](self.model_config).to(self.device)
        
        # Load state dict
        self.flow.load_state_dict(state_dict)
        self.flow.eval()
    
    def _infer_model_config(self, state_dict: dict) -> dict:
        """Infer model configuration from checkpoint state dict.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Inferred model configuration
        """
        # Find the highest coupling layer index to determine n_couplings
        max_coupling_idx = -1
        for key in state_dict.keys():
            if key.startswith("couplings.") and ".mask" in key:
                coupling_idx = int(key.split(".")[1])
                max_coupling_idx = max(max_coupling_idx, coupling_idx)
        n_couplings = max_coupling_idx + 1
        
        # Infer dimension from mask size
        mask_key = "couplings.0.mask"
        if mask_key in state_dict:
            dim = state_dict[mask_key].shape[0]
        else:
            raise ValueError("Cannot infer dimension: couplings.0.mask not found in checkpoint")
        
        # Infer hidden dimension from first layer weights
        weight_key = "couplings.0.s_net.0.weight"
        if weight_key in state_dict:
            hidden_dim = state_dict[weight_key].shape[0]
        else:
            raise ValueError("Cannot infer hidden_dim: couplings.0.s_net.0.weight not found in checkpoint")
        
        return {
            "dim": dim,
            "n_couplings": n_couplings, 
            "hidden_dim": hidden_dim,
            "use_permutation": True,  # Default assumption
        }
    
    def propose(
        self,
        x_lo: torch.Tensor,
        x_hi: torch.Tensor,
        h_lo: Optional[torch.Tensor] = None,
        h_hi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propose a flow-based swap between neighboring replicas.
        
        The proposal uses the trained flow to map configurations between
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
        
        # Store original dtypes for restoration
        orig_dtype_lo = x_lo.dtype
        orig_dtype_hi = x_hi.dtype
        
        # Convert to flow model's dtype (typically float32)
        flow_dtype = next(self.flow.parameters()).dtype
        x_lo = x_lo.to(device=self.device, dtype=flow_dtype)
        x_hi = x_hi.to(device=self.device, dtype=flow_dtype)
        
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
            
            # Convert back to original dtypes
            y_lo = y_lo.to(dtype=orig_dtype_lo)
            y_hi = y_hi.to(dtype=orig_dtype_hi)
            log_qratio = log_qratio.to(dtype=orig_dtype_lo)
        
        return y_lo, y_hi, log_qratio
    
    def update_checkpoint(self, new_checkpoint: Union[str, Path]):
        """Update the flow checkpoint and reload the model.
        
        Args:
            new_checkpoint: Path to new checkpoint file
        """
        self.flow_checkpoint = Path(new_checkpoint)
        self._load_flow() 