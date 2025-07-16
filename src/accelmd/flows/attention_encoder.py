"""Kernel attention encoder for molecular coordinate conditioning.

This module implements the AGG_θ function from equations (15-16) that performs
distance-based attention aggregation over molecular coordinates.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List

__all__ = ["KernelAttentionEncoder"]


class KernelAttentionEncoder(nn.Module):
    """Distance-based attention aggregation for molecular coordinates.
    
    Implements AGG_θ(h_1, ..., h_N | C) from equation (15) where attention
    weights are based on coordinate distances w_ij = exp(-||x_i - x_j||^2 / σ^2).
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension m per atom
    output_dim : int  
        Output dimension per atom (3N total for coordinates)
    lengthscales : List[float]
        Attention lengthscales σ for multi-head attention
    use_value_projection : bool
        Whether to use learnable value projection V
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lengthscales: List[float] = [1.0, 2.0, 4.0],  # Multi-head σ values
        use_value_projection: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  
        self.lengthscales = lengthscales
        self.num_heads = len(lengthscales)
        
        # Register lengthscales as buffer (non-trainable)
        self.register_buffer("sigma_values", torch.tensor(lengthscales))
        
        if use_value_projection:
            # Learnable value projection V ∈ R^{d × m} for each head
            self.value_projections = nn.ModuleList([
                nn.Linear(input_dim, output_dim, bias=False)
                for _ in range(self.num_heads)
            ])
            
            # Combine multi-head outputs
            self.output_projection = nn.Linear(
                self.num_heads * output_dim, output_dim
            )
        else:
            # No learnable projection (V = I)
            assert input_dim == output_dim, "input_dim must equal output_dim when use_value_projection=False"
            self.value_projections = None
            self.output_projection = None
    
    def forward(
        self,
        node_features: Tensor,  # [B, N, input_dim] h_i features  
        coordinates: Tensor,    # [B, N, 3] x_i coordinates for attention
        masked_elements: Optional[Tensor] = None,  # [B, N] padding mask
    ) -> Tensor:
        """Apply distance-based attention aggregation.
        
        Parameters
        ----------
        node_features : Tensor, shape [B, N, input_dim]
            Per-atom features h_i = f_θ(x_i, a_i, h_G)
        coordinates : Tensor, shape [B, N, 3]
            Current coordinates x_i for computing attention weights
        masked_elements : Tensor, shape [B, N], optional
            True for padding atoms
            
        Returns
        -------
        aggregated : Tensor, shape [B, N, output_dim] 
            Attention-aggregated features
        """
        B, N, _ = node_features.shape
        device = node_features.device
        
        # Ensure all inputs are on the same device as node_features
        coordinates = coordinates.to(device)
        if masked_elements is not None:
            masked_elements = masked_elements.to(device)
        
        # Compute pairwise distances ||x_i - x_j||^2
        coords_i = coordinates.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coordinates.unsqueeze(1)  # [B, 1, N, 3]
        dist_squared = ((coords_i - coords_j) ** 2).sum(dim=-1)  # [B, N, N]
        
        # Multi-head attention over different lengthscales
        head_outputs = []
        
        for head_idx, sigma in enumerate(self.lengthscales):
            # Compute attention weights w_ij = exp(-||x_i - x_j||^2 / σ^2)
            attention_logits = -dist_squared / (sigma ** 2)  # [B, N, N]
            
            # CRITICAL: Clamp attention logits to prevent numerical explosion
            attention_logits = torch.clamp(attention_logits, min=-20.0, max=20.0)
            
            # Mask out padding atoms in attention
            if masked_elements is not None:
                # Mask rows (queries) and columns (keys) for padding atoms
                mask_i = masked_elements.unsqueeze(2)  # [B, N, 1] 
                mask_j = masked_elements.unsqueeze(1)  # [B, 1, N]
                attention_mask = mask_i | mask_j  # [B, N, N]
                attention_logits = attention_logits.masked_fill(attention_mask, -1e9)
            
            # Softmax normalization
            attention_weights = torch.softmax(attention_logits, dim=-1)  # [B, N, N]
            
            # Apply value projection if enabled
            if self.value_projections is not None:
                values = self.value_projections[head_idx](node_features)  # [B, N, output_dim]
            else:
                values = node_features  # [B, N, input_dim]
                
            # Attention aggregation: sum_j w_ij * V * h_j  
            head_output = torch.bmm(attention_weights, values)  # [B, N, output_dim]
            head_outputs.append(head_output)
        
        # Combine multi-head outputs
        if len(head_outputs) == 1:
            aggregated = head_outputs[0]
        else:
            # Concatenate heads and project
            multi_head = torch.cat(head_outputs, dim=-1)  # [B, N, num_heads * output_dim]
            if self.output_projection is not None:
                aggregated = self.output_projection(multi_head)  # [B, N, output_dim]
            else:
                # Simple average if no projection
                aggregated = multi_head.view(B, N, self.num_heads, -1).mean(dim=2)
        
        # Zero out padding atoms in output
        if masked_elements is not None:
            aggregated = aggregated * (~masked_elements).unsqueeze(-1)
            
        return aggregated


class MultiScaleAttentionEncoder(KernelAttentionEncoder):
    """Multi-scale variant with learnable lengthscales.
    
    Extends KernelAttentionEncoder to make lengthscales σ learnable parameters
    instead of fixed hyperparameters.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 3,
        initial_lengthscales: Optional[List[float]] = None,
        use_value_projection: bool = True,
    ):
        # Initialize with learnable lengthscales
        if initial_lengthscales is None:
            initial_lengthscales = [1.0, 2.0, 4.0][:num_heads]
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim, 
            lengthscales=initial_lengthscales,
            use_value_projection=use_value_projection,
        )
        
        # Make lengthscales learnable (log-space for stability)
        self.log_sigma = nn.Parameter(
            torch.log(torch.tensor(initial_lengthscales))
        )
        
    def forward(
        self,
        node_features: Tensor,
        coordinates: Tensor, 
        masked_elements: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with learnable lengthscales."""
        B, N, _ = node_features.shape
        
        # Use learnable lengthscales
        sigma_values = torch.exp(self.log_sigma)  # [num_heads]
        
        # Compute distances
        coords_i = coordinates.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coordinates.unsqueeze(1)  # [B, 1, N, 3] 
        dist_squared = ((coords_i - coords_j) ** 2).sum(dim=-1)  # [B, N, N]
        
        # Multi-head attention
        head_outputs = []
        
        for head_idx in range(self.num_heads):
            sigma = sigma_values[head_idx]
            
            # Attention weights
            attention_logits = -dist_squared / (sigma ** 2)
            
            # Mask padding
            if masked_elements is not None:
                mask_i = masked_elements.unsqueeze(2)
                mask_j = masked_elements.unsqueeze(1) 
                attention_mask = mask_i | mask_j
                attention_logits = attention_logits.masked_fill(attention_mask, -1e9)
            
            attention_weights = torch.softmax(attention_logits, dim=-1)
            
            # Value projection and aggregation
            if self.value_projections is not None:
                values = self.value_projections[head_idx](node_features)
            else:
                values = node_features
                
            head_output = torch.bmm(attention_weights, values)
            head_outputs.append(head_output)
        
        # Combine heads
        if len(head_outputs) == 1:
            aggregated = head_outputs[0]
        else:
            multi_head = torch.cat(head_outputs, dim=-1)
            if self.output_projection is not None:
                aggregated = self.output_projection(multi_head)
            else:
                aggregated = multi_head.view(B, N, self.num_heads, -1).mean(dim=2)
        
        # Mask output 
        if masked_elements is not None:
            aggregated = aggregated * (~masked_elements).unsqueeze(-1)
            
        return aggregated 