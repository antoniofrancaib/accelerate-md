"""Message-passing coupling layer for molecular coordinates.

This module implements a dimension-agnostic coupling layer using message-passing
over molecular graphs. Based on your supervisor's suggestion:
- Direct message passing on bonds (edges)
- Atom types as node features
- No complex attention mechanisms
- Fully size-agnostic
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .mlp import MLP
from .graph_embedding import MessagePassingGNN

__all__ = ["MessagePassingCouplingLayer"]


class MessagePassingCouplingLayer(nn.Module):
    """Dimension-agnostic message-passing coupling layer.
    
    Implements your supervisor's approach:
    1. Message passing over molecular bonds to get enriched node features
    2. Use node features to predict per-atom scale/shift parameters
    3. Apply affine transformation to coordinates
    
    This is fully dimension-agnostic - works for any molecular size!
    
    Parameters
    ----------
    phase : int
        Coupling phase (0 or 1) for alternating atom masking
    atom_vocab_size : int
        Number of unique atom types
    atom_embed_dim : int
        Dimension of atom type embeddings
    hidden_dim : int
        Hidden dimension for message passing and MLPs
    num_mp_layers : int
        Number of message-passing layers
    """
    
    def __init__(
        self,
        phase: int,  # 0 or 1 for alternating masks
        atom_vocab_size: int = 4,
        atom_embed_dim: int = 64,
        hidden_dim: int = 128,
        num_mp_layers: int = 2,
    ):
        super().__init__()
        self.phase = phase
        
        # Message-passing network to process molecular structure
        self.message_passing = MessagePassingGNN(
            atom_vocab_size=atom_vocab_size,
            atom_embed_dim=atom_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mp_layers,
            output_dim=hidden_dim,
        )
        
        # Networks to predict scale and shift parameters
        # Input: message-passed features + coordinate information
        input_dim = hidden_dim + 3  # node features + coordinates
        
        self.scale_network = MLP(
            input_dim=input_dim,
            out_dim=3,  # Scale per coordinate (x, y, z)
            hidden_layer_dims=[hidden_dim],
        )
        
        self.shift_network = MLP(
            input_dim=input_dim,
            out_dim=3,  # Shift per coordinate (x, y, z)
            hidden_layer_dims=[hidden_dim],
        )
        
        # Initialize scale network for stable training
        with torch.no_grad():
            if hasattr(self.scale_network._layers, '__getitem__'):
                final_layer = self.scale_network._layers[-1]
                if hasattr(final_layer, 'weight'):
                    torch.nn.init.zeros_(final_layer.weight)
                    if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                        torch.nn.init.zeros_(final_layer.bias)
        
    def forward(
        self,
        coordinates: Tensor,  # [B, N, 3] input coordinates
        atom_types: Tensor,   # [B, N] atom type indices
        adj_list: Optional[Tensor] = None,  # [E, 2] edge connectivity
        edge_batch_idx: Optional[Tensor] = None,  # [E] batch indices for edges
        masked_elements: Optional[Tensor] = None,  # [B, N] padding mask
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Apply dimension-agnostic message-passing coupling transformation.
        
        Parameters
        ----------
        coordinates : Tensor, shape [B, N, 3]
            Input molecular coordinates
        atom_types : Tensor, shape [B, N] 
            Atom type indices
        adj_list : Tensor, shape [E, 2], optional
            Edge connectivity (molecular bonds)
        edge_batch_idx : Tensor, shape [E], optional
            Batch indices for edges
        masked_elements : Tensor, shape [B, N], optional
            Padding mask (True for padding atoms)
        reverse : bool
            Whether to apply reverse transformation
            
        Returns
        -------
        output_coords : Tensor, shape [B, N, 3]
            Transformed coordinates
        log_det : Tensor, shape [B]
            Log-determinant of Jacobian
        """
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Generate coupling mask (alternating atoms)
        atom_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B, N]
        coupling_mask = (atom_indices % 2 == self.phase)  # [B, N] boolean mask
        
        # Combine with padding mask
        if masked_elements is not None:
            coupling_mask = coupling_mask & (~masked_elements)
        
        # Split coordinates: unchanged vs to-be-transformed
        unchanged_mask = ~coupling_mask  # [B, N]
        
        # Step 1: Message passing to get enriched node features
        node_features = self.message_passing(
            atom_types=atom_types,
            coordinates=coordinates,  # Pass coordinates for distance features
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
        )  # [B, N, hidden_dim]
        
        # Step 2: Combine node features with coordinate information
        # Use unchanged coordinates as conditioning
        conditioning_coords = coordinates * unchanged_mask.unsqueeze(-1)
        
        # Concatenate node features with coordinates
        combined_features = torch.cat([
            node_features,  # [B, N, hidden_dim]
            conditioning_coords,  # [B, N, 3]
        ], dim=-1)  # [B, N, hidden_dim + 3]
        
        # Step 3: Predict scale and shift parameters
        raw_scales = self.scale_network(combined_features)  # [B, N, 3]
        shifts = self.shift_network(combined_features)      # [B, N, 3]
        
        # Allow expressive transformations while maintaining stability
        log_scales = torch.tanh(raw_scales) * 2.0  # Allow ±2.0 range (40x more expressive)
        
        # Apply coupling mask (only transform specified atoms)
        mask_3d = coupling_mask.unsqueeze(-1)  # [B, N, 1]
        log_scales = log_scales * mask_3d  # Zero out unchanged atoms
        shifts = shifts * mask_3d
        
        # Mask out padding atoms from Jacobian calculation
        if masked_elements is not None:
            valid_mask = (~masked_elements).unsqueeze(-1)  # [B, N, 1]
            log_scales = log_scales * valid_mask
            
        # Compute log-determinant (sum over atoms and coordinates)
        log_det = log_scales.sum(dim=(1, 2))  # [B]
        
        # Apply affine transformation
        scales = torch.exp(log_scales)  # [B, N, 3]
        
        if reverse:
            # Inverse: x = (y - t) / s
            output_coords = (coordinates - shifts) / (scales + 1e-8)
            log_det = -log_det  # Negative for inverse
        else:
            # Forward: y = s * x + t
            output_coords = scales * coordinates + shifts
            
        # Preserve unchanged coordinates
        output_coords = (
            output_coords * mask_3d +  # Transformed part
            coordinates * (~mask_3d)   # Unchanged part
        )
        
        return output_coords, log_det
        
    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"phase={self.phase}, message_passing" 