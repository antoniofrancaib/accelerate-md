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
        # New parameters for increased expressivity
        scale_range: float = 0.5,  # Allow ±50% scaling instead of ±5%
        deeper_networks: bool = True,  # Use deeper MLPs
        temperature_conditioning: bool = True,  # Add temperature awareness
    ):
        super().__init__()
        self.phase = phase
        self.scale_range = scale_range
        self.temperature_conditioning = temperature_conditioning
        
        # Message-passing network to process molecular structure
        self.message_passing = MessagePassingGNN(
            atom_vocab_size=atom_vocab_size,
            atom_embed_dim=atom_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mp_layers,
            output_dim=hidden_dim,
        )
        
        # Networks to predict scale and shift parameters
        # Input: message-passed features + coordinate information [+ temperature info]
        input_dim = hidden_dim + 3  # node features + coordinates
        if temperature_conditioning:
            input_dim += 2  # Add source and target temperature
        
        # Deeper, more expressive networks
        if deeper_networks:
            hidden_layers = [hidden_dim, hidden_dim, hidden_dim // 2]
        else:
            hidden_layers = [hidden_dim]
        
        self.scale_network = MLP(
            input_dim=input_dim,
            out_dim=3,  # Scale per coordinate (x, y, z)
            hidden_layer_dims=hidden_layers,
        )
        
        self.shift_network = MLP(
            input_dim=input_dim,
            out_dim=3,  # Shift per coordinate (x, y, z)
            hidden_layer_dims=hidden_layers,
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
        # New temperature conditioning parameters
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
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
        source_temp : float, optional
            Source temperature for conditioning
        target_temp : float, optional
            Target temperature for conditioning
            
        Returns
        -------
        output_coords : Tensor, shape [B, N, 3]
            Transformed coordinates
        log_det : Tensor, shape [B]
            Log-determinant of Jacobian
        """
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Generate chemically-aware coupling mask
        coupling_mask = self._create_coupling_mask(atom_types, masked_elements, device)
        
        # Split coordinates: unchanged vs to-be-transformed
        unchanged_mask = ~coupling_mask
        
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
        
        # Step 3: Add temperature conditioning if enabled
        if self.temperature_conditioning and source_temp is not None and target_temp is not None:
            # Create temperature features: [source_temp, target_temp]
            temp_features = torch.tensor([source_temp, target_temp], device=device)
            temp_features = temp_features.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # [B, N, 2]
            combined_features = torch.cat([combined_features, temp_features], dim=-1)
        
        # Step 4: Predict scale and shift parameters
        raw_scales = self.scale_network(combined_features)  # [B, N, 3]
        shifts = self.shift_network(combined_features)      # [B, N, 3]
        
        # Allow expressive transformations while maintaining stability
        log_scales = torch.tanh(raw_scales) * self.scale_range  # Configurable scaling range
        
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
    
    def _create_coupling_mask(self, atom_types: Tensor, masked_elements: Optional[Tensor], device) -> Tensor:
        """Create chemically-aware coupling mask instead of simple alternating.
        
        This creates more meaningful coupling patterns based on atom types and molecular structure.
        """
        B, N = atom_types.shape
        
        if self.phase == 0:
            # Phase 0: Transform heavy atoms (C, N, O) and leave H unchanged
            # Atom type mapping: 0=H, 1=C, 2=N, 3=O (common convention)
            coupling_mask = atom_types > 0  # Transform non-hydrogen atoms
        else:
            # Phase 1: Transform hydrogen atoms and leave heavy atoms unchanged
            coupling_mask = atom_types == 0  # Transform hydrogen atoms only
        
        # Combine with padding mask
        if masked_elements is not None:
            coupling_mask = coupling_mask & (~masked_elements)
        
        return coupling_mask
        
    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"phase={self.phase}, message_passing" 