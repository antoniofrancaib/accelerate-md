"""Graph-conditioned NVP coupling layer for molecular coordinates.

This module implements the complete graph-conditioned Real-NVP coupling layer
according to equations (8-10) and (13-16) from the Flow-PT formulation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .mlp import MLP
from .graph_embedding import GraphEmbedding
from .attention_encoder import KernelAttentionEncoder

__all__ = ["GraphNVPCouplingLayer"]


class GraphNVPCouplingLayer(nn.Module):
    """Graph-conditioned Real-NVP coupling layer.
    
    Implements the complete Flow-PT coupling layer that combines:
    1. Global graph embedding: h_G = GraphEmb(G)  [Eq. 13]
    2. Node features: h_i = f_θ(x_i, a_i, h_G)  [Eq. 14] 
    3. Attention aggregation: AGG_θ(h_1, ..., h_N)  [Eq. 15-16]
    4. Affine transformation: x^{l+1} = s_θ ⊙ x' + t_θ  [Eq. 8]
    
    Parameters
    ----------
    phase : int
        Coupling phase (0 or 1) for dynamic masking
    atom_vocab_size : int
        Number of unique atom types
    atom_embed_dim : int
        Dimension of atom type embeddings
    graph_embed_dim : int  
        Dimension of global graph embedding h_G (can be 0)
    node_feature_dim : int
        Dimension of per-atom features h_i
    hidden_dim : int
        Hidden dimension for MLPs
    attention_lengthscales : list
        Lengthscales for multi-head attention
    use_graph_embedding : bool
        Whether to use global graph conditioning
    """
    
    def __init__(
        self,
        phase: int,  # 0 or 1 for alternating masks
        atom_vocab_size: int = 4,
        atom_embed_dim: int = 32,
        graph_embed_dim: int = 64,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        attention_lengthscales: list = [1.0, 2.0, 4.0],
        use_graph_embedding: bool = True,
    ):
        super().__init__()
        self.phase = phase
        self.use_graph_embedding = use_graph_embedding
        self.graph_embed_dim = graph_embed_dim
        self.node_feature_dim = node_feature_dim
        
        # Global graph embedding module
        if use_graph_embedding and graph_embed_dim > 0:
            self.graph_embedder = GraphEmbedding(
                atom_vocab_size=atom_vocab_size,
                atom_embed_dim=atom_embed_dim,
                graph_embed_dim=graph_embed_dim,
                use_bonds=True,
            )
        else:
            self.graph_embedder = None
            graph_embed_dim = 0  # Override for consistency
            
        # Node feature network f_θ(x_i, a_i, h_G)
        node_input_dim = 3 + atom_embed_dim + graph_embed_dim  # coords + atom + graph
        self.atom_embedder = nn.Embedding(atom_vocab_size, atom_embed_dim)
        self.node_network = MLP(
            input_dim=node_input_dim,
            out_dim=node_feature_dim,
            hidden_layer_dims=[hidden_dim, hidden_dim],  # 2 hidden layers
        )
        
        # Attention aggregation AGG_θ
        self.attention_encoder = KernelAttentionEncoder(
            input_dim=node_feature_dim,
            output_dim=hidden_dim,  # Intermediate dimension
            lengthscales=attention_lengthscales,
            use_value_projection=True,
        )
        
        # Scale and shift networks (s_θ and t_θ)
        self.scale_network = MLP(
            input_dim=hidden_dim,
            out_dim=3,  # Scale per coordinate
            hidden_layer_dims=[hidden_dim],  # 1 hidden layer
        )
        
        self.shift_network = MLP(
            input_dim=hidden_dim,
            out_dim=3,  # Shift per coordinate  
            hidden_layer_dims=[hidden_dim],  # 1 hidden layer
        )
        
    def forward(
        self,
        coordinates: Tensor,  # [B, N, 3] input coordinates
        atom_types: Tensor,   # [B, N] atom type indices
        adj_list: Optional[Tensor] = None,  # [E, 2] edge connectivity
        edge_batch_idx: Optional[Tensor] = None,  # [E] batch indices for edges
        masked_elements: Optional[Tensor] = None,  # [B, N] padding mask
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Forward/reverse coupling transformation.
        
        Parameters
        ----------
        coordinates : Tensor, shape [B, N, 3]
            Input molecular coordinates
        atom_types : Tensor, shape [B, N] 
            Atom type indices
        adj_list : Tensor, shape [E, 2], optional
            Edge connectivity
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
        
        # Ensure all inputs are on the model device
        model_device = next(self.parameters()).device
        coordinates = coordinates.to(model_device)
        atom_types = atom_types.to(model_device)
        if adj_list is not None:
            adj_list = adj_list.to(model_device)
        if edge_batch_idx is not None:
            edge_batch_idx = edge_batch_idx.to(model_device)
        if masked_elements is not None:
            masked_elements = masked_elements.to(model_device)
            
        device = coordinates.device
        
        # Generate dynamic coupling mask
        atom_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B, N]
        coupling_mask = (atom_indices % 2 == self.phase)  # [B, N] boolean mask
        
        # Combine with padding mask
        if masked_elements is not None:
            coupling_mask = coupling_mask & (~masked_elements)
        
        # Split coordinates: x' (unchanged) and x (to be transformed)
        unchanged_mask = ~coupling_mask  # [B, N]
        x_prime = coordinates * unchanged_mask.unsqueeze(-1)  # [B, N, 3]
        
        # Step 1: Global graph embedding h_G = GraphEmb(G)
        if self.graph_embedder is not None:
            h_G = self.graph_embedder(
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx, 
                masked_elements=masked_elements,
            )  # [B, graph_embed_dim]
            # Broadcast to per-atom
            h_G_broadcast = h_G.unsqueeze(1).expand(-1, N, -1)  # [B, N, graph_embed_dim]
        else:
            h_G_broadcast = torch.zeros(B, N, 0, device=device)  # [B, N, 0]
            
        # Step 2: Per-atom features h_i = f_θ(x_i, a_i, h_G)
        atom_embeds = self.atom_embedder(atom_types)  # [B, N, atom_embed_dim]
        node_inputs = torch.cat([
            x_prime,  # Current coordinates (unchanged part)
            atom_embeds,  # Atom type embeddings  
            h_G_broadcast,  # Global graph context
        ], dim=-1)  # [B, N, 3 + atom_embed_dim + graph_embed_dim]
        
        h_nodes = self.node_network(node_inputs)  # [B, N, node_feature_dim]
        
        # Step 3: Attention aggregation AGG_θ(h_1, ..., h_N) 
        aggregated_features = self.attention_encoder(
            node_features=h_nodes,
            coordinates=x_prime,  # Use unchanged coordinates for attention
            masked_elements=masked_elements,
        )  # [B, N, hidden_dim]
        
        # Step 4: Scale and shift parameters
        log_scales = self.scale_network(aggregated_features)  # [B, N, 3]
        shifts = self.shift_network(aggregated_features)      # [B, N, 3]
        
        # Apply coupling mask to scale/shift (only transform specified atoms)
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
        return (
            f"phase={self.phase}, "
            f"graph_embed_dim={self.graph_embed_dim}, "
            f"node_feature_dim={self.node_feature_dim}, "
            f"use_graph_embedding={self.use_graph_embedding}"
        ) 