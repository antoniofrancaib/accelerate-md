"""Message-passing graph embedding for molecular structure conditioning.

This module implements a simple message-passing GNN approach where:
- Atom types (C,H,O,N) are node features  
- Bonds are edges for message passing
- Fully dimension-agnostic (works for any molecular size)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

__all__ = ["MessagePassingGNN"]


class MessagePassingGNN(nn.Module):
    """Simple message-passing GNN for molecular graphs.
    
    Implements your supervisor's suggestion:
    - Node features: atom types (C,H,O,N) 
    - Edges: molecular bonds
    - Message passing: standard GCN-style aggregation
    - Output: updated node features (dimension-agnostic)
    
    Parameters
    ----------
    atom_vocab_size : int
        Number of unique atom types (e.g., 4 for H,C,N,O)
    atom_embed_dim : int
        Initial atom embedding dimension
    hidden_dim : int
        Hidden dimension for message-passing layers
    num_layers : int
        Number of message-passing layers
    output_dim : int
        Output dimension per node
    """
    
    def __init__(
        self,
        atom_vocab_size: int = 4,  # H, C, N, O
        atom_embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 64,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Initial atom embeddings
        self.atom_embedder = nn.Embedding(atom_vocab_size, atom_embed_dim)
        
        # Message-passing layers
        self.mp_layers = nn.ModuleList()
        
        # Input layer: atom_embed_dim -> hidden_dim
        self.mp_layers.append(
            MessagePassingLayer(atom_embed_dim, hidden_dim)
        )
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.mp_layers.append(
                MessagePassingLayer(hidden_dim, hidden_dim)
            )
        
        # Output layer: hidden_dim -> output_dim
        if num_layers > 1:
            self.mp_layers.append(
                MessagePassingLayer(hidden_dim, output_dim)
            )
        else:
            # Single layer case
            self.mp_layers[0] = MessagePassingLayer(atom_embed_dim, output_dim)
            
        # Global graph pooling for graph-level features (simple mean pooling)
        self.global_pool = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim // 4)
        )
        
        # Graph-level feature integration
        self.graph_integrator = nn.Sequential(
            nn.Linear(output_dim + output_dim // 4, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(
        self, 
        atom_types: Tensor,  # [B, N] atom type indices
        coordinates: Optional[Tensor] = None,  # [B, N, 3] coordinates for distance features
        adj_list: Optional[Tensor] = None,  # [E, 2] edge indices  
        edge_batch_idx: Optional[Tensor] = None,  # [E] batch indices for edges
        masked_elements: Optional[Tensor] = None,  # [B, N] padding mask
    ) -> Tensor:
        """Apply message passing to update node features.
        
        Parameters
        ----------
        atom_types : Tensor, shape [B, N]
            Atom type indices for each atom
        coordinates : Tensor, shape [B, N, 3], optional
            Molecular coordinates for distance-based edge features
        adj_list : Tensor, shape [E, 2], optional
            Edge connectivity (bond pairs) - required for message passing
        edge_batch_idx : Tensor, shape [E], optional  
            Batch index for each edge - required for batched graphs
        masked_elements : Tensor, shape [B, N], optional
            True for padding atoms
            
        Returns
        -------
        node_features : Tensor, shape [B, N, output_dim]
            Updated node features after message passing
        """
        B, N = atom_types.shape
        
        # Ensure atom_types is on the same device as the embedder
        embedder_device = next(self.atom_embedder.parameters()).device
        atom_types = atom_types.to(embedder_device)
        
        # Move other tensors to the same device if provided
        if coordinates is not None:
            coordinates = coordinates.to(embedder_device)
        if adj_list is not None:
            adj_list = adj_list.to(embedder_device)
        if edge_batch_idx is not None:
            edge_batch_idx = edge_batch_idx.to(embedder_device)
        if masked_elements is not None:
            masked_elements = masked_elements.to(embedder_device)
        
        # Initial atom embeddings
        h = self.atom_embedder(atom_types)  # [B, N, atom_embed_dim]
        
        # Apply message-passing layers
        for layer in self.mp_layers:
            h = layer(
                node_features=h,
                coordinates=coordinates,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements
            )
            
        # Global graph pooling (simple mean pooling over valid nodes)
        if masked_elements is not None:
            # Mask out padding before pooling
            valid_mask = (~masked_elements).unsqueeze(-1).float()  # [B, N, 1]
            masked_h = h * valid_mask
            valid_counts = valid_mask.sum(dim=1, keepdim=True)  # [B, 1, 1]
            global_features = masked_h.sum(dim=1, keepdim=True) / (valid_counts + 1e-8)  # [B, 1, output_dim]
        else:
            global_features = h.mean(dim=1, keepdim=True)  # [B, 1, output_dim]
            
        # Process global features
        global_features = self.global_pool(global_features)  # [B, 1, output_dim//4]
        
        # Broadcast global features to all nodes
        global_features_broadcast = global_features.expand(B, N, -1)  # [B, N, output_dim//4]
        
        # Integrate local and global features
        combined_features = torch.cat([h, global_features_broadcast], dim=-1)  # [B, N, output_dim + output_dim//4]
        h = self.graph_integrator(combined_features)  # [B, N, output_dim]
            
        return h


class MessagePassingLayer(nn.Module):
    """Single message-passing layer with residual connections and layer norm.
    
    Implements: h_i^{l+1} = LayerNorm(h_i^l + ReLU(W * (h_i^l + AGG_j(M(h_i^l, h_j^l, e_ij)))))
    where AGG is sum aggregation and M processes node pairs with edge features.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Message function: combine source, target, and edge features
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * input_dim + 4, output_dim),  # +4 for distance, angle, dihedral, bond_type
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Update function: combine original features with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(input_dim + output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        node_features: Tensor,  # [B, N, input_dim]
        coordinates: Optional[Tensor] = None,  # [B, N, 3] for distance features
        adj_list: Optional[Tensor] = None,  # [E, 2]
        edge_batch_idx: Optional[Tensor] = None,  # [E]
        masked_elements: Optional[Tensor] = None,  # [B, N]
    ) -> Tensor:
        """Apply one message-passing step with distance-based edge features."""
        B, N, D = node_features.shape
        device = node_features.device
        
        if adj_list is None or edge_batch_idx is None:
            # No edges provided - just apply self-transformation
            return self.layer_norm(
                self.update_mlp(
                    torch.cat([node_features, torch.zeros(B, N, self.output_dim, device=device)], dim=-1)
                ) + self.residual_proj(node_features)
            )
        
        # Flatten node features for edge indexing: [B*N, D]
        h_flat = node_features.view(-1, D)
        
        # Ensure edge_batch_idx and adj_list have matching first dimension
        E = adj_list.shape[0]
        if edge_batch_idx.shape[0] != E:
            raise RuntimeError(
                f"Edge batch index size {edge_batch_idx.shape[0]} doesn't match "
                f"adjacency list size {E}. Expected edge_batch_idx to have {E} elements."
            )
        
        # Convert batched edge indices to global indices
        global_edge_index = adj_list.clone()  # [E, 2]
        batch_offset = edge_batch_idx * N  # [E]
        global_edge_index[:, 0] += batch_offset  # source nodes
        global_edge_index[:, 1] += batch_offset  # target nodes
        
        # Ensure indices are within bounds
        max_global_idx = B * N - 1
        if global_edge_index.max() > max_global_idx:
            raise RuntimeError(
                f"Global edge index {global_edge_index.max()} exceeds "
                f"maximum allowed index {max_global_idx} for {B} batches of {N} nodes."
            )
        
        # Extract source and target node features
        src_idx = global_edge_index[:, 0]  # [E]
        tgt_idx = global_edge_index[:, 1]  # [E]
        
        h_src = h_flat[src_idx]  # [E, D]
        h_tgt = h_flat[tgt_idx]  # [E, D]
        
        # Compute edge features (distance if coordinates provided)
        edge_features = torch.ones(E, 4, device=device)  # 4 features: [distance, angle, dihedral, bond_type]
        if coordinates is not None:
            coords_flat = coordinates.view(-1, 3)  # [B*N, 3]
            src_coords = coords_flat[src_idx]  # [E, 3]
            tgt_coords = coords_flat[tgt_idx]  # [E, 3]
            
            # Feature 1: Distance
            distances = torch.norm(src_coords - tgt_coords, dim=-1, keepdim=True)  # [E, 1]
            
            # Feature 2: Bond angle (simplified - using vector angles)
            # This is a placeholder - in real systems you'd compute actual bond angles
            bond_vectors = tgt_coords - src_coords  # [E, 3]
            # Compute angle with respect to a reference direction (e.g., z-axis)
            z_ref = torch.tensor([0., 0., 1.], device=device).expand_as(bond_vectors)
            cosines = torch.sum(bond_vectors * z_ref, dim=-1, keepdim=True) / (torch.norm(bond_vectors, dim=-1, keepdim=True) + 1e-8)
            angles = torch.acos(torch.clamp(cosines, -1+1e-6, 1-1e-6))  # [E, 1]
            
            # Feature 3: Dihedral angle (simplified - using cross products)
            # This is a placeholder for proper dihedral calculation
            cross_prods = torch.cross(bond_vectors, z_ref, dim=-1)
            dihedral_indicator = torch.norm(cross_prods, dim=-1, keepdim=True)  # [E, 1]
            
            # Feature 4: Bond type indicator (based on distance ranges)
            # Typical covalent bond ranges: 1.0-1.8 Å, adjust as needed
            bond_type = torch.sigmoid(2.0 * (1.5 - distances))  # [E, 1], higher for shorter bonds
            
            edge_features = torch.cat([distances, angles, dihedral_indicator, bond_type], dim=-1)
        
        # Compute messages: combine node features with edge features
        edge_input = torch.cat([h_src, h_tgt, edge_features], dim=-1)  # [E, 2*D + 4]
        messages = self.message_mlp(edge_input)  # [E, output_dim]
        
        # Aggregate messages for each node (sum aggregation)
        aggregated = torch.zeros(B * N, messages.shape[-1], device=device)
        aggregated.index_add_(0, tgt_idx, messages)  # Add messages to target nodes
        
        # Reshape back to [B, N, output_dim]
        aggregated = aggregated.view(B, N, -1)
        
        # Update: combine original features with aggregated messages
        combined = torch.cat([node_features, aggregated], dim=-1)
        updated = self.update_mlp(combined)
        
        # Apply padding mask
        if masked_elements is not None:
            updated = updated * (~masked_elements).unsqueeze(-1).float()
            
        # Apply residual connection and layer normalization
        updated = self.layer_norm(updated + self.residual_proj(node_features))
        
        return updated 