"""Graph embedding module for molecular structure conditioning.

This module implements the GraphEmb(G) function from equation (13) that creates
a global graph representation h_G ∈ R^d from molecular structure.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

__all__ = ["GraphEmbedding"]


class GraphEmbedding(nn.Module):
    """Global graph embedding from molecular structure.
    
    Computes h_G = GraphEmb(G) where G includes atom types and adjacency.
    This provides a global molecular context that can be zero-dimensional
    (d=0) for ablation studies.
    
    Parameters
    ----------
    atom_vocab_size : int
        Number of unique atom types (e.g., 4 for H,C,N,O)
    atom_embed_dim : int  
        Dimension of atom type embeddings
    graph_embed_dim : int
        Output dimension d of h_G (can be 0 for no graph conditioning)
    use_bonds : bool
        Whether to include bond information in graph embedding
    """
    
    def __init__(
        self,
        atom_vocab_size: int = 4,  # H, C, N, O
        atom_embed_dim: int = 32,
        graph_embed_dim: int = 64,
        use_bonds: bool = True,
    ):
        super().__init__()
        self.graph_embed_dim = graph_embed_dim
        self.use_bonds = use_bonds
        
        if graph_embed_dim == 0:
            # No graph conditioning (d=0 case)
            self.atom_embedder = None
            self.graph_encoder = None
        else:
            # Atom type embeddings
            self.atom_embedder = nn.Embedding(atom_vocab_size, atom_embed_dim)
            
            # Global graph encoder
            input_dim = atom_embed_dim
            if use_bonds:
                # Add bond embedding dimension
                input_dim += 16  # Simple bond feature dimension
                
            self.graph_encoder = nn.Sequential(
                nn.Linear(input_dim, graph_embed_dim * 2),
                nn.ReLU(),
                nn.Linear(graph_embed_dim * 2, graph_embed_dim),
            )
    
    def forward(
        self, 
        atom_types: Tensor,  # [B, N] atom type indices
        adj_list: Optional[Tensor] = None,  # [E, 2] edge indices  
        edge_batch_idx: Optional[Tensor] = None,  # [E] batch indices for edges
        masked_elements: Optional[Tensor] = None,  # [B, N] padding mask
    ) -> Tensor:
        """Compute global graph embedding h_G.
        
        Parameters
        ----------
        atom_types : Tensor, shape [B, N]
            Atom type indices for each atom
        adj_list : Tensor, shape [E, 2], optional
            Edge connectivity (bond pairs)
        edge_batch_idx : Tensor, shape [E], optional  
            Batch index for each edge
        masked_elements : Tensor, shape [B, N], optional
            True for padding atoms
            
        Returns
        -------
        h_G : Tensor, shape [B, d]
            Global graph embeddings for each molecule in batch
        """
        B = atom_types.shape[0]
        
        if self.graph_embed_dim == 0:
            # Return zero-dimensional embedding
            return torch.zeros(B, 0, device=atom_types.device)
            
        # Embed atom types
        atom_embeds = self.atom_embedder(atom_types)  # [B, N, atom_embed_dim]
        
        # Mask out padding atoms
        if masked_elements is not None:
            atom_embeds = atom_embeds * (~masked_elements).unsqueeze(-1)
        
        # Aggregate to global representation
        if self.use_bonds and adj_list is not None and edge_batch_idx is not None:
            # Include bond information
            h_mol = self._aggregate_with_bonds(
                atom_embeds, adj_list, edge_batch_idx, masked_elements
            )
        else:
            # Simple atom-only aggregation  
            h_mol = self._aggregate_atoms_only(atom_embeds, masked_elements)
            
        # Pass through graph encoder
        h_G = self.graph_encoder(h_mol)  # [B, graph_embed_dim]
        
        return h_G
    
    def _aggregate_atoms_only(
        self, 
        atom_embeds: Tensor,  # [B, N, atom_embed_dim]
        masked_elements: Optional[Tensor] = None,  # [B, N]
    ) -> Tensor:
        """Simple mean aggregation over atoms."""
        if masked_elements is not None:
            # Count non-masked atoms for proper averaging
            counts = (~masked_elements).sum(dim=1, keepdim=True).float()  # [B, 1]
            counts = torch.clamp(counts, min=1.0)  # Avoid division by zero
            h_mol = atom_embeds.sum(dim=1) / counts  # [B, atom_embed_dim]
        else:
            h_mol = atom_embeds.mean(dim=1)  # [B, atom_embed_dim]
        
        return h_mol
    
    def _aggregate_with_bonds(
        self,
        atom_embeds: Tensor,  # [B, N, atom_embed_dim] 
        adj_list: Tensor,  # [E, 2]
        edge_batch_idx: Tensor,  # [E]
        masked_elements: Optional[Tensor] = None,  # [B, N]
    ) -> Tensor:
        """Aggregate with bond information included."""
        B = atom_embeds.shape[0]
        
        # For now, use simple atom aggregation + bond count feature
        h_atoms = self._aggregate_atoms_only(atom_embeds, masked_elements)
        
        # Add simple bond statistics
        bond_counts = torch.zeros(B, device=atom_embeds.device)
        for b in range(B):
            bond_counts[b] = (edge_batch_idx == b).sum().float()
        
        # Simple bond features (can be made more sophisticated)
        bond_features = torch.zeros(B, 16, device=atom_embeds.device)
        bond_features[:, 0] = bond_counts  # Total bond count
        
        # Concatenate atom and bond features
        h_mol = torch.cat([h_atoms, bond_features], dim=-1)
        
        return h_mol 