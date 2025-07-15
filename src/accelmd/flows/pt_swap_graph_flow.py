"""Graph-conditioned normalizing flow for PT swap proposals.

This module implements the complete graph-conditioned architecture that maps
molecular_structure + coordinates to coordinates using graph neural networks
and distance-based attention mechanisms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple, Dict, Any

from ..targets import build_target
from .graph_coupling_layer import GraphNVPCouplingLayer

__all__ = ["PTSwapGraphFlow"]


class PTSwapGraphFlow(nn.Module):
    """Graph-conditioned normalizing flow for PT swap proposals.
    
    Implements the complete Flow-PT architecture with:
    - Global graph embedding h_G = GraphEmb(G)
    - Node features h_i = f_θ(x_i, a_i, h_G) 
    - Distance-based attention AGG_θ with w_ij = exp(-||x_i - x_j||^2 / σ^2)
    - Sequential coupling layers with alternating phases
    
    This is the enriched representation architecture expected to work better
    than simple coordinate-to-coordinate flows.
    
    Parameters
    ----------
    num_layers : int
        Number of graph coupling layers
    atom_vocab_size : int
        Number of unique atom types (e.g., 4 for H,C,N,O)
    atom_embed_dim : int
        Dimension of atom type embeddings
    graph_embed_dim : int
        Dimension of global graph embedding (0 for ablation)
    node_feature_dim : int
        Dimension of per-atom features
    hidden_dim : int
        Hidden dimension for MLPs
    attention_lengthscales : List[float]
        Attention lengthscales for multi-head mechanism
    source_temperature : float
        Source temperature for physics base distribution
    target_temperature : float  
        Target temperature for physics base distribution
    target_name : str
        Name of the Boltzmann target distribution
    target_kwargs : Dict[str, Any]
        Additional arguments for target distribution
    device : str
        Compute device
    """
    
    def __init__(
        self,
        num_layers: int = 8,
        atom_vocab_size: int = 4,
        atom_embed_dim: int = 32,
        graph_embed_dim: int = 64,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        attention_lengthscales: List[float] = [1.0, 2.0, 4.0],
        source_temperature: float = 1.0,
        target_temperature: float = 1.5,
        target_name: str = "aldp",
        target_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.atom_vocab_size = atom_vocab_size
        self.graph_embed_dim = graph_embed_dim
        self.source_temp = source_temperature
        self.target_temp = target_temperature
        
        # Build Boltzmann target distributions
        if target_kwargs is None:
            target_kwargs = {}
            
        self.source_target = build_target(
            target_name, device=device, **target_kwargs
        )
        self.target_target = build_target(
            target_name, device=device, **target_kwargs
        )
        
        # Graph coupling layers with alternating phases
        self.coupling_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            phase = layer_idx % 2  # Alternate between 0 and 1
            
            layer = GraphNVPCouplingLayer(
                phase=phase,
                atom_vocab_size=atom_vocab_size,
                atom_embed_dim=atom_embed_dim,
                graph_embed_dim=graph_embed_dim,
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                attention_lengthscales=attention_lengthscales,
                use_graph_embedding=(graph_embed_dim > 0),
            )
            self.coupling_layers.append(layer)
    
    def forward(
        self,
        coordinates: Tensor,  # [B, N, 3] input coordinates
        atom_types: Tensor,   # [B, N] atom type indices
        adj_list: Optional[Tensor] = None,  # [E, 2] edge connectivity
        edge_batch_idx: Optional[Tensor] = None,  # [E] batch indices
        masked_elements: Optional[Tensor] = None,  # [B, N] padding mask
        reverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply graph-conditioned flow transformation.
        
        Parameters
        ----------
        coordinates : Tensor, shape [B, N, 3]
            Input molecular coordinates
        atom_types : Tensor, shape [B, N]
            Atom type indices
        adj_list : Tensor, shape [E, 2], optional
            Edge connectivity list
        edge_batch_idx : Tensor, shape [E], optional
            Batch indices for edges
        masked_elements : Tensor, shape [B, N], optional
            Padding mask (True for padding atoms)
        reverse : bool
            Whether to apply reverse (sampling) direction
            
        Returns
        -------
        output_coords : Tensor, shape [B, N, 3]
            Transformed coordinates
        total_log_det : Tensor, shape [B]
            Total log-determinant across all layers
        """
        current_coords = coordinates
        total_log_det = torch.zeros(coordinates.shape[0], device=coordinates.device)
        
        # Apply coupling layers in forward or reverse order
        layers = self.coupling_layers
        if reverse:
            layers = reversed(layers)
            
        for layer in layers:
            current_coords, log_det = layer(
                coordinates=current_coords,
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                reverse=reverse,
            )
            total_log_det += log_det
            
        return current_coords, total_log_det
    
    def log_likelihood(
        self,
        source_coords: Tensor,  # [B, N, 3] source configuration
        target_coords: Tensor,  # [B, N, 3] target configuration  
        atom_types: Tensor,     # [B, N] atom types
        adj_list: Optional[Tensor] = None,
        edge_batch_idx: Optional[Tensor] = None,
        masked_elements: Optional[Tensor] = None,
        direction: str = "forward",  # "forward" or "reverse"
    ) -> Tensor:
        """Compute log-likelihood for bidirectional training.
        
        Parameters
        ----------
        source_coords : Tensor, shape [B, N, 3]
            Source temperature coordinates
        target_coords : Tensor, shape [B, N, 3]  
            Target temperature coordinates
        atom_types : Tensor, shape [B, N]
            Atom type indices
        adj_list : Tensor, optional
            Edge connectivity
        edge_batch_idx : Tensor, optional
            Batch indices for edges
        masked_elements : Tensor, optional  
            Padding mask
        direction : str
            "forward" for low→high temp, "reverse" for high→low temp
            
        Returns
        -------
        log_likelihood : Tensor, shape [B]
            Log-likelihood values for each sample
        """
        if direction == "forward":
            # Forward: p(target | source) using source base distribution
            predicted_coords, log_det = self.forward(
                coordinates=source_coords,
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                reverse=False,  # Forward flow
            )
            
            # Physics-informed likelihood with source temperature base
            log_prob_base = self._log_boltzmann_masked(
                coordinates=source_coords,
                temperature=self.source_temp,
                masked_elements=masked_elements,
            )
            
            # MSE term for flow accuracy (optional regularization)
            mse_loss = ((predicted_coords - target_coords) ** 2).sum(dim=(1, 2))
            if masked_elements is not None:
                # Account for variable molecule sizes
                valid_atoms = (~masked_elements).sum(dim=1).float()  # [B]
                mse_loss = mse_loss / (valid_atoms * 3)  # Normalize by degrees of freedom
            
            log_likelihood = log_prob_base + log_det - 0.1 * mse_loss  # Small MSE weight
            
        elif direction == "reverse":
            # Reverse: p(source | target) using target base distribution
            predicted_coords, log_det = self.forward(
                coordinates=target_coords,
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                reverse=True,  # Reverse flow
            )
            
            # Physics-informed likelihood with target temperature base
            log_prob_base = self._log_boltzmann_masked(
                coordinates=target_coords,
                temperature=self.target_temp,
                masked_elements=masked_elements,
            )
            
            # MSE term
            mse_loss = ((predicted_coords - source_coords) ** 2).sum(dim=(1, 2))
            if masked_elements is not None:
                valid_atoms = (~masked_elements).sum(dim=1).float()
                mse_loss = mse_loss / (valid_atoms * 3)
                
            log_likelihood = log_prob_base + log_det - 0.1 * mse_loss
            
        else:
            raise ValueError(f"Unknown direction: {direction}")
            
        return log_likelihood
    
    def sample_proposal(
        self,
        source_coords: Tensor,
        atom_types: Tensor,
        adj_list: Optional[Tensor] = None,
        edge_batch_idx: Optional[Tensor] = None,
        masked_elements: Optional[Tensor] = None,
        direction: str = "forward",
    ) -> Tensor:
        """Generate swap proposal coordinates.
        
        Parameters
        ----------
        source_coords : Tensor, shape [B, N, 3]
            Source coordinates at temperature T1
        atom_types : Tensor, shape [B, N]
            Atom type indices
        adj_list : Tensor, optional
            Edge connectivity
        edge_batch_idx : Tensor, optional
            Batch indices for edges
        masked_elements : Tensor, optional
            Padding mask
        direction : str
            "forward" for T_low→T_high, "reverse" for T_high→T_low
            
        Returns
        -------
        proposal_coords : Tensor, shape [B, N, 3]
            Proposed coordinates at target temperature
        """
        with torch.no_grad():
            if direction == "forward":
                # T_low → T_high: use forward flow
                proposal_coords, _ = self.forward(
                    coordinates=source_coords,
                    atom_types=atom_types,
                    adj_list=adj_list,
                    edge_batch_idx=edge_batch_idx,
                    masked_elements=masked_elements,
                    reverse=False,
                )
            elif direction == "reverse":
                # T_high → T_low: use reverse flow
                proposal_coords, _ = self.forward(
                    coordinates=source_coords,
                    atom_types=atom_types,
                    adj_list=adj_list,
                    edge_batch_idx=edge_batch_idx,
                    masked_elements=masked_elements,
                    reverse=True,
                )
            else:
                raise ValueError(f"Unknown direction: {direction}")
                
        return proposal_coords
    
    def _log_boltzmann_masked(
        self,
        coordinates: Tensor,  # [B, N, 3]
        temperature: float,
        masked_elements: Optional[Tensor] = None,  # [B, N]
    ) -> Tensor:
        """Compute masked Boltzmann log-probability."""
        B = coordinates.shape[0]
        
        # Flatten coordinates while masking padding
        if masked_elements is not None:
            coords_masked = coordinates * (~masked_elements).unsqueeze(-1)
            coords_flat = coords_masked.view(B, -1)  # [B, 3N]
        else:
            coords_flat = coordinates.view(B, -1)
            
        # Boltzmann log-probability at given temperature
        log_probs = []
        for i in range(B):
            if masked_elements is not None:
                # Extract only non-masked coordinates
                valid_mask = ~masked_elements[i]  # [N]
                valid_coords = coordinates[i][valid_mask].view(-1)  # [3*n_valid]
                log_prob = self.source_target.log_prob(valid_coords, temperature)
            else:
                log_prob = self.source_target.log_prob(coords_flat[i], temperature)
            log_probs.append(log_prob)
            
        return torch.stack(log_probs)
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"PTSwapGraphFlow(layers={self.num_layers}, "
            f"graph_embed_dim={self.graph_embed_dim}, "
            f"T_source={self.source_temp:.3f}, "
            f"T_target={self.target_temp:.3f})"
        ) 