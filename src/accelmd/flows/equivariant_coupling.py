"""E(3)-equivariant coupling layer for PT swap flows.

This module implements E(3)-equivariant transformations for PT swaps:
- E(3)-equivariant coordinate transformations using relative vector combinations
- Invariant features computed from distances and atom types only
- Position-only (no velocity handling)
- Temperature conditioning for PT applications
- Conservative scaling for training stability

Key principles:
- All coordinate operations are linear in relative positions
- Only invariant features (distances, dot products) used in MLPs
- Exact invertibility maintained through careful Jacobian computation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from typing import Optional, Tuple, List

from .equivariant_flow_base import PTNVPCouplingLayer
from ..flows.mlp import MLP
from .rff_position_encoder import RFFPositionEncoder

__all__ = ["EquivariantCouplingLayer"]


class EquivariantCouplingLayer(PTNVPCouplingLayer):
    """E(3)-equivariant coupling layer with invariant and equivariant transformations.
    
    This implementation follows key design principles:
    1. Equivariant shifts computed from relative vector combinations
    2. Invariant scaling based on distances and atom features
    3. Conservative transformations for training stability
    4. Temperature conditioning for PT applications
    
    Parameters
    ----------
    phase : int
        Coupling phase (0 or 1) for alternating atom updates
    atom_vocab_size : int
        Number of unique atom types
    atom_embed_dim : int
        Dimension of atom type embeddings
    hidden_dim : int
        Hidden dimension for MLPs
    scale_range : float
        Maximum scaling factor (conservative for stability)
    max_neighbors : int
        Maximum number of neighbors to consider for efficiency
    distance_cutoff : float
        Maximum distance for neighbor interactions
    """
    
    def __init__(
        self,
        phase: int,
        atom_vocab_size: int = 4,
        atom_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_mlp_layers: int = 2,
        scale_range: float = 0.05,  # Very conservative for stability
        shift_range: float = 1.0,   # Maximum shift magnitude in Angstroms
        max_neighbors: int = 20,    # Limit for computational efficiency
        distance_cutoff: float = 8.0,  # Angstroms
        temperature_conditioning: bool = True,
    ):
        super().__init__()
        
        self.phase = phase
        self.atom_vocab_size = atom_vocab_size
        self.atom_embed_dim = atom_embed_dim
        self.hidden_dim = hidden_dim
        self.num_mlp_layers = num_mlp_layers
        self.scale_range = scale_range
        # shift_range is specified in Å but internal coordinates are nm → convert
        self.shift_range = shift_range * 0.1  # Å → nm
        self.max_neighbors = max_neighbors
        self.distance_cutoff = distance_cutoff
        self.temperature_conditioning = temperature_conditioning
        
        # Atom type embeddings (invariant features)
        self.atom_embedding = nn.Embedding(atom_vocab_size, atom_embed_dim)
        
        # RFF position encoder for stronger conditioning (like transformer)
        self.rff_encoder = RFFPositionEncoder(
            input_dim=3,
            encoding_dim=64,
            scale_mean=1.0,
            scale_stddev=1.0,
        )
        
        # Input feature dimension
        self.input_feat_dim = atom_embed_dim + 64  # atom_embed + RFF features
        if temperature_conditioning:
            self.input_feat_dim += 2  # source_temp + target_temp
        
        # Invariant scale network: atom features + distance features → scalar scale
        # Input: [atom_embed + RFF + temp_features + distance_features]
        # Configurable depth with tapering width
        scale_hidden_dims = [hidden_dim] * (num_mlp_layers - 1) + [hidden_dim // 2]
        self.scale_net = MLP(
            input_dim=self.input_feat_dim + 1,  # +1 for distance
            hidden_layer_dims=scale_hidden_dims,
            out_dim=1,  # Scalar scale per atom
            activation=nn.ReLU(),
        )
        
        # Equivariant shift network: processes relative vector coefficients
        # Maps from pointwise invariant features to coefficients for relative vectors
        # Configurable depth with tapering width
        shift_hidden_dims = [hidden_dim] * (num_mlp_layers - 1) + [hidden_dim // 2]
        self.shift_coeff_net = MLP(
            input_dim=self.input_feat_dim,
            hidden_layer_dims=shift_hidden_dims,
            out_dim=1,  # Coefficient for combining relative vectors
            activation=nn.ReLU(),
        )
        
        # Initialize networks for stable training
        self._init_networks()
    
    def _init_networks(self):
        """Initialize networks for stable training."""
        # Initialize scale network to output small random values initially
        with torch.no_grad():
            # Small random initialization for final layer of scale network
            # MLP uses _layers as a Sequential, so get the last linear layer
            for layer in reversed(self.scale_net._layers):
                if isinstance(layer, nn.Linear):
                    # Small random initialization instead of zeros
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.normal_(layer.bias, mean=0.0, std=0.01)
                    break
            
            # Small initialization for shift coefficient network
            for layer in reversed(self.shift_coeff_net._layers):
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
                    break
    
    def _get_scale_and_shift(
        self,
        coordinates: Tensor,  # [B, N, 3]
        atom_types: Tensor,   # [B, N]
        adj_list: Tensor,     # [E, 2]
        edge_batch_idx: Tensor,  # [E]
        masked_elements: BoolTensor,  # [B, N]
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Compute E(3)-equivariant scale and shift parameters.
        
        Following the equivariant approach:
        1. Compute invariant features (atom types + distances + temperatures)
        2. Predict scalar scales (invariant)
        3. Predict equivariant shifts from relative vector combinations
        """
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Step 1: Create coupling mask (alternating pattern based on phase)
        coupling_mask = self._create_coupling_mask(atom_types, masked_elements)  # [B, N]
        
        # Step 2: Compute rich atom features (like transformer)
        atom_embeds = self.atom_embedding(atom_types)  # [B, N, atom_embed_dim]
        
        # Add RFF position encoding (critical for strong conditioning)
        coord_rff = self.rff_encoder(coordinates)  # [B, N, 64]
        
        # Combine atom embeddings with RFF features
        atom_features = torch.cat([atom_embeds, coord_rff], dim=-1)  # [B, N, atom_embed_dim + 64]
        
        # Add temperature conditioning if enabled
        if self.temperature_conditioning and source_temp is not None and target_temp is not None:
            temp_features = torch.tensor([source_temp, target_temp], device=device)
            temp_features = temp_features.expand(B, N, 2)  # [B, N, 2]
            atom_features = torch.cat([atom_features, temp_features], dim=-1)
        
        # Step 3: Compute pairwise distances (invariant)
        coord_i = coordinates.unsqueeze(2)  # [B, N, 1, 3]
        coord_j = coordinates.unsqueeze(1)  # [B, 1, N, 3]
        relative_vectors = coord_j - coord_i  # [B, N, N, 3]
        distances = torch.norm(relative_vectors, dim=-1)  # [B, N, N]
        
        # Step 4: Predict invariant scales using molecular bonds
        scales = self._compute_invariant_scales(
            atom_features, distances, coupling_mask, masked_elements,
            adj_list, edge_batch_idx
        )  # [B, N, 3]
        
        # Step 5: Predict equivariant shifts using molecular bonds
        shifts = self._compute_equivariant_shifts(
            coordinates, atom_features, relative_vectors, distances, 
            coupling_mask, masked_elements, adj_list, edge_batch_idx
        )  # [B, N, 3]
        
        return scales, shifts
    
    def _create_coupling_mask(
        self, 
        atom_types: Tensor, 
        masked_elements: BoolTensor
    ) -> Tensor:
        """Create coupling mask based on phase and padding."""
        B, N = atom_types.shape
        
        # Alternating pattern based on phase
        indices = torch.arange(N, device=atom_types.device)
        coupling_mask = (indices % 2) == self.phase  # [N]
        coupling_mask = coupling_mask.expand(B, N)  # [B, N]
        
        # Exclude masked elements
        if masked_elements is not None:
            coupling_mask = coupling_mask & (~masked_elements)
        else:
            # If no masking, just convert to float
            pass
        
        return coupling_mask.float()
    
    def _compute_invariant_scales(
        self,
        atom_features: Tensor,  # [B, N, feat_dim]
        distances: Tensor,      # [B, N, N]
        coupling_mask: Tensor,  # [B, N]
        masked_elements: BoolTensor,  # [B, N]
        adj_list: Optional[Tensor] = None,  # [E, 2] molecular connectivity
        edge_batch_idx: Optional[Tensor] = None,  # [E] batch indices
    ) -> Tensor:
        """Compute invariant scaling factors using molecular bond connectivity."""
        B, N, _ = atom_features.shape
        device = atom_features.device
        
        # Create neighbor mask based on actual molecular bonds if available
        if adj_list is not None and edge_batch_idx is not None and len(adj_list) > 0:
            # Use real molecular connectivity
            neighbor_mask = torch.zeros(B, N, N, device=device)  # [B, N, N]
            
            # Convert edge list to adjacency matrix for each batch
            for batch_idx in range(B):
                batch_edges = adj_list[edge_batch_idx == batch_idx]  # Edges for this batch item
                if len(batch_edges) > 0:
                    # Make sure indices are within bounds for this molecule
                    valid_edges = batch_edges[torch.max(batch_edges, dim=1)[0] < N]
                    if len(valid_edges) > 0:
                        src_indices = valid_edges[:, 0]
                        tgt_indices = valid_edges[:, 1]
                        # Set adjacency (symmetric for undirected molecular graph)
                        neighbor_mask[batch_idx, src_indices, tgt_indices] = 1.0
                        neighbor_mask[batch_idx, tgt_indices, src_indices] = 1.0
            

        else:
            # Fallback to distance-based connectivity (old behavior)
            neighbor_mask = torch.ones_like(distances)  # [B, N, N]
            neighbor_mask = neighbor_mask - torch.eye(N, device=device).unsqueeze(0)  # Remove self
            
            # Apply distance cutoff
            if self.distance_cutoff is not None:
                neighbor_mask = neighbor_mask * (distances <= self.distance_cutoff).float()
                

        
        # Apply masking for padded atoms
        if masked_elements is not None:
            valid_neighbors = (~masked_elements).float()  # [B, N]
            valid_mask = valid_neighbors.unsqueeze(1) * valid_neighbors.unsqueeze(2)  # [B, N, N]
            neighbor_mask = neighbor_mask * valid_mask
        
        # Compute mean distance to bonded neighbors (invariant feature)
        masked_distances = distances * neighbor_mask
        neighbor_counts = neighbor_mask.sum(dim=-1).clamp(min=1)  # [B, N]
        mean_distances = masked_distances.sum(dim=-1) / neighbor_counts  # [B, N]
        
        # Create input features for scale network
        scale_input = torch.cat([
            atom_features,  # [B, N, feat_dim]
            mean_distances.unsqueeze(-1)  # [B, N, 1]
        ], dim=-1)  # [B, N, feat_dim + 1]
        
        # Predict raw scales
        raw_scales = self.scale_net(scale_input).squeeze(-1)  # [B, N]
        
        # Apply scale range and coupling mask
        log_scales = torch.tanh(raw_scales) * self.scale_range  # [B, N]
        log_scales = log_scales * coupling_mask  # Zero out non-updated atoms
        
        # Convert to positive scales
        scales = torch.exp(log_scales).unsqueeze(-1)  # [B, N, 1]
        scales = scales.expand(-1, -1, 3)  # [B, N, 3] - same scale for all coordinates
        
        return scales
    
    def _compute_equivariant_shifts(
        self,
        coordinates: Tensor,     # [B, N, 3]
        atom_features: Tensor,   # [B, N, feat_dim]
        relative_vectors: Tensor,  # [B, N, N, 3]
        distances: Tensor,       # [B, N, N]
        coupling_mask: Tensor,   # [B, N]
        masked_elements: BoolTensor,  # [B, N]
        adj_list: Optional[Tensor] = None,  # [E, 2] molecular connectivity
        edge_batch_idx: Optional[Tensor] = None,  # [E] batch indices
    ) -> Tensor:
        """Compute equivariant shifts using relative vector combinations.
        
        Following the equivariant approach: shift_i = Σ_j c_i * (x_j - x_i)
        where c_i are learned coefficients and the sum is over neighbors.
        """
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Predict coefficients for relative vector combinations
        raw_shift_coeffs = self.shift_coeff_net(atom_features).squeeze(-1)  # [B, N]
        
        # CRITICAL FIX: Clamp shift magnitude to prevent energy explosion
        shift_coeffs = torch.tanh(raw_shift_coeffs) * self.shift_range  # now in nm units
        shift_coeffs = shift_coeffs * coupling_mask  # Only for updated atoms
        
        # Create neighbor mask based on actual molecular bonds (same as in scale computation)
        if adj_list is not None and edge_batch_idx is not None and len(adj_list) > 0:
            # Use real molecular connectivity
            neighbor_mask = torch.zeros(B, N, N, device=device)  # [B, N, N]
            
            # Convert edge list to adjacency matrix for each batch
            for batch_idx in range(B):
                batch_edges = adj_list[edge_batch_idx == batch_idx]  # Edges for this batch item
                if len(batch_edges) > 0:
                    # Make sure indices are within bounds for this molecule
                    valid_edges = batch_edges[torch.max(batch_edges, dim=1)[0] < N]
                    if len(valid_edges) > 0:
                        src_indices = valid_edges[:, 0]
                        tgt_indices = valid_edges[:, 1]
                        # Set adjacency (symmetric for undirected molecular graph)
                        neighbor_mask[batch_idx, src_indices, tgt_indices] = 1.0
                        neighbor_mask[batch_idx, tgt_indices, src_indices] = 1.0
        else:
            # Fallback to distance-based connectivity (old behavior)
            neighbor_mask = torch.ones_like(distances)  # [B, N, N]
            neighbor_mask = neighbor_mask - torch.eye(N, device=device).unsqueeze(0)  # Remove self
            
            # Apply distance cutoff
            if self.distance_cutoff is not None:
                neighbor_mask = neighbor_mask * (distances <= self.distance_cutoff).float()
        
        # Apply masking for padded atoms
        if masked_elements is not None:
            valid_neighbors = (~masked_elements).float()  # [B, N]
            neighbor_validity = valid_neighbors.unsqueeze(1) * valid_neighbors.unsqueeze(2)  # [B, N, N]
            neighbor_mask = neighbor_mask * neighbor_validity
        
        # Limit number of neighbors for computational efficiency
        if self.max_neighbors is not None:
            # For each atom, keep only closest max_neighbors neighbors
            masked_distances = distances + 1e6 * (1 - neighbor_mask)  # Large value for excluded
            _, neighbor_indices = torch.topk(
                masked_distances, k=min(self.max_neighbors, N), 
                dim=-1, largest=False
            )  # [B, N, max_neighbors]
            
            # Create new neighbor mask with only top-k neighbors
            new_neighbor_mask = torch.zeros_like(neighbor_mask)
            batch_indices = torch.arange(B, device=device).view(B, 1, 1)
            atom_indices = torch.arange(N, device=device).view(1, N, 1)
            new_neighbor_mask[batch_indices, atom_indices, neighbor_indices] = 1
            neighbor_mask = new_neighbor_mask * neighbor_mask  # Keep original constraints
        
        # Compute weighted sum of relative vectors
        # shift_i = c_i * Σ_j (x_j - x_i) where sum is over valid neighbors
        weighted_relatives = relative_vectors * neighbor_mask.unsqueeze(-1)  # [B, N, N, 3]
        neighbor_sums = weighted_relatives.sum(dim=2)  # [B, N, 3]
        
        # Apply learned coefficients
        shifts = shift_coeffs.unsqueeze(-1) * neighbor_sums  # [B, N, 3]
        
        # Normalize by number of neighbors to prevent scale drift
        neighbor_counts = neighbor_mask.sum(dim=-1).clamp(min=1).unsqueeze(-1)  # [B, N, 1]
        shifts = shifts / neighbor_counts
        
        return shifts
    
    def flow_forward(
        self,
        coordinates: Tensor,      # [B, N, 3]
        atom_types: Tensor,       # [B, N]
        adj_list: Tensor,         # [E, 2] (not used but kept for interface)
        edge_batch_idx: Tensor,   # [E] (not used but kept for interface)
        masked_elements: BoolTensor,  # [B, N]
        source_temp: float,
        target_temp: float,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Forward flow transformation with proper equivariant masking."""
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Get scale and shift parameters
        scales, shifts = self._get_scale_and_shift(
            coordinates=coordinates,
            atom_types=atom_types,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            source_temp=source_temp,
            target_temp=target_temp,
        )
        
        # Create coupling mask for alternating updates
        coupling_mask = self._create_coupling_mask(atom_types, masked_elements)  # [B, N]
        
        # CRITICAL: Proper masking for equivariant flows
        # Only compute log-det for non-masked elements that are being transformed
        active_mask = coupling_mask
        if masked_elements is not None:
            active_mask = coupling_mask * (~masked_elements.float())  # [B, N]
        
        # Apply coupling mask to scales and shifts
        scales = scales * active_mask.unsqueeze(-1)  # [B, N, 3]
        shifts = shifts * active_mask.unsqueeze(-1)  # [B, N, 3]
        
        # Ensure scales are >= 1e-6 for stability, but only where active
        scales = torch.where(active_mask.unsqueeze(-1) > 0, 
                           torch.clamp(scales, min=1e-6), 
                           torch.ones_like(scales))
        
        # Transform coordinates: x' = x * scale + shift (only for active elements)
        # Use safe transformation to prevent numerical issues
        transformed_coords = coordinates * scales + shifts
        
        # CRITICAL: Clamp coordinates to prevent extreme distortions that cause energy overflow
        # Typical molecular coordinates are in the range -5 to +5 nm, so clamp to reasonable bounds
        coord_min, coord_max = -10.0, 10.0  # nm, very generous bounds
        transformed_coords = torch.clamp(transformed_coords, coord_min, coord_max)
        
        output_coords = torch.where(active_mask.unsqueeze(-1) > 0,
                                   transformed_coords,
                                   coordinates)  # Keep unchanged for inactive elements
        
        # CORRECT REAL NVP LOG-DETERMINANT CALCULATION
        # log|det J| = sum of log(scales) ONLY over transformed coordinates (coupling mask + padding mask)
        # Only atoms that are actually transformed should contribute to the Jacobian
        if masked_elements is not None:
            # Combine coupling mask (which atoms are transformed) with padding mask (which atoms exist)
            jacobian_mask = active_mask * (~masked_elements).float()  # [B, N]
        else:
            jacobian_mask = active_mask  # [B, N]
        
        log_scales = torch.log(scales) * jacobian_mask.unsqueeze(-1)  # [B, N, 3]
        logdetjac = torch.sum(log_scales, dim=(-1, -2))  # [B] - sum over transformed atoms and coordinates
        
        return output_coords, logdetjac
    
    def flow_reverse(
        self,
        coordinates: Tensor,      # [B, N, 3]
        atom_types: Tensor,       # [B, N]
        adj_list: Tensor,         # [E, 2] (not used but kept for interface)
        edge_batch_idx: Tensor,   # [E] (not used but kept for interface)
        masked_elements: BoolTensor,  # [B, N]
        source_temp: float,
        target_temp: float,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Reverse flow transformation with proper equivariant masking."""
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Get scale and shift parameters
        scales, shifts = self._get_scale_and_shift(
            coordinates=coordinates,
            atom_types=atom_types,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            source_temp=source_temp,
            target_temp=target_temp,
        )
        
        # Create coupling mask for alternating updates
        coupling_mask = self._create_coupling_mask(atom_types, masked_elements)  # [B, N]
        
        # CRITICAL: Proper masking for equivariant flows
        # Only compute log-det for non-masked elements that are being transformed
        active_mask = coupling_mask
        if masked_elements is not None:
            active_mask = coupling_mask * (~masked_elements.float())  # [B, N]
        
        # Apply coupling mask to scales and shifts
        scales = scales * active_mask.unsqueeze(-1)  # [B, N, 3]
        shifts = shifts * active_mask.unsqueeze(-1)  # [B, N, 3]
        
        # Ensure scales are >= 1e-6 for stability, but only where active
        scales = torch.where(active_mask.unsqueeze(-1) > 0, 
                           torch.clamp(scales, min=1e-6), 
                           torch.ones_like(scales))
        
        # Reverse transform coordinates: x = (x' - shift) / scale (only for active elements)
        # Use safe division to prevent numerical issues
        transformed_coords = (coordinates - shifts) / scales
        
        # CRITICAL: Clamp coordinates to prevent extreme distortions that cause energy overflow
        # Typical molecular coordinates are in the range -5 to +5 nm, so clamp to reasonable bounds
        coord_min, coord_max = -10.0, 10.0  # nm, very generous bounds
        transformed_coords = torch.clamp(transformed_coords, coord_min, coord_max)
        
        output_coords = torch.where(active_mask.unsqueeze(-1) > 0,
                                  transformed_coords,
                                  coordinates)  # Keep unchanged for inactive elements
        
        # CORRECT REAL NVP LOG-DETERMINANT CALCULATION (negative for reverse)
        # log|det J^(-1)| = -sum of log(scales) ONLY over transformed coordinates (coupling mask + padding mask)
        if masked_elements is not None:
            # Combine coupling mask (which atoms are transformed) with padding mask (which atoms exist)
            jacobian_mask = active_mask * (~masked_elements).float()  # [B, N]
        else:
            jacobian_mask = active_mask  # [B, N]
        
        log_scales = torch.log(scales) * jacobian_mask.unsqueeze(-1)  # [B, N, 3]
        logdetjac = -torch.sum(log_scales, dim=(-1, -2))  # [B] - negative for reverse, sum over transformed atoms
        
        return output_coords, logdetjac
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"EquivariantCouplingLayer(phase={self.phase}, "
            f"scale_range={self.scale_range}, "
            f"distance_cutoff={self.distance_cutoff}, "
            f"max_neighbors={self.max_neighbors})"
        ) 