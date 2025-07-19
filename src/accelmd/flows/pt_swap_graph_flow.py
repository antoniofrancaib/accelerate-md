"""Dimension-agnostic message-passing flow for PT swap proposals.

This module implements the supervisor's suggested approach:
- Message passing over molecular bonds (edges)
- Atom types (C,H,O,N) as node features
- No complex attention mechanisms
- Fully dimension-agnostic (works for any molecular size)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple, Dict, Any

from ..targets import build_target
from .graph_coupling_layer import MessagePassingCouplingLayer

__all__ = ["PTSwapGraphFlow"]


class PTSwapGraphFlow(nn.Module):
    """Dimension-agnostic message-passing flow for PT swap proposals.
    
    Implements your supervisor's suggested architecture:
    - Simple message passing on molecular bonds
    - Atom types as node features (no complex embeddings)
    - Fully size-agnostic (works for AK ↔ AA ↔ any peptide)
    - Clean and efficient design
    
    Parameters
    ----------
    num_layers : int
        Number of coupling layers
    atom_vocab_size : int
        Number of unique atom types (e.g., 4 for H,C,N,O)
    atom_embed_dim : int
        Dimension of atom type embeddings
    hidden_dim : int
        Hidden dimension for message passing and MLPs
    num_mp_layers : int
        Number of message-passing layers per coupling layer
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
        atom_embed_dim: int = 64,
        hidden_dim: int = 128,
        num_mp_layers: int = 2,
        source_temperature: float = 1.0,
        target_temperature: float = 1.5,
        target_name: str = "aldp",
        target_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.atom_vocab_size = atom_vocab_size
        self.source_temp = source_temperature
        self.target_temp = target_temperature
        
        # Build Boltzmann target distributions with correct temperatures
        if target_kwargs is None:
            target_kwargs = {}
            
        # Source target (low temperature)
        source_kwargs = target_kwargs.copy()
        source_kwargs['temperature'] = source_temperature
        self.source_target = build_target(
            target_name, device=device, **source_kwargs
        )
        
        # Target target (high temperature)  
        target_kwargs_copy = target_kwargs.copy()
        target_kwargs_copy['temperature'] = target_temperature
        self.target_target = build_target(
            target_name, device=device, **target_kwargs_copy
        )
        
        # Aliases for training code compatibility
        self.base_low = self.source_target   # Low temperature base distribution
        self.base_high = self.target_target  # High temperature base distribution
        
        # Message-passing coupling layers with alternating phases
        self.coupling_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            phase = layer_idx % 2  # Alternate between 0 and 1
            
            layer = MessagePassingCouplingLayer(
                phase=phase,
                atom_vocab_size=atom_vocab_size,
                atom_embed_dim=atom_embed_dim,
                hidden_dim=hidden_dim,
                num_mp_layers=num_mp_layers,
                # Enhanced parameters for better expressivity
                scale_range=0.5,  # Allow ±50% scaling
                deeper_networks=True,  # Use deeper MLPs
                temperature_conditioning=True,  # Enable temperature awareness
            )
            self.coupling_layers.append(layer)
            
        # Move entire model to specified device
        self.to(device)
    
    def forward(
        self,
        coordinates: Tensor,  # [B, N, 3] input coordinates
        atom_types: Optional[Tensor] = None,   # [B, N] atom type indices (required)
        adj_list: Optional[Tensor] = None,  # [E, 2] edge connectivity (required)
        edge_batch_idx: Optional[Tensor] = None,  # [E] batch indices
        masked_elements: Optional[Tensor] = None,  # [B, N] padding mask
        reverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply dimension-agnostic message-passing flow transformation.
        
        Parameters
        ----------
        coordinates : Tensor, shape [B, N, 3]
            Input molecular coordinates (N can be any size!)
        atom_types : Tensor, shape [B, N]
            Atom type indices (required for molecular structure)
        adj_list : Tensor, shape [E, 2]
            Edge connectivity list (required for molecular bonds)
        edge_batch_idx : Tensor, shape [E], optional
            Batch indices for edges (auto-generated if None)
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
        # Validate required inputs
        if atom_types is None:
            raise ValueError("atom_types is required for message-passing flow. Dataset should provide real molecular data.")
        if adj_list is None:
            raise ValueError("adj_list is required for message-passing flow. Dataset should provide real molecular bond connectivity.")
        
        # Generate edge batch indices if not provided
        if edge_batch_idx is None:
            B = coordinates.shape[0]
            n_edges_per_mol = adj_list.shape[0]  # Edges in the single molecule template
            
            # Create batched adjacency list: replicate single molecule for each batch
            # adj_list is [E, 2] for single molecule -> [B*E, 2] for batch
            adj_list_batched = torch.cat([adj_list for _ in range(B)], dim=0)
            
            # Update adj_list to use the batched version
            adj_list = adj_list_batched
            
            # Create proper edge batch indices: B copies of n_edges_per_mol
            edge_batch_idx = torch.repeat_interleave(
                torch.arange(B, device=coordinates.device), 
                n_edges_per_mol
            )
        
        current_coords = coordinates
        model_device = next(self.parameters()).device
        total_log_det = torch.zeros(coordinates.shape[0], device=model_device)
        
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
                # Pass temperature information for conditioning
                source_temp=self.source_temp,
                target_temp=self.target_temp,
            )
            # Ensure log_det is on the same device as total_log_det
            total_log_det += log_det.to(total_log_det.device)
            
        return current_coords, total_log_det
    
    def inverse(
        self, 
        coordinates: Tensor, 
        atom_types: Optional[Tensor] = None, 
        adj_list: Optional[Tensor] = None,
        edge_batch_idx: Optional[Tensor] = None,
        masked_elements: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Inverse flow transformation for trainer compatibility.
        
        Parameters
        ----------
        coordinates : Tensor, shape [B, N, 3]
            Input coordinates
        atom_types : Tensor, shape [B, N], optional
            Atom type indices (required for graph flow)
        adj_list : Tensor, shape [E, 2], optional  
            Edge connectivity (required for graph flow)
        edge_batch_idx : Tensor, shape [E], optional
            Batch indices for edges (auto-generated if None)
        masked_elements : Tensor, shape [B, N], optional
            Padding mask (True for padding atoms)
            
        Returns
        -------
        output_coords : Tensor, shape [B, N, 3]
            Transformed coordinates
        total_log_det : Tensor, shape [B]
            Total log-determinant
        """
        if atom_types is None or adj_list is None:
            raise ValueError(
                "Message-passing flow requires atom_types and adj_list. "
                "Update trainer to pass molecular data from batch."
            )
        
        # Call forward with reverse=True
        return self.forward(
            coordinates=coordinates,
            atom_types=atom_types,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            reverse=True,
        )
    
    def log_likelihood(
        self,
        x_coords: Tensor,  # [B, N, 3] source configuration
        y_coords: Tensor,  # [B, N, 3] target configuration
        reverse: bool = False,  # direction flag
        # Molecular data (required)
        atom_types: Optional[Tensor] = None,
        adj_list: Optional[Tensor] = None,
        edge_batch_idx: Optional[Tensor] = None,
        masked_elements: Optional[Tensor] = None,
        # Legacy interface compatibility
        source_coords: Optional[Tensor] = None,  
        target_coords: Optional[Tensor] = None,
        direction: Optional[str] = None,
        peptide_names: Optional[List[str]] = None,  # For multi-peptide mode
    ) -> Tensor:
        """Compute log-likelihood for bidirectional training.
        
        Parameters
        ----------
        x_coords : Tensor, shape [B, N, 3]
            Source coordinates
        y_coords : Tensor, shape [B, N, 3]  
            Target coordinates
        reverse : bool
            Direction flag
        atom_types, adj_list, etc. : Tensor
            Required molecular data
            
        Returns
        -------
        log_likelihood : Tensor, shape [B]
            Log-likelihood values for each sample
        """
        # Interface conversion
        if source_coords is None:
            source_coords = x_coords
        if target_coords is None:
            target_coords = y_coords
        if direction is None:
            direction = "reverse" if reverse else "forward"
            
        # Validate required molecular data
        if atom_types is None:
            raise ValueError("atom_types is required for message-passing flow. Dataset should provide real molecular data.")
        if adj_list is None:
            raise ValueError("adj_list is required for message-passing flow. Dataset should provide real molecular bond connectivity.")
            
        # Generate edge batch indices if not provided
        if edge_batch_idx is None:
            B = source_coords.shape[0]
            n_edges = adj_list.shape[0]
            edge_batch_idx = torch.repeat_interleave(
                torch.arange(B, device=source_coords.device), 
                n_edges
            )
            
        if direction == "forward":
            # Forward: log p(target | source) = log p_low(f^-1(target)) + log|det J_f^-1|
            reconstructed_coords, log_det = self.forward(
                coordinates=target_coords,  # Apply inverse to target
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                reverse=True,  # Inverse flow (f^-1)
            )
            
            # Evaluate reconstructed coordinates with low temperature base
            log_prob_base = self._log_boltzmann_masked(
                coordinates=reconstructed_coords,
                target_distribution=self.source_target,  # Low temperature
                masked_elements=masked_elements,
                peptide_names=peptide_names,  # For multi-peptide mode
                temperature=self.source_temp,  # Low temperature
            )
            
            log_likelihood = log_prob_base + log_det
            
        elif direction == "reverse":
            # Reverse: log p(source | target) = log p_high(f(source)) + log|det J_f|
            reconstructed_coords, log_det = self.forward(
                coordinates=source_coords,  # Apply forward to source
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                reverse=False,  # Forward flow (f)
            )
            
            # Evaluate reconstructed coordinates with high temperature base
            log_prob_base = self._log_boltzmann_masked(
                coordinates=reconstructed_coords,
                target_distribution=self.target_target,  # High temperature
                masked_elements=masked_elements,
                peptide_names=peptide_names,  # For multi-peptide mode
                temperature=self.target_temp,  # High temperature
            )
            
            log_likelihood = log_prob_base + log_det
            
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
        target_distribution,  # AldpBoltzmann instance
        masked_elements: Optional[Tensor] = None,  # [B, N]
        peptide_names: Optional[List[str]] = None,  # For multi-peptide mode
        temperature: Optional[float] = None,  # Temperature for creating peptide-specific targets
    ) -> Tensor:
        """Compute masked Boltzmann log-probability."""
        B = coordinates.shape[0]
        
        # Handle multi-peptide batches by creating peptide-specific targets
        # FIXED: Check for peptide_names first, regardless of masked_elements
        if peptide_names is not None and temperature is not None:
            from ..targets import build_target
            
            log_probs = []
            for i in range(B):
                peptide_name = peptide_names[i]
                
                # Extract valid coordinates for this sample
                if masked_elements is not None:
                    valid_mask = ~masked_elements[i]  # [N]
                    valid_coords = coordinates[i][valid_mask].view(-1)  # [3*n_valid]
                else:
                    valid_coords = coordinates[i].view(-1)  # All coordinates are valid
                
                # Create peptide-specific target
                if peptide_name.upper() == "AX":
                    sample_target = build_target(
                        "aldp", 
                        temperature=temperature, 
                        device="cpu"
                    )
                else:
                    pdb_path = f"datasets/pt_dipeptides/{peptide_name}/ref.pdb"
                    sample_target = build_target(
                        "dipeptide",
                        temperature=temperature,
                        device="cpu",
                        pdb_path=pdb_path,
                        env="implicit"
                    )
                
                # Evaluate target on valid coordinates
                log_prob = sample_target.log_prob(valid_coords.unsqueeze(0)).squeeze(0)
                log_probs.append(log_prob)
                
            return torch.stack(log_probs)
        
        # Original single-peptide logic (only used when peptide_names is None)
        # Flatten coordinates while masking padding
        if masked_elements is not None:
            coords_masked = coordinates * (~masked_elements).unsqueeze(-1)
            coords_flat = coords_masked.view(B, -1)  # [B, 3N]
        else:
            coords_flat = coordinates.view(B, -1)
            
        # Boltzmann log-probability
        log_probs = []
        for i in range(B):
            if masked_elements is not None:
                # Extract only non-masked coordinates
                valid_mask = ~masked_elements[i]  # [N]
                valid_coords = coordinates[i][valid_mask].view(-1)  # [3*n_valid]
                log_prob = target_distribution.log_prob(valid_coords.unsqueeze(0)).squeeze(0)
            else:
                log_prob = target_distribution.log_prob(coords_flat[i].unsqueeze(0)).squeeze(0)
            log_probs.append(log_prob)
            
        return torch.stack(log_probs)
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"PTSwapGraphFlow(layers={self.num_layers}, "
            f"message_passing, "
            f"T_source={self.source_temp:.3f}, "
            f"T_target={self.target_temp:.3f})"
        ) 