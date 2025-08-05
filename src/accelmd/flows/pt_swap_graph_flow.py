"""PT swap flow using Timewarp-style E(3)-equivariant graph neural networks.

This module implements the main PT swap flow model using Timewarp's architecture:
- E(3)-equivariant message passing with dense coupling layers
- Position-only transformations (no velocities)
- Physics-informed base distributions (Boltzmann at source temperature)
- Conservative scaling for multi-peptide training stability

Key improvements over previous graph implementation:
- Uses proven Timewarp equivariant coupling layers
- Better numerical stability with conservative scale ranges
- Proper masking for variable-size molecules
- Temperature conditioning throughout
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict, Any

from .equivariant_flow_base import PTSequentialFlow
from .equivariant_coupling import EquivariantCouplingLayer
from ..targets import build_target

__all__ = ["PTSwapGraphFlow"]


class PTSwapGraphFlow(nn.Module):
    """PT swap flow using Timewarp-style E(3)-equivariant architecture.
    
    This implementation adapts Timewarp's successful equivariant flow design for
    PT swap proposals, with key modifications for our use case:
    - Input is molecular coordinates (not latent noise)
    - Physics-informed base distribution (Boltzmann)
    - Position-only transformations
    - Conservative scaling for stability
    
    Parameters
    ----------
    num_layers : int
        Number of coupling layers
    atom_vocab_size : int
        Number of unique atom types
    atom_embed_dim : int
        Dimension of atom embeddings
    hidden_dim : int
        Hidden dimension for MLPs
    scale_range : float
        Maximum scaling factor (conservative for stability)
    source_temperature : float
        Source temperature for PT
    target_temperature : float
        Target temperature for PT
    target_name : str
        Target distribution name
    target_kwargs : dict, optional
        Additional target parameters
    device : str
        Computation device
    """
    
    def __init__(
        self,
        num_layers: int = 8,
        atom_vocab_size: int = 4,  # H, C, N, O
        atom_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_mlp_layers: int = 2,  # Number of MLP layers
        scale_range: float = 0.05,  # Initial conservative scaling
        scale_range_end: float = 0.15,  # Final scale range after scheduling
        scale_range_schedule_epochs: int = 20,  # Epochs to reach final scale
        shift_range: float = 1.0,  # Maximum shift magnitude in Angstroms
        max_neighbors: int = 20,
        distance_cutoff: float = 8.0,
        source_temperature: float = 1.0,
        target_temperature: float = 1.5,
        target_name: str = "aldp",
        target_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.atom_vocab_size = atom_vocab_size
        self.atom_embed_dim = atom_embed_dim
        self.hidden_dim = hidden_dim
        self.num_mlp_layers = num_mlp_layers
        self.scale_range = scale_range
        self.scale_range_end = scale_range_end
        self.scale_range_schedule_epochs = scale_range_schedule_epochs
        self.shift_range = shift_range
        self.max_neighbors = max_neighbors
        self.distance_cutoff = distance_cutoff
        self.source_temperature = source_temperature
        self.target_temperature = target_temperature
        self.device = device
        
        # Build coupling layers with alternating phases
        layers = []
        for i in range(num_layers):
            phase = i % 2
            layer = EquivariantCouplingLayer(
                phase=phase,
                atom_vocab_size=atom_vocab_size,
                atom_embed_dim=atom_embed_dim,
                hidden_dim=hidden_dim,
                num_mlp_layers=num_mlp_layers,
                scale_range=scale_range,  # Will be updated during training
                shift_range=shift_range,
                max_neighbors=max_neighbors,
                distance_cutoff=distance_cutoff,
                temperature_conditioning=True,
            )
            layers.append(layer)
        
        self.flow = PTSequentialFlow(layers)
        
        # Initialize target distributions
        self.source_target = build_target(target_name, temperature=source_temperature, device=device, **(target_kwargs or {}))
        self.target_target = build_target(target_name, temperature=target_temperature, device=device, **(target_kwargs or {}))
        
        # Aliases for compatibility with trainer loss calculation
        self.base_low = self.source_target
        self.base_high = self.target_target
        
        # Move to device
        self.to(device)
    
    def update_scale_range(self, current_epoch: int):
        """Update scale range based on training progress.
        
        Gradually increases scale range from initial value to final value over
        the specified number of epochs for more aggressive transformations as
        training progresses.
        
        Parameters
        ----------
        current_epoch : int
            Current training epoch (0-indexed)
        """
        if current_epoch >= self.scale_range_schedule_epochs:
            # Use final scale range
            target_scale_range = self.scale_range_end
        else:
            # Linear interpolation between initial and final scale range
            progress = current_epoch / self.scale_range_schedule_epochs
            target_scale_range = self.scale_range + progress * (self.scale_range_end - self.scale_range)
        
        # Update all coupling layers
        for layer in self.flow.layers:
            if hasattr(layer, 'scale_range'):
                layer.scale_range = target_scale_range
    
    def forward(
        self,
        coordinates: Tensor,  # [B, N, 3]
        atom_types: Optional[Tensor] = None,  # [B, N]
        adj_list: Optional[Tensor] = None,  # [E, 2] (not used but kept for interface)
        edge_batch_idx: Optional[Tensor] = None,  # [E] (not used but kept for interface)
        masked_elements: Optional[Tensor] = None,  # [B, N]
        reverse: bool = False,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Apply flow transformation to molecular coordinates.
        
        Parameters
        ----------
        coordinates : Tensor, shape [B, N, 3]
            Input molecular coordinates
        atom_types : Tensor, shape [B, N], optional
            Atom type indices (required for graph flow)
        adj_list : Tensor, optional
            Not used in Timewarp architecture (kept for interface compatibility)
        edge_batch_idx : Tensor, optional
            Not used in Timewarp architecture (kept for interface compatibility)
        masked_elements : Tensor, shape [B, N], optional
            Padding mask (True for padding atoms)
        reverse : bool
            Direction of transformation
            
        Returns
        -------
        output_coords : Tensor, shape [B, N, 3]
            Transformed coordinates
        log_det : Tensor, shape [B]
            Log-determinant of transformation
        """
        if atom_types is None:
            raise ValueError("atom_types is required for graph flow")
        
        B, N, _ = coordinates.shape
        device = coordinates.device
        
        # Create dummy adjacency list if not provided (Timewarp uses dense interactions)
        if adj_list is None:
            # Create empty adjacency list - Timewarp computes interactions from distances
            adj_list = torch.empty(0, 2, dtype=torch.long, device=device)
        
        if edge_batch_idx is None:
            edge_batch_idx = torch.empty(0, dtype=torch.long, device=device)
        
        # Apply flow transformation with real molecular connectivity
        output_coords, log_det = self.flow(
            coordinates=coordinates,
            atom_types=atom_types,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            reverse=reverse,
            source_temp=self.source_temperature,
            target_temp=self.target_temperature,
        )
        
        return output_coords, log_det
    
    def log_likelihood(
        self,
        x_coords: Tensor,  # [B, N, 3] source configuration
        y_coords: Tensor,  # [B, N, 3] target configuration
        atom_types: Tensor,  # [B, N]
        adj_list: Optional[Tensor] = None,  # Not used
        edge_batch_idx: Optional[Tensor] = None,  # Not used
        masked_elements: Optional[Tensor] = None,  # [B, N]
        reverse: bool = False,
        **kwargs
    ) -> Tensor:
        """Compute log-likelihood of transformation from x_coords to y_coords.
        
        Following Timewarp's approach but with physics-informed base distribution:
        log p(y|x) = log p_Boltz(x) + log|det J|
        
        where p_Boltz is the Boltzmann distribution at the source temperature.
        """
        if reverse:
            # Reverse direction: target → source
            # Transform target coords through flow to get "predicted" source coords
            transformed_coords, log_det = self.forward(
                coordinates=y_coords,
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                reverse=True,  # Apply inverse transformation
            )
            
            # Compute Boltzmann log-probability at target temperature
            log_boltz = self._log_boltzmann_masked(
                y_coords,
                self.target_target,
                masked_elements,
                peptide_names=kwargs.get("peptide_names"),
                temperature=self.target_temperature,
            )
        else:
            # Forward direction: source → target
            # Transform source coords through flow to get "predicted" target coords
            transformed_coords, log_det = self.forward(
                coordinates=x_coords,
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                reverse=False,  # Apply forward transformation
            )
            
            # Compute Boltzmann log-probability at source temperature
            log_boltz = self._log_boltzmann_masked(
                x_coords,
                self.source_target,
                masked_elements,
                peptide_names=kwargs.get("peptide_names"),
                temperature=self.source_temperature,
            )
        
        # Total log-likelihood: base distribution + change of variables
        log_likelihood = log_boltz + log_det
        
        return log_likelihood
    
    def _log_boltzmann_masked(
        self,
        coordinates: Tensor,  # [B, N, 3]
        target_distribution,  # Boltzmann target instance (single-peptide mode)
        masked_elements: Optional[Tensor] = None,  # [B, N]
        peptide_names: Optional[list[str]] = None,  # For multi-peptide batches
        temperature: Optional[float] = None,
    ) -> Tensor:
        """Compute Boltzmann log-probability, handling padding **and** mixed-peptide batches.

        If *peptide_names* is supplied, we build a peptide-specific Boltzmann target for
        every sample so that dimension mismatches disappear.
        """

        B, N, _ = coordinates.shape

        # ----------------------------------------------------
        # 1. Multi-peptide path – build peptide-specific targets
        # ----------------------------------------------------
        if peptide_names is not None and temperature is not None:
            from ..targets import build_target  # Local import to avoid heavy deps at module import time

            log_probs: list[Tensor] = []
            for i in range(B):
                pep = peptide_names[i]

                # --- extract valid coordinates for this sample
                if masked_elements is not None:
                    valid_mask = ~masked_elements[i]
                    coords_flat = coordinates[i][valid_mask].view(-1)  # [3*n_valid]
                else:
                    coords_flat = coordinates[i].view(-1)

                # --- build peptide-specific Boltzmann target
                if pep.upper() == "AX":  # legacy alias for alanine dipeptide
                    sample_target = build_target(
                        "aldp",
                        temperature=temperature,
                        device="cpu",
                    )
                else:
                    pdb_path = f"datasets/pt_dipeptides/{pep}/ref.pdb"
                    sample_target = build_target(
                        "dipeptide",
                        temperature=temperature,
                        device="cpu",
                        pdb_path=pdb_path,
                        env="implicit",
                    )

                log_prob = sample_target.log_prob(coords_flat.unsqueeze(0)).squeeze(0)
                log_probs.append(log_prob)

            return torch.stack(log_probs)

        # ----------------------------------------------------
        # 2. Single-peptide fast path (previous behaviour)
        # ----------------------------------------------------
        if masked_elements is not None:
            log_probs = []
            for i in range(B):
                valid_mask = ~masked_elements[i]
                coords_flat = coordinates[i][valid_mask].view(-1)
                log_prob = target_distribution.log_prob(coords_flat.unsqueeze(0)).squeeze(0)
                log_probs.append(log_prob)
            return torch.stack(log_probs)

        # No masking, single peptide
        coords_flat = coordinates.view(B, -1)
        return target_distribution.log_prob(coords_flat)
    
    def sample_proposal(
        self,
        source_coords: Tensor,  # [B, N, 3]
        atom_types: Tensor,     # [B, N]
        adj_list: Optional[Tensor] = None,
        edge_batch_idx: Optional[Tensor] = None,
        masked_elements: Optional[Tensor] = None,
        direction: str = "forward"
    ) -> Tensor:
        """Generate swap proposal using the trained flow.
        
        Parameters
        ----------
        source_coords : Tensor, shape [B, N, 3]
            Source molecular coordinates
        atom_types : Tensor, shape [B, N]
            Atom type indices
        direction : str
            "forward" for low→high temp, "reverse" for high→low temp
            
        Returns
        -------
        proposal_coords : Tensor, shape [B, N, 3]
            Proposed coordinates at target temperature
        """
        with torch.no_grad():
            if direction == "forward":
                # Low → High temperature: use forward flow
                proposal_coords, _ = self.forward(
                    coordinates=source_coords,
                    atom_types=atom_types,
                    adj_list=adj_list,
                    edge_batch_idx=edge_batch_idx,
                    masked_elements=masked_elements,
                    reverse=False,
                )
            else:
                # High → Low temperature: use reverse flow
                proposal_coords, _ = self.forward(
                    coordinates=source_coords,
                    atom_types=atom_types,
                    adj_list=adj_list,
                    edge_batch_idx=edge_batch_idx,
                    masked_elements=masked_elements,
                    reverse=True,
                )
        
        return proposal_coords
    
    def inverse(
        self,
        coordinates: Tensor,  # [B, N, 3]
        atom_types: Optional[Tensor] = None,  # [B, N]
        adj_list: Optional[Tensor] = None,  # Not used
        edge_batch_idx: Optional[Tensor] = None,  # Not used
        masked_elements: Optional[Tensor] = None,  # [B, N]
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Inverse transformation for training code compatibility.
        
        This is just a wrapper around forward(reverse=True).
        """
        return self.forward(
            coordinates=coordinates,
            atom_types=atom_types,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            reverse=True,
            **kwargs
        )
    
    def to(self, device):
        """Move model to device and update targets."""
        super().to(device)
        if hasattr(self, 'source_target') and hasattr(self.source_target, 'to'):
            self.source_target = self.source_target.to(device)
        if hasattr(self, 'target_target') and hasattr(self.target_target, 'to'):
            self.target_target = self.target_target.to(device)
        self.device = str(device)
        return self 