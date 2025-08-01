"""Dimension-agnostic transformer flow for PT swap proposals.

Based on Timewarp's transformer architecture with atom embeddings and
proper masking for variable molecule sizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple, Dict, Any

from ..targets import build_target
from .transformer_coupling_layer import TransformerCouplingLayer
from .transformer_block import TransformerConfig
from .rff_position_encoder import RFFPositionEncoderConfig

__all__ = ["PTSwapTransformerFlow"]


class PTSwapTransformerFlow(nn.Module):
    """Dimension-agnostic transformer flow for PT swap proposals.
    
    Implements Timewarp's transformer architecture:
    - Simple atom type embeddings (no complex graph neural networks)
    - Random Fourier Features for position encoding
    - Transformer attention with proper masking for variable lengths
    - Alternating position/velocity coupling layers
    
    This architecture is fully dimension-agnostic and can handle molecules
    of any size without retraining.
    
    Parameters
    ----------
    num_layers : int
        Number of coupling layers
    atom_vocab_size : int
        Number of unique atom types (e.g., 4 for H,C,N,O)
    atom_embed_dim : int
        Dimension of atom type embeddings
    transformer_hidden_dim : int
        Hidden dimension for transformer layers
    mlp_hidden_layer_dims : List[int]
        Hidden layer dimensions for MLPs
    num_transformer_layers : int
        Number of transformer encoder layers per coupling layer
    source_temperature : float
        Source temperature for physics base distribution
    target_temperature : float
        Target temperature for physics base distribution
    target_name : str
        Name of the Boltzmann target distribution
    target_kwargs : Dict[str, Any]
        Additional arguments for target distribution
    transformer_config : TransformerConfig
        Transformer configuration
    rff_position_encoder_config : RFFPositionEncoderConfig
        RFF position encoder configuration
    device : str
        Compute device
    """
    
    def __init__(
        self,
        num_layers: int = 8,
        atom_vocab_size: int = 4,  # H, C, N, O
        atom_embed_dim: int = 32,
        transformer_hidden_dim: int = 128,
        mlp_hidden_layer_dims: List[int] = [128, 128],
        num_transformer_layers: int = 2,
        source_temperature: float = 1.0,
        target_temperature: float = 1.5,
        target_name: str = "aldp",
        target_kwargs: Optional[Dict[str, Any]] = None,
        transformer_config: Optional[TransformerConfig] = None,
        rff_position_encoder_config: Optional[RFFPositionEncoderConfig] = None,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.atom_vocab_size = atom_vocab_size
        self.atom_embed_dim = atom_embed_dim
        self.source_temp = source_temperature
        self.target_temp = target_temperature
        
        # Default configurations
        if transformer_config is None:
            transformer_config = TransformerConfig(
                n_head=8,
                dim_feedforward=2048,
                dropout=0.0,  # No dropout for deterministic likelihood
            )
        
        if rff_position_encoder_config is None:
            rff_position_encoder_config = RFFPositionEncoderConfig(
                encoding_dim=64,
                scale_mean=1.0,
                scale_stddev=1.0,
            )
        
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
        
        # Atom type embeddings - this is the key to dimension agnosticism
        # Simple embedding layer for transformer architecture (decoupled from graph approach)
        self.atom_embedder = nn.Embedding(
            num_embeddings=atom_vocab_size,
            embedding_dim=atom_embed_dim,
        )
        
        # Create coupling layers alternating between positions and velocities
        self.layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            # Alternate between transforming positions and velocities
            transformed_vars = "positions" if layer_idx % 2 == 0 else "velocities"
            
            layer = TransformerCouplingLayer(
                atom_embedding_dim=atom_embed_dim,
                transformer_hidden_dim=transformer_hidden_dim,
                mlp_hidden_layer_dims=mlp_hidden_layer_dims,
                num_transformer_layers=num_transformer_layers,
                transformed_vars=transformed_vars,
                transformer_config=transformer_config,
                rff_position_encoder_config=rff_position_encoder_config,
            )
            self.layers.append(layer)
    
    def _embed_atoms(self, atom_types: Tensor) -> Tensor:
        """Embed atom types into continuous representations.
        
        Parameters
        ----------
        atom_types : Tensor, shape [B, N]
            Atom type indices (should be on the same device as the model)
            
        Returns
        -------
        Tensor, shape [B, N, atom_embed_dim]
            Atom embeddings
        """
        # Ensure embedding layer and input are on the same device
        device = atom_types.device
        self.atom_embedder = self.atom_embedder.to(device)
        return self.atom_embedder(atom_types)
    
    def _create_padding_mask(self, atom_types: Tensor) -> torch.BoolTensor:
        """Create padding mask from atom types.
        
        Assumes atom type 0 is padding (or we can detect padding from zeros).
        
        Parameters
        ----------
        atom_types : Tensor, shape [B, N]
            Atom type indices (0 for padding)
            
        Returns
        -------
        torch.BoolTensor, shape [B, N]
            Padding mask (True for padded positions)
        """
        # Keep atom_types on its original device
        # Simple approach: assume atom_type == 0 means padding
        return atom_types == 0
    
    def forward(
        self,
        coordinates: Tensor,  # [B, N, 3] - to match PTSwapGraphFlow interface  
        atom_types: Optional[Tensor] = None,  # [B, N]
        adj_list: Optional[Tensor] = None,  # For interface compatibility (not used)
        edge_batch_idx: Optional[Tensor] = None,  # For interface compatibility (not used)
        masked_elements: Optional[Tensor] = None,  # [B, N] padding mask
        reverse: bool = False,  # For interface compatibility
        velocities: Optional[Tensor] = None,  # [B, N, 3]
        return_log_det: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass through the transformer flow.
        
        Parameters
        ----------
        coordinates : Tensor, shape [B, N, 3]
            Input coordinates (padded to max length in batch)
        atom_types : Tensor, shape [B, N], optional
            Atom type indices (required for transformer flow)
        adj_list : Tensor, optional
            Not used by transformer flow (for interface compatibility)
        edge_batch_idx : Tensor, optional
            Not used by transformer flow (for interface compatibility)  
        masked_elements : Tensor, shape [B, N], optional
            Padding mask (True for padded positions)
        reverse : bool
            Whether to apply reverse transformation
        velocities : Tensor, shape [B, N, 3], optional
            Input velocities (if None, will be sampled from base distribution)
        return_log_det : bool
            Whether to return log determinant
            
        Returns
        -------
        transformed_coords : Tensor, shape [B, N, 3]
            Transformed coordinates
        log_det : Tensor, shape [B], optional
            Log determinant of the transformation
        """
        # Validate required inputs
        if atom_types is None:
            raise ValueError("atom_types is required for transformer flow")
        
        batch_size, max_atoms, _ = coordinates.shape
        device = coordinates.device

        # Ensure the entire model (including nested layers) resides on the same
        # device as the input coordinates. This guards against cases where the
        # model was instantiated on CPU but the data is on GPU (common in
        # distributed / SLURM jobs). Moving once per forward is inexpensive
        # compared to training cost and prevents device mismatch errors.
        if next(self.parameters()).device != device:
            self.to(device)
        
        # Handle velocities
        if velocities is None:
            # Sample from base distribution (Gaussian)
            velocities = torch.randn_like(coordinates)
        
        # Move atom_types to the same device as coordinates
        atom_types = atom_types.to(device)
        
        # Create atom embeddings
        atom_embeddings = self._embed_atoms(atom_types)  # [B, N, atom_embed_dim]
        
        # Create padding mask if not provided
        if masked_elements is None:
            masked_elements = self._create_padding_mask(atom_types)  # [B, N]
        
        # Initialize log determinant
        log_det = torch.zeros(batch_size, device=device) if return_log_det else None
        
        # Current state
        z_coords = coordinates.clone()
        z_velocs = velocities.clone()
        
        # Apply coupling layers in forward or reverse order
        layers = self.layers
        if reverse:
            layers = reversed(layers)
            
        for layer in layers:
            if layer.transformed_vars == "positions":
                # Transform coordinates - use CURRENT state for conditioning (before transformation)
                scale, shift = layer._get_scale_and_shift(
                    atom_types=atom_types,
                    z_coords=z_coords,  # Current state before transformation
                    z_velocs=z_velocs,  # Current state before transformation  
                    x_features=atom_embeddings,
                    x_coords=coordinates,  # Original coords for conditioning
                    x_velocs=velocities,  # Original velocities for conditioning
                    adj_list=None,
                    edge_batch_idx=None,
                    masked_elements=masked_elements,
                )
                
                # Apply proper forward/inverse affine transformation
                if reverse:
                    # Inverse transformation: z = (z - shift) / scale
                    # Add small epsilon to prevent division by zero
                    z_coords = (z_coords - shift) / (scale + 1e-8)
                else:
                    # Forward transformation: z = z * scale + shift
                    z_coords = z_coords * scale + shift
                
                # Update log determinant with correct signs
                if return_log_det:
                    # Only count non-masked elements
                    scale_log = torch.log(scale + 1e-8)  # Small epsilon for stability
                    scale_log = scale_log * (~masked_elements).unsqueeze(-1).float()
                    if reverse:
                        log_det -= scale_log.sum(dim=[1, 2])  # Subtract for inverse
                    else:
                        log_det += scale_log.sum(dim=[1, 2])  # Add for forward
                    
            else:  # layer.transformed_vars == "velocities"
                # Transform velocities - use CURRENT state for conditioning (before transformation)
                scale, shift = layer._get_scale_and_shift(
                    atom_types=atom_types,
                    z_coords=z_coords,  # Current state before transformation
                    z_velocs=z_velocs,  # Current state before transformation
                    x_features=atom_embeddings,
                    x_coords=coordinates,  # Original coords for conditioning
                    x_velocs=velocities,  # Original velocities for conditioning
                    adj_list=None,
                    edge_batch_idx=None,
                    masked_elements=masked_elements,
                )
                
                # Apply proper forward/inverse affine transformation
                if reverse:
                    # Inverse transformation: z = (z - shift) / scale
                    # Add small epsilon to prevent division by zero
                    z_velocs = (z_velocs - shift) / (scale + 1e-8)
                else:
                    # Forward transformation: z = z * scale + shift
                    z_velocs = z_velocs * scale + shift
                
                # Update log determinant with correct signs
                if return_log_det:
                    # Only count non-masked elements
                    scale_log = torch.log(scale + 1e-8)  # Small epsilon for stability
                    scale_log = scale_log * (~masked_elements).unsqueeze(-1).float()
                    if reverse:
                        log_det -= scale_log.sum(dim=[1, 2])  # Subtract for inverse
                    else:
                        log_det += scale_log.sum(dim=[1, 2])  # Add for forward
        
        return z_coords, log_det
    
    def log_likelihood(
        self,
        x_coords: Tensor,  # [B, N, 3] source configuration
        y_coords: Tensor,  # [B, N, 3] target configuration  
        reverse: bool = False,  # direction flag
        # Additional arguments for interface compatibility
        source_coords: Optional[Tensor] = None,
        target_coords: Optional[Tensor] = None,
        atom_types: Optional[Tensor] = None,
        adj_list: Optional[Tensor] = None,  # Not used by transformer
        edge_batch_idx: Optional[Tensor] = None,  # Not used by transformer
        masked_elements: Optional[Tensor] = None,
        direction: Optional[str] = None,
        peptide_names: Optional[List[str]] = None,  # For multi-peptide mode
    ) -> Tensor:
        """Compute physics-informed log-likelihood for bidirectional training.
        
        Uses change of variables formula for normalizing flows with Boltzmann base distributions:
        
        Forward direction (reverse=False): 
            log p(y_high | x_low) = log p_Boltz_high(y_high) + log|det J_f(x_low)|
            where f: x_low → y_high
            
        Reverse direction (reverse=True):
            log p(x_low | y_high) = log p_Boltz_low(x_low) + log|det J_f^{-1}(y_high)|  
            where f^{-1}: y_high → x_low
        
        Parameters
        ----------
        x_coords : Tensor, shape [B, N, 3]
            Source coordinates (old interface)
        y_coords : Tensor, shape [B, N, 3]  
            Target coordinates (old interface)
        reverse : bool
            Direction flag (old interface)
        source_coords, target_coords, atom_types, etc. : optional
            New interface arguments
            
        Returns
        -------
        log_likelihood : Tensor, shape [B]
            Physics-informed log-likelihood values for each sample
        """
        # Interface conversion: handle both old and new calling conventions
        if source_coords is None:
            source_coords = x_coords
        if target_coords is None:
            target_coords = y_coords
        if direction is None:
            direction = "reverse" if reverse else "forward"
            
        # Validate required molecular data
        if atom_types is None:
            raise ValueError("atom_types is required for transformer flow. Dataset should provide real molecular data.")
            
        if not reverse:  # Forward direction: T_low → T_high
            # Physics-informed forward: log p(y_high | x_low) = log p_Boltz_high(y_high) + log|det J_f(x_low)|
            # Transform source coordinates to target space
            transformed_coords, log_det = self.forward(
                coordinates=source_coords,  # Transform from source (T_low)
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                reverse=False,  # Forward transformation f(x_low)
                return_log_det=True,
            )
            
            # Evaluate target coordinates with high temperature Boltzmann distribution
            log_prob_base = self._log_boltzmann_masked(
                coordinates=target_coords,  # Evaluate actual target coordinates
                target_distribution=self.target_target,  # High temperature base
                masked_elements=masked_elements,
                peptide_names=peptide_names,  # For multi-peptide mode
                temperature=self.target_temp,  # High temperature
            )
            
            log_likelihood = log_prob_base + log_det
            
        else:  # Reverse direction: T_high → T_low  
            # Physics-informed reverse: log p(x_low | y_high) = log p_Boltz_low(x_low) + log|det J_f^{-1}(y_high)|
            # Transform target coordinates to source space
            transformed_coords, log_det = self.forward(
                coordinates=target_coords,  # Transform from target (T_high)
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                reverse=True,  # Inverse transformation f^{-1}(y_high)
                return_log_det=True,
            )
            
            # Evaluate source coordinates with low temperature Boltzmann distribution
            log_prob_base = self._log_boltzmann_masked(
                coordinates=source_coords,  # Evaluate actual source coordinates
                target_distribution=self.source_target,  # Low temperature base
                masked_elements=masked_elements,
                peptide_names=peptide_names,  # For multi-peptide mode
                temperature=self.source_temp,  # Low temperature
            )
            
            log_likelihood = log_prob_base + log_det
            
        return log_likelihood
    
    def _log_boltzmann_masked(
        self,
        coordinates: Tensor,  # [B, N, 3]
        target_distribution,  # Boltzmann target instance (source_target or target_target)
        masked_elements: Optional[Tensor] = None,  # [B, N]
        peptide_names: Optional[List[str]] = None,  # For multi-peptide mode
        temperature: Optional[float] = None,  # Temperature for creating peptide-specific targets
    ) -> Tensor:
        """Compute masked Boltzmann log-probability using the specified target distribution."""
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
                
                # CRITICAL FIX: Ensure coordinate dimensions match target expectations
                expected_dim = sample_target.dim
                actual_dim = valid_coords.shape[0]
                
                if actual_dim != expected_dim:
                    # This should not happen if data is consistent, but let's handle it gracefully
                    if actual_dim < expected_dim:
                        # Pad with zeros (though this is not ideal)
                        padded_coords = torch.zeros(expected_dim, device=valid_coords.device, dtype=valid_coords.dtype)
                        padded_coords[:actual_dim] = valid_coords
                        valid_coords = padded_coords
                        print(f"Warning: Padded coordinates for {peptide_name} from {actual_dim} to {expected_dim}")
                    else:
                        # Truncate (also not ideal)
                        valid_coords = valid_coords[:expected_dim]
                        print(f"Warning: Truncated coordinates for {peptide_name} from {actual_dim} to {expected_dim}")
                
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
            
        # Boltzmann log-probability using the provided target distribution
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
    
    def inverse(
        self,
        coordinates: Tensor,  # [B, N, 3]
        atom_types: Optional[Tensor] = None,  # [B, N]
        adj_list: Optional[Tensor] = None,  # For interface compatibility (not used)
        edge_batch_idx: Optional[Tensor] = None,  # For interface compatibility (not used)
        masked_elements: Optional[Tensor] = None,  # [B, N] padding mask
    ) -> Tuple[Tensor, Tensor]:
        """Inverse transformation for trainer compatibility.
        
        Parameters
        ----------
        coordinates : Tensor, shape [B, N, 3]
            Input coordinates
        atom_types : Tensor, shape [B, N], optional
            Atom type indices
        adj_list : Tensor, optional
            Not used (for interface compatibility)
        edge_batch_idx : Tensor, optional
            Not used (for interface compatibility)
        masked_elements : Tensor, shape [B, N], optional
            Padding mask (True for padded positions)
            
        Returns
        -------
        output_coords : Tensor, shape [B, N, 3]
            Transformed coordinates
        log_det : Tensor, shape [B]
            Log determinant
        """
        return self.forward(
            coordinates=coordinates,
            atom_types=atom_types,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            reverse=True,
            return_log_det=True,
        )
    
    def log_prob(
        self,
        coords: Tensor,  # [B, N, 3]
        atom_types: Tensor,  # [B, N]
        target_coords: Tensor,  # [B, N, 3]
    ) -> Tensor:
        """Compute log probability of transformation.
        
        Parameters
        ----------
        coords : Tensor, shape [B, N, 3]
            Source coordinates
        atom_types : Tensor, shape [B, N]
            Atom type indices
        target_coords : Tensor, shape [B, N, 3]
            Target coordinates to evaluate probability for
            
        Returns
        -------
        Tensor, shape [B]
            Log probability
        """
        # This would require implementing the inverse transformation
        # For now, placeholder implementation
        raise NotImplementedError("Inverse transformation not implemented yet")
    
    def sample(
        self,
        coords: Tensor,  # [B, N, 3]
        atom_types: Tensor,  # [B, N]
        num_samples: int = 1,
    ) -> Tensor:
        """Sample transformed coordinates.
        
        Parameters
        ----------
        coords : Tensor, shape [B, N, 3]
            Source coordinates
        atom_types : Tensor, shape [B, N]
            Atom type indices
        num_samples : int
            Number of samples to generate
            
        Returns
        -------
        Tensor, shape [num_samples, B, N, 3]
            Sampled coordinates
        """
        samples = []
        for _ in range(num_samples):
            transformed_coords, _ = self.forward(coords, atom_types)
            samples.append(transformed_coords)
        return torch.stack(samples, dim=0) 