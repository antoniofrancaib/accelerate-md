"""Transformer-based coupling layer for normalizing flows.

Based on Timewarp's TransformerCouplingLayer implementation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Literal, Tuple, Optional

from .transformer_block import TransformerBlock, TransformerConfig
from .rff_position_encoder import RFFPositionEncoder, RFFPositionEncoderConfig

__all__ = ["TransformerCouplingLayer"]


class TransformerCouplingLayer(nn.Module):
    """Transformer-based coupling layer for molecular normalizing flows.
    
    This layer transforms either positions or velocities using transformer
    attention to capture molecular interactions. Based on Timewarp's design.
    
    The input features are:
    - Atom embeddings: [B, N, atom_embed_dim]  
    - Coordinates: [B, N, 3]
    - Velocities: [B, N, 3] 
    - Half of z (untransformed variables): [B, N, 3]
    - RFF position encoding: [B, N, rff_dim]
    
    Parameters
    ----------
    atom_embedding_dim : int
        Dimension of atom type embeddings
    transformer_hidden_dim : int
        Hidden dimension for transformer layers
    mlp_hidden_layer_dims : List[int]
        Hidden layer dimensions for MLPs
    num_transformer_layers : int
        Number of transformer encoder layers
    transformed_vars : Literal["positions", "velocities"]
        Which variables this layer transforms
    transformer_config : TransformerConfig
        Transformer configuration
    rff_position_encoder_config : RFFPositionEncoderConfig
        RFF position encoder configuration
    """
    
    def __init__(
        self,
        atom_embedding_dim: int,
        transformer_hidden_dim: int,
        mlp_hidden_layer_dims: List[int],
        num_transformer_layers: int,
        transformed_vars: Literal["positions", "velocities"],
        transformer_config: TransformerConfig,
        rff_position_encoder_config: RFFPositionEncoderConfig,
    ):
        super().__init__()
        
        self.transformed_vars = transformed_vars
        
        # RFF position encoder for coordinates
        self.position_encoder = RFFPositionEncoder(
            input_dim=3,
            encoding_dim=rff_position_encoder_config.encoding_dim,
            scale_mean=rff_position_encoder_config.scale_mean,
            scale_stddev=rff_position_encoder_config.scale_stddev,
        )
        
        # Calculate input dimension for transformer
        # Features: atom_embeddings + x_coords + x_velocs + z_half + rff_encoding
        input_dim = (
            atom_embedding_dim +  # atom embeddings
            3 +  # x coordinates  
            3 +  # x velocities
            3 +  # half of z (either z_coords or z_velocs)
            rff_position_encoder_config.encoding_dim  # RFF encoding
        )
        
        # Scale transformer (predicts log-scale, will be exponentiated)
        self.scale_transformer = TransformerBlock(
            input_dim=input_dim,
            output_dim=3,  # Scale for x, y, z
            latent_dim=transformer_hidden_dim,
            mlp_hidden_layer_dims=mlp_hidden_layer_dims,
            num_transformer_layers=num_transformer_layers,
            transformer_config=transformer_config,
        )
        
        # Shift transformer  
        self.shift_transformer = TransformerBlock(
            input_dim=input_dim,
            output_dim=3,  # Shift for x, y, z
            latent_dim=transformer_hidden_dim,
            mlp_hidden_layer_dims=mlp_hidden_layer_dims,
            num_transformer_layers=num_transformer_layers,
            transformer_config=transformer_config,
        )
    
    def forward(
        self,
        z: Tensor,  # [B, N, 3] - variables being transformed
        atom_embeddings: Tensor,  # [B, N, atom_embed_dim]
        x_coords: Tensor,  # [B, N, 3] - conditioning coordinates
        x_velocs: Tensor,  # [B, N, 3] - conditioning velocities  
        z_other_half: Tensor,  # [B, N, 3] - other half of z (coords or velocs)
        masked_elements: torch.BoolTensor,  # [B, N] - padding mask
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass computing scale and shift transformations.
        
        Parameters
        ----------
        z : Tensor, shape [B, N, 3]
            Variables being transformed (positions or velocities)
        atom_embeddings : Tensor, shape [B, N, atom_embed_dim]
            Atom type embeddings
        x_coords : Tensor, shape [B, N, 3]
            Conditioning coordinates
        x_velocs : Tensor, shape [B, N, 3]
            Conditioning velocities
        z_other_half : Tensor, shape [B, N, 3]
            Other half of z variables (not being transformed this layer)
        masked_elements : torch.BoolTensor, shape [B, N]
            Padding mask (True for padded positions)
            
        Returns
        -------
        scale : Tensor, shape [B, N, 3]
            Scale parameters (positive)
        shift : Tensor, shape [B, N, 3]  
            Shift parameters
        """
        # Encode coordinates with RFF
        x_coords_enc = self.position_encoder(x_coords)  # [B, N, rff_dim]
        
        # Concatenate all conditioning features
        # Features: [atom_embeddings, x_coords, x_velocs, z_other_half, rff_encoding]
        untransformed_vars = torch.cat([
            atom_embeddings,  # [B, N, atom_embed_dim]
            x_coords,         # [B, N, 3]
            x_velocs,         # [B, N, 3]
            z_other_half,     # [B, N, 3]
            x_coords_enc,     # [B, N, rff_dim]
        ], dim=-1)  # [B, N, total_dim]
        
        # Compute scale and shift using transformers
        raw_log_scale = self.scale_transformer(
            untransformed_vars, 
            masked_elements=masked_elements
        )  # [B, N, 3]
        
        shift = self.shift_transformer(
            untransformed_vars,
            masked_elements=masked_elements  
        )  # [B, N, 3]
        
        # Regularize log-scale to prevent numerical instability (matching other coupling layers)
        log_scale = torch.tanh(raw_log_scale) * 0.05  # Limit to ±0.05
        
        # Exponentiate to get positive scale
        scale = torch.exp(log_scale)  # [B, N, 3]
        
        return scale, shift
    
    def _get_scale_and_shift(
        self,
        atom_types: Tensor,  # [B, N] - for compatibility
        z_coords: Tensor,  # [B, N, 3]
        z_velocs: Tensor,  # [B, N, 3] 
        x_features: Tensor,  # [B, N, D] - atom embeddings
        x_coords: Tensor,  # [B, N, 3]
        x_velocs: Tensor,  # [B, N, 3]
        adj_list: Optional[Tensor],  # Not used in transformer approach
        edge_batch_idx: Optional[Tensor],  # Not used
        masked_elements: torch.BoolTensor,  # [B, N]
        cache=None,  # Not used
        logger=None,  # Not used
    ) -> Tuple[Tensor, Tensor]:
        """Compatibility method matching Timewarp's interface.
        
        This method provides the same interface as Timewarp's coupling layers
        for easier integration with existing flow architectures.
        """
        if self.transformed_vars == "positions":
            # Transform positions using velocities as conditioning
            return self.forward(
                z=z_coords,
                atom_embeddings=x_features,
                x_coords=x_coords,
                x_velocs=x_velocs,
                z_other_half=z_velocs,  # Use velocities as other half
                masked_elements=masked_elements,
            )
        elif self.transformed_vars == "velocities":
            # Transform velocities using positions as conditioning  
            return self.forward(
                z=z_velocs,
                atom_embeddings=x_features,
                x_coords=x_coords,
                x_velocs=x_velocs,
                z_other_half=z_coords,  # Use coordinates as other half
                masked_elements=masked_elements,
            )
        else:
            raise ValueError(f"Unknown transformed_vars: {self.transformed_vars}") 