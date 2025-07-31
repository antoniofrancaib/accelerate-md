"""Transformer block for processing variable-length molecular sequences.

Based on Timewarp's TransformerBlock implementation with proper masking.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence
from dataclasses import dataclass

from .mlp import MLP

__all__ = ["TransformerBlock", "TransformerConfig"]


@dataclass
class TransformerConfig:
    """Configuration for transformer encoder layers."""
    n_head: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.0  # Note: dropout causes stochasticity in likelihood computation


class TransformerBlock(nn.Module):
    """Transformer block that processes variable-length molecular sequences.
    
    Handles molecules of different sizes using proper attention masking.
    The architecture:
    1. Input MLP: projects input features to latent space
    2. Transformer encoder: applies self-attention with masking  
    3. Output MLP: projects back to desired output dimension
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension 
    output_dim : int
        Output dimension
    latent_dim : int
        Latent dimension for transformer layers
    mlp_hidden_layer_dims : Sequence[int]
        Hidden layer dimensions for input/output MLPs
    num_transformer_layers : int
        Number of transformer encoder layers
    transformer_config : TransformerConfig
        Configuration for transformer layers
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        mlp_hidden_layer_dims: Sequence[int],
        num_transformer_layers: int,
        transformer_config: TransformerConfig,
    ):
        super().__init__()
        
        # Input projection MLP
        self.in_mlp = MLP(
            input_dim=input_dim,
            hidden_layer_dims=mlp_hidden_layer_dims,
            out_dim=latent_dim,
        )
        
        # Transformer encoder stack
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=transformer_config.n_head,
                dim_feedforward=transformer_config.dim_feedforward,
                dropout=transformer_config.dropout,
                activation="relu",
                batch_first=True,  # Input shape: [batch, seq, feature]
            ),
            num_layers=num_transformer_layers,
        )
        
        # Output projection MLP
        self.out_mlp = MLP(
            input_dim=latent_dim,
            hidden_layer_dims=mlp_hidden_layer_dims,
            out_dim=output_dim,
        )
    
    def forward(
        self, 
        input_seq: Tensor,  # [B, N, input_dim]
        masked_elements: torch.BoolTensor,  # [B, N]
    ) -> Tensor:
        """Forward pass with attention masking for variable lengths.
        
        Parameters
        ----------
        input_seq : Tensor, shape [B, N, input_dim]
            Input sequence with concatenated atom embeddings, coordinates, etc.
        masked_elements : torch.BoolTensor, shape [B, N]
            Boolean mask where True indicates padded/masked positions
            
        Returns
        -------
        Tensor, shape [B, N, output_dim]
            Processed sequence output
        """
        # Project to latent space
        feature_seq = self.in_mlp(input_seq)  # [B, N, latent_dim]
        
        # Apply transformer with masking
        # src_key_padding_mask masks padded positions in attention
        out = self.transformer(
            feature_seq, 
            src_key_padding_mask=masked_elements
        )  # [B, N, latent_dim]
        
        # Project to output space
        return self.out_mlp(out)  # [B, N, output_dim] 