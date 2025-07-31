"""Random Fourier Features position encoder for 3D coordinates.

Based on Timewarp's RFFPositionEncoder implementation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from dataclasses import dataclass

__all__ = ["RFFPositionEncoder", "RFFPositionEncoderConfig"]


@dataclass
class RFFPositionEncoderConfig:
    """Configuration for Random Fourier Features position encoder."""
    encoding_dim: int = 64
    scale_mean: float = 1.0
    scale_stddev: float = 1.0


class RFFPositionEncoder(nn.Module):
    """Random Fourier Features position encoder for 3D coordinates.
    
    Encodes 3D positions using random Fourier features to help the transformer
    understand spatial relationships without explicit distance calculations.
    
    Parameters
    ----------
    input_dim : int
        Input coordinate dimension (typically 3 for xyz)
    encoding_dim : int 
        Output encoding dimension
    scale_mean : float
        Mean of the random frequency scale distribution
    scale_stddev : float
        Standard deviation of the random frequency scale distribution
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        encoding_dim: int = 64,
        scale_mean: float = 1.0,
        scale_stddev: float = 1.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Random frequencies - fixed during training
        # Shape: [encoding_dim//2, input_dim]
        self.register_buffer(
            "frequencies",
            torch.randn(encoding_dim // 2, input_dim) * scale_stddev + scale_mean
        )
        
        # Ensure even encoding dimension for sin/cos pairs
        assert encoding_dim % 2 == 0, f"encoding_dim must be even, got {encoding_dim}"
    
    def forward(self, coords: Tensor) -> Tensor:
        """Encode 3D coordinates using random Fourier features.
        
        Parameters
        ----------
        coords : Tensor, shape [B, N, 3] or [B, N, input_dim]
            Input coordinates
            
        Returns
        -------
        Tensor, shape [B, N, encoding_dim]
            Random Fourier feature encoding
        """
        # coords: [B, N, 3]
        # frequencies: [encoding_dim//2, 3] 
        
        # Ensure frequencies are on the same device as input coordinates
        frequencies = self.frequencies.to(coords.device)
        
        # Compute dot products: [B, N, encoding_dim//2]
        projections = torch.matmul(coords, frequencies.T)
        
        # Apply sin/cos to get [B, N, encoding_dim]
        sin_features = torch.sin(projections)
        cos_features = torch.cos(projections)
        
        # Concatenate sin and cos features
        return torch.cat([sin_features, cos_features], dim=-1) 