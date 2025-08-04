"""Equivariant flow base classes adapted for PT swap flows.

This module provides the core flow abstractions adapted for
parallel tempering swap proposals:
- Position-only transformations (no velocities)
- Coordinates as input (not random noise)
- Graph structure conditioning only
- Physics-informed base distributions

Key adaptations:
- Removed velocity handling throughout
- Removed Gaussian latent assumptions  
- Added temperature conditioning for PT
- Simplified interface for molecular coordinates
"""

from __future__ import annotations

import abc
import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from typing import Sequence, Optional, Tuple, Any

__all__ = ["PTFlow", "PTSequentialFlow"]

# Type for log-determinant, may be None when not computing likelihood
LogDetType = Optional[Tensor]


class PTFlow(nn.Module, abc.ABC):
    """Base class for PT swap flow layers.
    
    Key differences from standard flows:
    - Input is molecular coordinates (not latent noise)
    - Position-only transformations (no velocities)
    - Conditioning on graph structure only (invariant properties)
    - Temperature-aware for PT applications
    """
    
    def __init__(self, atom_embedder: Optional[nn.Module] = None):
        super().__init__()
        self.atom_embedder = atom_embedder
    
    @abc.abstractmethod
    def forward(
        self,
        coordinates: Tensor,  # [B, N, 3] molecular coordinates
        atom_types: Tensor,   # [B, N] atom type indices
        adj_list: Tensor,     # [E, 2] edge connectivity
        edge_batch_idx: Tensor,  # [E] batch indices for edges
        masked_elements: BoolTensor,  # [B, N] padding mask
        log_det: LogDetType = None,  # [B] accumulated log-determinant
        reverse: bool = False,
        source_temp: Optional[float] = None,  # Source temperature
        target_temp: Optional[float] = None,  # Target temperature
        **kwargs
    ) -> Tuple[Tensor, LogDetType]:
        """Apply flow transformation to molecular coordinates.
        
        Parameters
        ----------
        coordinates : Tensor, shape [B, N, 3]
            Input molecular coordinates
        atom_types : Tensor, shape [B, N]
            Atom type indices for molecular structure
        adj_list : Tensor, shape [E, 2]
            Edge connectivity list (molecular bonds)
        edge_batch_idx : Tensor, shape [E]
            Batch indices for edges
        masked_elements : BoolTensor, shape [B, N]
            Padding mask (True for padding atoms)
        log_det : Tensor, optional, shape [B]
            Accumulated log-determinant from previous layers
        reverse : bool
            Direction of transformation
        source_temp : float, optional
            Source temperature for conditioning
        target_temp : float, optional
            Target temperature for conditioning
            
        Returns
        -------
        output_coords : Tensor, shape [B, N, 3]
            Transformed coordinates
        log_det : Tensor, shape [B]
            Updated log-determinant
        """
        pass


class PTSequentialFlow(PTFlow):
    """Sequential container for PT flow layers.
    
    Chains multiple PT flow layers together, handling forward/reverse direction
    and accumulating log-determinants for likelihood computation.
    """
    
    def __init__(
        self, 
        layers: Sequence[PTFlow], 
        atom_embedder: Optional[nn.Module] = None
    ):
        super().__init__(atom_embedder=atom_embedder)
        self.layers = nn.ModuleList(layers)
    
    def forward(
        self,
        coordinates: Tensor,  # [B, N, 3]
        atom_types: Tensor,   # [B, N]
        adj_list: Tensor,     # [E, 2]
        edge_batch_idx: Tensor,  # [E]
        masked_elements: BoolTensor,  # [B, N]
        log_det: LogDetType = None,
        reverse: bool = False,
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tensor, LogDetType]:
        """Apply sequential flow transformations.
        
        In forward mode: applies layers in order 0 → 1 → 2 → ...
        In reverse mode: applies layers in order ... → 2 → 1 → 0
        
        This ensures that reverse(forward(x)) = x for invertibility.
        """
        # Initialize log-determinant if not provided
        if log_det is None:
            batch_size = coordinates.shape[0]
            device = coordinates.device
            log_det = torch.zeros(batch_size, device=device)
        
        # Apply layers in forward or reverse order
        layer_indices = range(len(self.layers))
        if reverse:
            layer_indices = reversed(layer_indices)
        
        current_coords = coordinates
        
        for i in layer_indices:
            current_coords, log_det = self.layers[i](
                coordinates=current_coords,
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                log_det=log_det,
                reverse=reverse,
                source_temp=source_temp,
                target_temp=target_temp,
                **kwargs
            )
        
        return current_coords, log_det


class PTNVPCouplingLayer(PTFlow, abc.ABC):
    """Base NVP coupling layer for PT flows.
    
    Implements Real NVP coupling transformations with the key adaptations:
    - Position-only (no velocity transformations)
    - Molecular coordinates as input (not latent variables)
    - Graph structure conditioning
    - Temperature awareness for PT
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(
        self,
        coordinates: Tensor,  # [B, N, 3]
        atom_types: Tensor,   # [B, N]
        adj_list: Tensor,     # [E, 2]
        edge_batch_idx: Tensor,  # [E]
        masked_elements: BoolTensor,  # [B, N]
        log_det: LogDetType = None,
        reverse: bool = False,
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tensor, LogDetType]:
        """Apply NVP coupling transformation."""
        if reverse:
            output_coords, layer_log_det = self.flow_reverse(
                coordinates=coordinates,
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                source_temp=source_temp,
                target_temp=target_temp,
                **kwargs
            )
        else:
            output_coords, layer_log_det = self.flow_forward(
                coordinates=coordinates,
                atom_types=atom_types,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                source_temp=source_temp,
                target_temp=target_temp,
                **kwargs
            )
        
        # Accumulate log-determinant
        if log_det is not None:
            log_det = log_det + layer_log_det
        else:
            log_det = layer_log_det
            
        return output_coords, log_det
    
    def flow_forward(
        self,
        coordinates: Tensor,
        atom_types: Tensor,
        adj_list: Tensor,
        edge_batch_idx: Tensor,
        masked_elements: BoolTensor,
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Forward coupling transformation: x → y = s*x + t"""
        scale, shift = self._get_scale_and_shift(
            coordinates=coordinates,
            atom_types=atom_types,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            source_temp=source_temp,
            target_temp=target_temp,
            **kwargs
        )
        
        # Apply affine transformation: y = s * x + t
        output_coords = coordinates * scale + shift
        
        # Compute log-determinant: log|det J| = sum(log(s))
        # Mask out padding elements from log-det computation
        if masked_elements is not None:
            log_scales = torch.log(scale) * (~masked_elements[:, :, None])
        else:
            log_scales = torch.log(scale)
        log_det = torch.sum(log_scales, dim=(-1, -2))  # [B]
        
        return output_coords, log_det
    
    def flow_reverse(
        self,
        coordinates: Tensor,
        atom_types: Tensor,
        adj_list: Tensor,
        edge_batch_idx: Tensor,
        masked_elements: BoolTensor,
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Reverse coupling transformation: y → x = (y - t) / s"""
        scale, shift = self._get_scale_and_shift(
            coordinates=coordinates,
            atom_types=atom_types,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            source_temp=source_temp,
            target_temp=target_temp,
            **kwargs
        )
        
        # Apply inverse affine transformation: x = (y - t) / s
        output_coords = (coordinates - shift) / scale
        
        # Compute log-determinant: log|det J^-1| = -sum(log(s))
        # Mask out padding elements from log-det computation
        if masked_elements is not None:
            log_scales = torch.log(scale) * (~masked_elements[:, :, None])
        else:
            log_scales = torch.log(scale)
        log_det = -torch.sum(log_scales, dim=(-1, -2))  # [B]
        
        return output_coords, log_det
    
    @abc.abstractmethod
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
        """Compute scale and shift parameters for coupling transformation.
        
        Returns
        -------
        scale : Tensor, shape [B, N, 3]
            Per-coordinate scaling factors (must be positive)
        shift : Tensor, shape [B, N, 3]
            Per-coordinate shift values
        """
        pass 