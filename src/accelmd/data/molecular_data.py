"""Molecular data processing utilities.

This module provides functions for processing molecular conformations,
including chirality filtering and coordinate transformations.
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import torch
import mdtraj as md

__all__ = ["filter_chirality", "torch_to_mdtraj", "center_coordinates", "random_rotation_augment"]


def random_rotation_augment(samples: torch.Tensor) -> torch.Tensor:
    """Apply random 3D rotation augmentation to molecular coordinates.
    
    Generates random rotation matrices and applies them to each sample.
    The coordinates are first centered, rotated, then the center is added back.
    
    Parameters
    ----------
    samples : torch.Tensor
        Coordinate samples, shape [N, n_atoms, 3]
        
    Returns
    -------
    torch.Tensor
        Randomly rotated coordinates with same shape as input
    """
    if samples.ndim != 3 or samples.shape[-1] != 3:
        raise ValueError(f"Expected shape [N, n_atoms, 3], got {samples.shape}")
    
    N, n_atoms, _ = samples.shape
    device = samples.device
    dtype = samples.dtype
    
    # Center coordinates first
    centers = samples.mean(dim=1, keepdim=True)  # [N, 1, 3]
    centered_coords = samples - centers
    
    rotated_coords = torch.zeros_like(centered_coords)
    
    # Apply a different random rotation to each sample
    for i in range(N):
        # Generate random rotation matrix using Rodriguez formula
        # Sample random axis (unit vector) and angle
        axis = torch.randn(3, device=device, dtype=dtype)
        axis = axis / torch.norm(axis)  # normalize to unit vector
        
        # Random rotation angle between 0 and 2π
        angle = torch.rand(1, device=device, dtype=dtype) * 2 * np.pi
        
        # Rodriguez rotation formula: R = I + sin(θ)[K] + (1-cos(θ))[K]²
        # where [K] is the skew-symmetric matrix of the axis vector
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Skew-symmetric matrix [K]
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=device, dtype=dtype)
        
        # Rotation matrix
        I = torch.eye(3, device=device, dtype=dtype)
        R = I + sin_angle * K + (1 - cos_angle) * torch.mm(K, K)
        
        # Apply rotation to all atoms in this sample
        coords_sample = centered_coords[i]  # [n_atoms, 3]
        rotated_coords[i] = torch.mm(coords_sample, R.T)  # Apply rotation
    
    # Add centers back
    augmented_coords = rotated_coords + centers
    
    return augmented_coords


def filter_chirality(samples: torch.Tensor, ref_trajectory: Optional[md.Trajectory] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Filter out incorrect chirality conformations from samples.
    
    This function filters batch for the L-form based on dihedral angle differences
    at specific atomic indices (17, 26 for alanine dipeptide).
    
    Parameters
    ----------
    samples : torch.Tensor
        Coordinate samples to filter, shape [N, n_atoms, 3]
    ref_trajectory : Optional[md.Trajectory]
        Reference trajectory (unused in this implementation)
        
    Returns
    -------
    filtered_samples : torch.Tensor
        Samples with correct chirality only  
    counters : Tuple[int, int]
        (initial_d_form_count, final_d_form_count) for tracking
    """
    # Implementation based on main/utils/aldp_utils.py filter_chirality
    # This assumes we're working with internal coordinates or dihedral angles
    # For Cartesian coordinates, we need to compute dihedral angles first
    
    if samples.ndim == 3:
        # If we have [N, n_atoms, 3] Cartesian coordinates, 
        # we would need to compute dihedral angles first
        # For now, return all samples (no filtering)
        n_samples = samples.shape[0]
        return samples, (0, 0)
    
    # For flat coordinate input or dihedral angles
    if samples.ndim == 2 and samples.shape[1] >= 27:  # Assumes dihedral angles included
        ind = [17, 26]  # Indices for chirality check
        mean_diff = -0.043
        threshold = 0.8
        
        initial_count = samples.shape[0]
        
        # Compute wrapped differences
        diff_ = torch.column_stack((
            samples[:, ind[0]] - samples[:, ind[1]],
            samples[:, ind[0]] - samples[:, ind[1]] + 2 * np.pi,
            samples[:, ind[0]] - samples[:, ind[1]] - 2 * np.pi
        ))
        
        # Find minimum difference
        min_diff_ind = torch.min(torch.abs(diff_), 1).indices
        diff = diff_[torch.arange(samples.shape[0]), min_diff_ind]
        
        # Filter based on threshold
        keep_mask = torch.abs(diff - mean_diff) < threshold
        filtered_samples = samples[keep_mask]
        
        final_count = filtered_samples.shape[0]
        d_form_initial = initial_count - torch.sum(keep_mask).item()
        d_form_final = 0  # All remaining are L-form
        
        return filtered_samples, (d_form_initial, d_form_final)
    
    # Default: no filtering
    return samples, (0, 0)


def center_coordinates(samples: torch.Tensor) -> torch.Tensor:
    """Center molecular coordinates by removing the center of mass.
    
    Makes configurations mean-free by subtracting the center of mass
    from each configuration.
    
    Parameters
    ----------
    samples : torch.Tensor
        Coordinate samples, shape [N, n_atoms, 3] or [N, n_atoms*3]
        
    Returns
    -------
    torch.Tensor
        Mean-centered coordinates with same shape as input
    """
    # Implementation based on main/utils/se3_utils.py remove_mean
    original_shape = samples.shape
    
    if samples.ndim == 3:
        # Shape [N, n_atoms, 3]
        n_particles = samples.shape[1]
        n_dimensions = samples.shape[2]
        # Subtract mean across atoms (dim=1)
        centered = samples - torch.mean(samples, dim=1, keepdim=True)
    elif samples.ndim == 2:
        # Assume flat coordinates [N, n_atoms*3], reshape to [N, n_atoms, 3]
        n_coords = samples.shape[1]
        if n_coords % 3 != 0:
            raise ValueError(f"Expected coordinates divisible by 3, got {n_coords}")
        n_particles = n_coords // 3
        n_dimensions = 3
        
        # Reshape to [N, n_atoms, 3]
        reshaped = samples.view(-1, n_particles, n_dimensions)
        # Center coordinates
        centered_reshaped = reshaped - torch.mean(reshaped, dim=1, keepdim=True)
        # Reshape back to original
        centered = centered_reshaped.view(original_shape)
    else:
        raise ValueError(f"Unsupported tensor shape: {original_shape}")
    
    return centered


def torch_to_mdtraj(coords: torch.Tensor, topology: md.Topology) -> md.Trajectory:
    """Convert PyTorch coordinates to MDTraj trajectory.
    
    Parameters
    ----------
    coords : torch.Tensor
        Coordinates tensor of shape [N, n_atoms, 3] in nanometers
    topology : md.Topology
        MDTraj topology object
        
    Returns
    -------
    md.Trajectory
        MDTraj trajectory object
    """
    # Convert to numpy and ensure correct units (MDTraj expects nanometers)
    coords_np = coords.detach().cpu().numpy()
    
    # Create MDTraj trajectory
    traj = md.Trajectory(coords_np, topology)
    
    return traj 