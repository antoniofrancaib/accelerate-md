"""
Utility functions for generating GMM mode locations in arbitrary dimensions.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_gmm_modes(n_mixes, dim, arrangement, cfg, device):
    """
    Generate GMM mode locations in arbitrary dimensions.
    
    Args:
        n_mixes: Number of mixture components
        dim: Dimensionality of the space
        arrangement: Mode arrangement strategy ('grid', 'circle', or 'random')
        cfg: Configuration dictionary with arrangement parameters
        device: Torch device for tensor creation
        
    Returns:
        Tensor of mode locations with shape (n_mixes, dim)
    """
    if arrangement == "grid":
        if dim == 2:
            # Use original 2D grid logic for backward compatibility
            grid_x_range = cfg.get("grid_x_range", [-4.0, 4.0])
            grid_y_range = cfg.get("grid_y_range", [-4.0, 4.0])
            
            # Use specified grid dimensions or calculate them automatically
            if "grid_rows" in cfg and "grid_cols" in cfg:
                rows = int(cfg["grid_rows"])
                cols = int(cfg["grid_cols"])
            else:
                # Automatically determine grid dimensions
                rows = int(np.ceil(np.sqrt(n_mixes)))
                cols = int(np.ceil(n_mixes / rows))
            
            # Generate grid positions
            x_points = torch.linspace(grid_x_range[0], grid_x_range[1], cols, device=device)
            y_points = torch.linspace(grid_y_range[0], grid_y_range[1], rows, device=device)
            
            # Create a mesh grid
            grid_x, grid_y = torch.meshgrid(x_points, y_points, indexing='ij')
            grid_positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            
            # Take the first n_mixes positions
            if grid_positions.shape[0] >= n_mixes:
                locs = grid_positions[:n_mixes]
            else:
                # If grid is too small, duplicate some positions
                repeats_needed = int(np.ceil(n_mixes / grid_positions.shape[0]))
                repeated_positions = grid_positions.repeat(repeats_needed, 1)
                locs = repeated_positions[:n_mixes]
        else:
            # Higher-dimensional grid: for D>3 the mesh grid approach becomes exponentially expensive
            # Instead, we'll generate a set of points along each axis and combine them strategically
            
            # For higher dimensions, we need to be careful about memory usage
            # Calculate how many points per dimension we can afford
            # We want roughly n_mixes points, but n_mixes^(1/dim) might be too large
            # We'll cap at a reasonable number
            max_points_per_dim = 5  # Cap to avoid memory explosion
            points_per_dim = min(int(np.ceil(n_mixes ** (1 / dim))), max_points_per_dim)
            
            print(f"[DEBUG] Grid arrangement in {dim}D with {points_per_dim} points per dimension")
            if points_per_dim**dim < n_mixes:
                logger.warning(f"Warning: Grid with {points_per_dim} points per dimension ({points_per_dim**dim} total) " +
                              f"doesn't provide enough points for {n_mixes} modes.")
            
            # Memory-efficient approach: generate random indices for each dimension
            # This is instead of creating a full meshgrid which would be too large
            locs = torch.zeros((n_mixes, dim), device=device)
            
            # Create coordinate arrays for each dimension
            dim_arrays = []
            for d in range(dim):
                range_key = f"grid_dim{d}_range"
                dim_range = cfg.get(range_key, [-4.0, 4.0])
                dim_arrays.append(torch.linspace(dim_range[0], dim_range[1], points_per_dim, device=device))
            
            # Fill with selected grid points - either take all available points or sample randomly
            if points_per_dim**dim <= n_mixes:
                # Take all available grid points
                all_indices = torch.cartesian_prod(*[torch.arange(points_per_dim, device=device) for _ in range(dim)])
                for i in range(min(n_mixes, all_indices.shape[0])):
                    for d in range(dim):
                        locs[i, d] = dim_arrays[d][all_indices[i, d]]
                
                # If we need more points, duplicate some with small offsets
                if all_indices.shape[0] < n_mixes:
                    remaining = n_mixes - all_indices.shape[0]
                    duplicated_indices = all_indices[torch.randint(0, all_indices.shape[0], (remaining,), device=device)]
                    small_noise = torch.randn((remaining, dim), device=device) * 0.1
                    
                    for i in range(remaining):
                        for d in range(dim):
                            locs[all_indices.shape[0] + i, d] = dim_arrays[d][duplicated_indices[i, d]] + small_noise[i, d]
            else:
                # Randomly sample from the grid
                for i in range(n_mixes):
                    random_indices = torch.randint(0, points_per_dim, (dim,), device=device)
                    for d in range(dim):
                        locs[i, d] = dim_arrays[d][random_indices[d]]
            
    elif arrangement == "circle":
        if dim == 2:
            # 2D circle arrangement
            radius = float(cfg.get("uniform_mode_radius", 3.0))
            angles = torch.linspace(0, 2 * torch.pi, n_mixes + 1, device=device)[:-1]
            locs = torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1)
        else:
            # Higher dimensions: use points on a hypersphere
            radius = float(cfg.get("uniform_mode_radius", 3.0))
            # Generate random points on unit hypersphere
            points = torch.randn(n_mixes, dim, device=device)
            # Normalize to get unit vectors
            points = points / torch.norm(points, dim=1, keepdim=True)
            # Scale by radius
            locs = radius * points
            
    elif arrangement == "random":
        # Random locations with specified scaling
        scale = float(cfg.get("loc_scaling", 3.0))
        locs = (torch.rand(n_mixes, dim, device=device) - 0.5) * 2 * scale
        
    else:
        raise ValueError(f"Unknown mode_arrangement: '{arrangement}'. Use 'circle', 'grid', or 'random'.")
    
    print(f"[DEBUG] Generated {locs.shape[0]} mode locations in {dim}D space")
    return locs 