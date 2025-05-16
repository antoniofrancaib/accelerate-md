#!/usr/bin/env python3
"""
Visualize GMM distributions at all temperature levels specified in the config file.

This script loads a GMM configuration and visualizes how the distribution looks
at each temperature level in the parallel tempering schedule.

Usage:
    python visualize_temperature_levels.py --config configs/pt/gmm.yaml
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.accelmd.targets.gmm import GMM
from src.accelmd.utils.config import load_config


def create_gmm_from_config(config, device="cpu"):
    """Create a GMM instance from configuration."""
    gmm_cfg = config.get('gmm', {})
    
    # Initialize GMM with basic parameters
    gmm = GMM(
        dim=gmm_cfg.get('dim', 2),
        n_mixes=gmm_cfg.get('n_mixes', 5),
        loc_scaling=gmm_cfg.get('loc_scaling', 1.0),
        device=device
    )
    
    # Apply configurations based on custom_modes flag
    custom_modes = gmm_cfg.get("custom_modes", True)
    
    with torch.no_grad():
        if custom_modes:
            # Apply standard custom parameters if provided
            if "locations" in gmm_cfg:
                gmm.locs.copy_(torch.tensor(gmm_cfg["locations"], device=device))
                
            if "scales" in gmm_cfg:
                gmm.scale_trils.copy_(torch.tensor(gmm_cfg["scales"], device=device))
                
            if "weights" in gmm_cfg:
                gmm.cat_probs.copy_(torch.tensor(gmm_cfg["weights"], device=device))
        else:
            # Generate evenly-spaced modes on a circle (for 2D only)
            if gmm_cfg.get("dim", 2) != 2:
                raise ValueError("Uniform mode generation is only implemented for dim=2")
                
            n = gmm_cfg.get("n_mixes", 5)
            scale_val = float(gmm_cfg.get("uniform_mode_scale", 0.25))
            
            # Determine mode arrangement
            mode_arrangement = gmm_cfg.get("mode_arrangement", "circle")
            
            if mode_arrangement == "circle":
                # Place modes evenly around a circle
                radius = float(gmm_cfg.get("uniform_mode_radius", 3.0))
                angles = torch.linspace(0, 2 * torch.pi, n + 1, device=device)[:-1]
                locs = torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1)
            
            elif mode_arrangement == "grid":
                # Place modes in a grid pattern
                grid_x_range = gmm_cfg.get("grid_x_range", [-4.0, 4.0])
                grid_y_range = gmm_cfg.get("grid_y_range", [-4.0, 4.0])
                
                # Use specified grid dimensions or calculate them automatically
                if "grid_rows" in gmm_cfg and "grid_cols" in gmm_cfg:
                    rows = int(gmm_cfg.get("grid_rows"))
                    cols = int(gmm_cfg.get("grid_cols"))
                else:
                    # Automatically determine grid dimensions to be approximately square
                    rows = int(np.ceil(np.sqrt(n)))
                    cols = int(np.ceil(n / rows))
                
                # Generate grid positions
                x_points = torch.linspace(grid_x_range[0], grid_x_range[1], cols, device=device)
                y_points = torch.linspace(grid_y_range[0], grid_y_range[1], rows, device=device)
                
                # Create a mesh grid
                grid_x, grid_y = torch.meshgrid(x_points, y_points, indexing='ij')
                grid_positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
                
                # If we have more grid positions than needed, take only the first n
                if grid_positions.shape[0] > n:
                    locs = grid_positions[:n]
                else:
                    # If we need more positions than the grid provides, duplicate some
                    # This shouldn't normally happen if grid dimensions are specified correctly
                    print(f"Warning: Grid dimensions ({rows}x{cols}={rows*cols}) don't provide enough positions for {n} modes. Some modes will overlap.")
                    repeats_needed = int(np.ceil(n / grid_positions.shape[0]))
                    repeated_positions = grid_positions.repeat(repeats_needed, 1)
                    locs = repeated_positions[:n]
            
            else:
                raise ValueError(f"Unknown mode_arrangement: '{mode_arrangement}'. Use 'circle' or 'grid'.")

            gmm.locs.copy_(locs)
            
            # Isotropic covariance with specified scale
            scale_tril = torch.diag(torch.full((gmm_cfg.get("dim", 2),), scale_val, device=device))
            gmm.scale_trils.copy_(torch.stack([scale_tril] * n))
            
            # Uniform mixture weights
            gmm.cat_probs.copy_(torch.full((n,), 1.0 / n, device=device))
    
    return gmm


def compute_temperature_schedule(config):
    """Compute temperature schedule based on config settings."""
    pt_cfg = config.get('pt', {})
    t_low = float(pt_cfg.get('temp_low', 1.0))
    t_high = float(pt_cfg.get('temp_high', 100.0))
    n_temps = int(pt_cfg.get('total_n_temp', 2))
    schedule_type = pt_cfg.get('temp_schedule', 'geom')
    
    if schedule_type == 'linear':
        return np.linspace(t_low, t_high, n_temps)
    else:  # default is geometric
        return np.geomspace(t_low, t_high, n_temps)


def plot_gmm_at_temperature(gmm, temperature, resolution=100, ax=None, title=None):
    """
    Plot GMM distribution at a specific temperature.
    
    Args:
        gmm: Base GMM distribution (T=1.0)
        temperature: Temperature value to visualize
        resolution: Grid resolution for visualization
        ax: Matplotlib axis to plot on (if None, creates a new one)
        title: Plot title (if None, uses default based on temperature)
    
    Returns:
        The matplotlib axis with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Create tempered version of GMM
    tempered_gmm = gmm.tempered_version(temperature, scaling_method='sqrt')
    
    # Get GMM means for plotting
    means = gmm.locs.cpu().numpy()
    
    # Determine appropriate bounds based on the mode locations
    padding = 2.0  # Additional space around the modes
    min_x = np.min(means[:, 0]) - padding
    max_x = np.max(means[:, 0]) + padding
    min_y = np.min(means[:, 1]) - padding
    max_y = np.max(means[:, 1]) + padding
    
    # Create grid for density evaluation
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Prepare grid points for evaluation
    points = np.column_stack([X.flatten(), Y.flatten()])
    points_tensor = torch.tensor(points, dtype=torch.float32, device=gmm.device)
    
    # Evaluate log-probability at each point
    log_probs = tempered_gmm.log_prob(points_tensor).cpu().numpy()
    
    # Convert to probability and reshape to grid
    probs = np.exp(log_probs)
    Z = probs.reshape(X.shape)
    
    # Create contour plot
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    
    # Add mode centers
    ax.scatter(means[:, 0], means[:, 1], c='red', s=60, marker='x', linewidth=2)
    
    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Set title and labels
    if title is None:
        title = f"Temperature T = {temperature:.2f}"
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    return ax, contour


def plot_all_temperature_levels(config, output_dir=None, resolution=100):
    """
    Create a grid of plots showing the GMM at all temperature levels.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save the output figure
        resolution: Grid resolution for density plotting
    """
    # Get device
    device = torch.device(config.get('device', 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        device = torch.device('cpu')
    
    # Create GMM
    gmm = create_gmm_from_config(config, device=device)
    n_modes = gmm.locs.shape[0]
    
    # Get temperature schedule
    temperatures = compute_temperature_schedule(config)
    n_temps = len(temperatures)
    
    # Create a grid of subplots - use a single row if only 2-3 temps,
    # otherwise arrange in an approximately square grid
    if n_temps <= 3:
        fig, axes = plt.subplots(1, n_temps, figsize=(6*n_temps, 5))
    else:
        # Calculate rows and columns for a more square grid
        n_cols = int(np.ceil(np.sqrt(n_temps)))
        n_rows = int(np.ceil(n_temps / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        # Flatten axes array for easy iteration
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each temperature level
    contour = None  # Store the last contour for colorbar
    for i, temp in enumerate(temperatures):
        print(f"Plotting temperature level {i+1}/{n_temps}: T = {temp:.2f}")
        ax = axes[i] if n_temps > 1 else axes
        ax, contour = plot_gmm_at_temperature(
            gmm, 
            temperature=temp,
            resolution=resolution,
            ax=ax,
            title=f"Temperature {i+1}/{n_temps}: T = {temp:.2f}"
        )
    
    # Hide unused subplots if we have extras
    for i in range(n_temps, len(axes)):
        if n_temps > 1:  # Check if axes is a list (for multiple temps)
            axes[i].set_visible(False)
    
    # Add a colorbar using the last contour
    if contour is not None:
        cbar = fig.colorbar(contour, ax=axes, shrink=0.7)
        cbar.set_label('Probability Density')
    
    # Set title for the figure
    custom_modes = config.get('gmm', {}).get('custom_modes', True)
    mode_type = "custom" if custom_modes else "uniform"
    fig.suptitle(f'GMM with {n_modes} {mode_type} modes at {n_temps} temperature levels', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save figure if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        config_name = "gmm"  # Default name
        
        # Try to extract config file name if it was loaded from a file
        if hasattr(config, '_config_path'):
            config_name = os.path.splitext(os.path.basename(config._config_path))[0]
        
        output_path = os.path.join(output_dir, f"{config_name}_temperature_levels.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return fig


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Visualize GMM at all temperature levels')
    parser.add_argument('--config', type=str, default='configs/pt/gmm.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='plots/temperatures',
                       help='Directory to save output figures')
    parser.add_argument('--resolution', type=int, default=100,
                       help='Resolution for density grid (higher = smoother but slower)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Plot GMM at all temperature levels
    fig = plot_all_temperature_levels(
        config, 
        output_dir=args.output_dir,
        resolution=args.resolution
    )
    
    plt.show()


if __name__ == "__main__":
    main() 