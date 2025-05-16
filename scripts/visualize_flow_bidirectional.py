#!/usr/bin/env python3
"""
Visualize bidirectional RealNVP flow transformations between temperature levels.

This script loads a trained RealNVP flow model and visualizes how it transforms
samples between different temperature distributions. It demonstrates the model's
ability to map between low and high temperature distributions in both directions.

Usage:
    python visualize_flow_bidirectional.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.accelmd.utils.config import load_config
from src.accelmd.targets.gmm import GMM
from src.accelmd.models.realnvp import create_realnvp_flow


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
            # Generate evenly-spaced modes (2-D specific)
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
                    repeats_needed = int(np.ceil(n / grid_positions.shape[0]))
                    repeated_positions = grid_positions.repeat(repeats_needed, 1)
                    locs = repeated_positions[:n]
            else:
                raise ValueError(f"Unknown mode_arrangement: '{mode_arrangement}'. Use 'circle' or 'grid'.")
                
            gmm.locs.copy_(locs)
            
            # Isotropic covariance with specified scale
            scale_tril = torch.diag(torch.full((2,), scale_val, device=device))
            gmm.scale_trils.copy_(torch.stack([scale_tril] * n))
            
            # Uniform mixture weights
            gmm.cat_probs.copy_(torch.full((n,), 1.0 / n, device=device))
    
    return gmm


def plot_density_contours(ax, distribution, x_range=(-5, 5), y_range=(-5, 5), resolution=100, cmap='viridis', alpha=0.7, title=None):
    """Plot density contours for a distribution on a matplotlib axis."""
    # Create grid for contour plot
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Prepare grid points for evaluation
    points = np.column_stack([X.flatten(), Y.flatten()])
    # Get device from the device attribute or component_distribution's device
    if hasattr(distribution, 'device'):
        device = distribution.device
    elif hasattr(distribution, 'component_distribution') and hasattr(distribution.component_distribution, 'device'):
        device = distribution.component_distribution.device
    else:
        # Default to CPU if no device can be determined
        device = torch.device('cpu')
    
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    # Compute log probabilities
    with torch.no_grad():
        try:
            # For GMM and similar distributions with log_prob method
            log_probs = distribution.log_prob(points_tensor).cpu().numpy()
        except AttributeError:
            # For a raw tensor of samples, use KDE (kernel density estimation)
            from scipy.stats import gaussian_kde
            samples = distribution.cpu().numpy()
            kde = gaussian_kde(samples.T)
            log_probs = np.log(kde(points.T) + 1e-10)  # Add small epsilon to avoid log(0)
    
    # Convert to probabilities and reshape
    probs = np.exp(log_probs)
    Z = probs.reshape(X.shape)
    
    # Plot contours
    contour = ax.contourf(X, Y, Z, levels=20, cmap=cmap, alpha=alpha)
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    return contour


def plot_scatter_with_arrows(ax, source_samples, target_samples, source_color='blue', target_color='red', 
                            arrow_color='gray', alpha=0.6, s=10, plot_arrows=True, max_arrows=100):
    """Plot source and target samples with arrows showing the transformation."""
    # Plot samples
    ax.scatter(source_samples[:, 0], source_samples[:, 1], c=source_color, s=s, alpha=alpha, marker='o')
    ax.scatter(target_samples[:, 0], target_samples[:, 1], c=target_color, s=s, alpha=alpha, marker='x')
    
    # Add arrows for a subset of points
    if plot_arrows and len(source_samples) > 0:
        n_arrows = min(max_arrows, len(source_samples))
        indices = np.random.choice(len(source_samples), n_arrows, replace=False)
        
        for idx in indices:
            ax.arrow(
                source_samples[idx, 0], source_samples[idx, 1],
                target_samples[idx, 0] - source_samples[idx, 0],
                target_samples[idx, 1] - source_samples[idx, 1],
                head_width=0.1, head_length=0.15, fc=arrow_color, ec=arrow_color, alpha=0.5
            )


def main():
    """Main function to create the bidirectional flow visualization."""
    # Config and checkpoint paths
    config_path = 'configs/pt/gmm.yaml'
    ckpt_path = '/home/jaf98/rds/hpc-work/accelerate-md/checkpoints/realnvp_gmm/flow_1.00_to_10.00.pt'
    output_file = 'bidirectional_verification_5modes.png'
    
    # Set device to CPU for visualization
    device = torch.device('cpu')
    
    # Load configuration
    config = load_config(config_path)
    
    # Get temperature information
    t_low = float(config['pt']['temp_low'])
    t_high = float(config['pt']['temp_high'])
    
    # Create GMM distribution for base temperature
    gmm = create_gmm_from_config(config, device=device)
    n_modes = gmm.locs.shape[0]
    print(f"Created GMM with {n_modes} modes")
    
    # Create high temperature version
    hi_gmm = gmm.tempered_version(t_high, scaling_method='sqrt')
    
    # Create RealNVP model
    model_cfg = config.get('trainer', {}).get('realnvp', {}).get('model', {})
    flow = create_realnvp_flow(model_cfg).to(device)
    
    # Load model weights
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    flow.load_state_dict(checkpoint)
    flow.eval()
    
    # Set number of samples for visualization
    n_samples = 1000
    
    # 1. Generate samples from each temperature
    print("Generating GMM samples...")
    with torch.no_grad():
        # Sample from both distributions
        low_temp_samples = gmm.sample((n_samples,))
        high_temp_samples = hi_gmm.sample((n_samples,))
        
        # 2. Apply flow transformations
        print("Applying flow transformations...")
        # Low → High (forward)
        transformed_to_high, _ = flow.forward(low_temp_samples)
        # High → Low (inverse)
        transformed_to_low, _ = flow.inverse(high_temp_samples)
    
    # 3. Create the visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=(18, 12))
    
    # Determine appropriate plot ranges
    all_samples = torch.cat([
        low_temp_samples, high_temp_samples, 
        transformed_to_high, transformed_to_low
    ], dim=0)
    x_min, y_min = all_samples.min(dim=0)[0].numpy() - 1
    x_max, y_max = all_samples.max(dim=0)[0].numpy() + 1
    plot_range = (x_min, x_max), (y_min, y_max)
    
    # TOP ROW: Original distributions
    # --------------------------------
    # Low temperature distribution
    ax1 = fig.add_subplot(2, 3, 1)
    plot_density_contours(ax1, gmm, x_range=plot_range[0], y_range=plot_range[1], 
                          title=f"Low Temp (T={t_low}) Distribution")
    ax1.scatter(low_temp_samples[:, 0], low_temp_samples[:, 1], 
                c='blue', s=10, alpha=0.6, marker='o')
    ax1.set_xlim(plot_range[0])
    ax1.set_ylim(plot_range[1])
    
    # High temperature distribution
    ax2 = fig.add_subplot(2, 3, 3)
    plot_density_contours(ax2, hi_gmm, x_range=plot_range[0], y_range=plot_range[1], 
                          title=f"High Temp (T={t_high}) Distribution")
    ax2.scatter(high_temp_samples[:, 0], high_temp_samples[:, 1], 
                c='red', s=10, alpha=0.6, marker='x')
    ax2.set_xlim(plot_range[0])
    ax2.set_ylim(plot_range[1])
    
    # BOTTOM ROW: Flow transformations
    # --------------------------------
    # Low → High transformation
    ax4 = fig.add_subplot(2, 3, 4)
    plot_density_contours(ax4, gmm, x_range=plot_range[0], y_range=plot_range[1], 
                          alpha=0.3, title=f"Low → High (Forward)")
    plot_scatter_with_arrows(ax4, low_temp_samples, transformed_to_high, 
                            source_color='blue', target_color='red')
    ax4.set_xlim(plot_range[0])
    ax4.set_ylim(plot_range[1])
    
    # High → Low transformation
    ax6 = fig.add_subplot(2, 3, 6)
    plot_density_contours(ax6, hi_gmm, x_range=plot_range[0], y_range=plot_range[1], 
                          alpha=0.3, title=f"High → Low (Inverse)")
    plot_scatter_with_arrows(ax6, high_temp_samples, transformed_to_low, 
                            source_color='red', target_color='blue')
    ax6.set_xlim(plot_range[0])
    ax6.set_ylim(plot_range[1])
    
    # CENTER: Comparison plots
    # --------------------------------
    # Actual vs Transformed (Low)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(low_temp_samples[:, 0], low_temp_samples[:, 1], 
                c='blue', s=10, alpha=0.4, marker='o', label='Real Low Temp')
    ax5.scatter(transformed_to_low[:, 0], transformed_to_low[:, 1], 
                c='green', s=10, alpha=0.4, marker='+', label='Flow: High→Low')
    ax5.set_title("Flow Accuracy: Low Temperature")
    ax5.set_xlim(plot_range[0])
    ax5.set_ylim(plot_range[1])
    ax5.legend()
    
    # Actual vs Transformed (High)
    ax3 = fig.add_subplot(2, 3, 2)
    ax3.scatter(high_temp_samples[:, 0], high_temp_samples[:, 1], 
                c='red', s=10, alpha=0.4, marker='x', label='Real High Temp')
    ax3.scatter(transformed_to_high[:, 0], transformed_to_high[:, 1], 
                c='green', s=10, alpha=0.4, marker='+', label='Flow: Low→High')
    ax3.set_title("Flow Accuracy: High Temperature")
    ax3.set_xlim(plot_range[0])
    ax3.set_ylim(plot_range[1])
    ax3.legend()
    
    # Add a global title
    fig.suptitle(f"Bidirectional RealNVP Flow: T={t_low} ↔ T={t_high} Transformations ({n_modes} modes)", 
                fontsize=16)
    
    # Save and show the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    print(f"Saving visualization to {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print("Done!")


if __name__ == "__main__":
    main() 