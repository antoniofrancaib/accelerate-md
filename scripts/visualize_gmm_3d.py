#!/usr/bin/env python3
"""
GMM Contour Plots with MCMC Samples based on configuration from gmm.yaml
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Add the parent directory to the path to import main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.accelmd.targets.gmm import GMM
from src.accelmd.sampler.sampler import LangevinDynamics
from src.accelmd.utils.config import load_config


def create_gmm_from_config(config, device="cpu"):
    """Create a Gaussian Mixture Model from configuration."""
    gmm_config = config.get('gmm', {})
    
    # Initialize GMM with basic parameters
    gmm = GMM(
        dim=gmm_config.get('dim', 2),
        n_mixes=gmm_config.get('n_mixes', 5),
        loc_scaling=gmm_config.get('loc_scaling', 1.0),
        device=device
    )
    
    # Apply custom configuration
    with torch.no_grad():
        # Apply custom locations if provided
        if 'locations' in gmm_config:
            custom_locs = torch.tensor(gmm_config['locations'], device=device)
            gmm.locs.copy_(custom_locs)
            
        # Apply custom scales if provided
        if 'scales' in gmm_config:
            custom_scales = torch.tensor(gmm_config['scales'], device=device)
            gmm.scale_trils.copy_(custom_scales)
            
        # Apply custom weights if provided
        if 'weights' in gmm_config:
            custom_weights = torch.tensor(gmm_config['weights'], device=device)
            gmm.cat_probs.copy_(custom_weights)
    
    return gmm


def run_mcmc_sampling(gmm, n_samples=1000, step_size=0.5, burn_in=200, replica_seed=42):
    """
    Run MCMC sampling using LangevinDynamics from the existing codebase
    
    Args:
        gmm: The GMM distribution object
        n_samples: Number of samples to generate
        step_size: Step size for the sampler
        burn_in: Number of initial samples to discard
        replica_seed: Seed for this replica
        
    Returns:
        All samples including burn-in (for trajectory visualization)
    """
    # Set seed for this replica
    torch.manual_seed(replica_seed)
    
    # Create initial state
    initial_state = torch.zeros((1, gmm.dim), device=gmm.device)
    
    # Define energy function
    def energy_func(x):
        return -gmm.log_prob(x)
    
    # Initialize sampler
    sampler = LangevinDynamics(
        x=initial_state,
        energy_func=energy_func,
        step_size=step_size,
        mh=True,  # Use Metropolis-Hastings correction
        device=gmm.device
    )
    
    # Collect all samples including burn-in
    all_samples = [initial_state.clone()]
    
    # Run sampling
    for i in range(n_samples + burn_in):
        sample, _ = sampler.sample()
        all_samples.append(sample.clone())
    
    # Stack all samples
    all_samples = torch.cat(all_samples, dim=0)
    
    # Return all samples and post-burn-in samples
    return all_samples, all_samples[burn_in:]


def determine_plot_bounds(samples_list, means, padding=1.0):
    """
    Determine appropriate bounds for plotting based on multiple sample sets and means.
    
    Args:
        samples_list: List of tensors of samples from the distribution
        means: Tensor of GMM component means
        padding: Extra padding around the data range
        
    Returns:
        x_bounds: (min_x, max_x) for plotting
        y_bounds: (min_y, max_y) for plotting
    """
    # Convert means to numpy
    means_np = means.cpu().numpy()
    
    # Convert all samples to numpy and combine
    all_points = [means_np]
    for samples in samples_list:
        all_points.append(samples.cpu().numpy())
    
    all_points = np.vstack(all_points)
    
    # Get min and max for each dimension
    min_x, min_y = np.min(all_points, axis=0) - padding
    max_x, max_y = np.max(all_points, axis=0) + padding
    
    return (min_x, max_x), (min_y, max_y)


def plot_gmm_contours_side_by_side(gmm, samples_list, replica_labels, bounds=None, resolution=100):
    """
    Create side-by-side contour plots of the GMM distribution with MCMC samples.
    
    Args:
        gmm: The GMM distribution object
        samples_list: List of tensors containing MCMC samples for each replica
        replica_labels: Labels for each replica
        bounds: The (min, max) bounds for both x and y axes, or None to auto-determine
        resolution: Grid resolution for visualization
    """
    # Get GMM means
    means = gmm.locs
    
    # Auto-determine bounds if not provided
    if bounds is None:
        x_bounds, y_bounds = determine_plot_bounds(samples_list, means)
    else:
        x_bounds = y_bounds = (bounds[0], bounds[1])
    
    # Create a meshgrid for visualization
    x = np.linspace(x_bounds[0], x_bounds[1], resolution)
    y = np.linspace(y_bounds[0], y_bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid points
    points = np.column_stack([X.flatten(), Y.flatten()])
    points_tensor = torch.tensor(points, dtype=torch.float32, device=gmm.device)
    
    # Evaluate probabilities
    log_probs = gmm.log_prob(points_tensor).cpu().numpy()
    probs = np.exp(log_probs)
    Z = probs.reshape(X.shape)
    
    # Convert means to numpy
    means_np = means.cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(samples_list), figsize=(18, 6), constrained_layout=True)
    
    # Handle the case when there's only one replica
    if len(samples_list) == 1:
        axes = [axes]
    
    # Plot each replica in its own subplot
    for i, (samples, label, ax) in enumerate(zip(samples_list, replica_labels, axes)):
        # Plot contour
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        
        # Plot MCMC samples
        samples_np = samples.cpu().numpy()
        ax.scatter(samples_np[:, 0], samples_np[:, 1], c='red', s=10, alpha=0.5, marker='o')
        
        # Plot GMM component means
        ax.scatter(means_np[:, 0], means_np[:, 1], c='yellow', s=80, marker='x', linewidth=2)
        
        # Set axis limits to match the calculated bounds
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(label)
    
    # Add a common colorbar
    cbar = fig.colorbar(contour, ax=axes, shrink=0.6)
    cbar.set_label('Probability Density')
    
    # Set an overall title
    fig.suptitle('GMM Contour Plots with MCMC Samples based on gmm.yaml config', fontsize=16)
    
    return fig


def main():
    """Main function to visualize GMM contour with MCMC samples based on config file."""
    # Create output directory
    os.makedirs("gmm_visualizations", exist_ok=True)
    
    # Set device to CPU for visualization
    device = "cpu"
    
    # Load configuration from YAML file
    config_path = "configs/pt/gmm.yaml"
    config = load_config(config_path)
    
    # Force CPU device for visualization regardless of config
    print(f"Using CPU device for visualization")
    
    # Create GMM from config
    gmm = create_gmm_from_config(config, device=device)
    
    # Get the number of modes from the loaded GMM
    n_modes = gmm.locs.shape[0]
    print(f"Loaded GMM with {n_modes} modes from {config_path}")
    
    # Get sampling parameters from config if available
    pt_config = config.get('pt', {})
    step_size = pt_config.get('step_size', 0.5)
    
    # Define replica parameters
    replica_configs = [
        {"step_size": step_size, "seed": 42, "label": f"GMM with {n_modes} modes"}
    ]
    
    # Run sampling for each replica
    all_samples_list = []
    post_burnin_samples_list = []
    replica_labels = []
    
    for config in replica_configs:
        print(f"Running {config['label']}...")
        all_samples, post_burnin_samples = run_mcmc_sampling(
            gmm=gmm,
            n_samples=1000,
            step_size=config["step_size"],
            burn_in=200,
            replica_seed=config["seed"]
        )
        
        all_samples_list.append(all_samples)
        post_burnin_samples_list.append(post_burnin_samples)
        replica_labels.append(config["label"])
    
    # Create contour plot with samples
    fig = plot_gmm_contours_side_by_side(
        gmm=gmm,
        samples_list=post_burnin_samples_list,
        replica_labels=replica_labels,
        bounds=None
    )
    
    # Save the plot
    output_path = "gmm_visualizations/gmm_contours_from_config.png"
    fig.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    # Display the plot
    plt.show()


if __name__ == "__main__":
    main() 