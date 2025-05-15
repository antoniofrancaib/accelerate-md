#!/usr/bin/env python3
"""
GMM Contour Plots with MCMC Samples based on configuration from gmm.yaml or other config files
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path

# Add the parent directory to the path to import main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.accelmd.targets.gmm import GMM
from src.accelmd.samplers.mcmc.langevin import LangevinDynamics
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


def plot_gmm_contours_side_by_side(gmm, samples_list, replica_labels, bounds=None, resolution=100, figsize=(18, 6)):
    """
    Create side-by-side contour plots of the GMM distribution with MCMC samples.
    
    Args:
        gmm: The GMM distribution object
        samples_list: List of tensors containing MCMC samples for each replica
        replica_labels: Labels for each replica
        bounds: The (min, max) bounds for both x and y axes, or None to auto-determine
        resolution: Grid resolution for visualization
        figsize: Figure size (width, height) in inches
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
    fig, axes = plt.subplots(1, len(samples_list), figsize=figsize, constrained_layout=True)
    
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
        # For high number of modes, use smaller markers
        marker_size = 80 if gmm.n_mixes < 10 else 40 if gmm.n_mixes < 20 else 20
        ax.scatter(means_np[:, 0], means_np[:, 1], c='yellow', s=marker_size, marker='x', linewidth=2)
        
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
    fig.suptitle(f'GMM Contour Plot with {gmm.n_mixes} modes', fontsize=16)
    
    return fig


def plot_gmm_temp_comparison(gmm, high_temp_value=50.0, resolution=100, figsize=(18, 6)):
    """
    Create side-by-side contour plots showing the GMM at different temperatures.
    
    Args:
        gmm: The base GMM distribution object (temp=1.0)
        high_temp_value: The higher temperature to compare with
        resolution: Grid resolution for visualization
        figsize: Figure size (width, height) in inches
    """
    # Create high-temperature version of the GMM
    high_temp_gmm = gmm.tempered_version(high_temp_value)
    
    # Get bounds that encompass both distributions
    means = gmm.locs
    bounds_padding = 1.0 if gmm.n_mixes < 10 else 1.5 if gmm.n_mixes < 20 else 2.0
    
    # Convert means to numpy and determine bounds
    means_np = means.cpu().numpy()
    min_x, min_y = np.min(means_np, axis=0) - bounds_padding
    max_x, max_y = np.max(means_np, axis=0) + bounds_padding
    x_bounds, y_bounds = (min_x, max_x), (min_y, max_y)
    
    # Create a meshgrid for visualization
    x = np.linspace(x_bounds[0], x_bounds[1], resolution)
    y = np.linspace(y_bounds[0], y_bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid points
    points = np.column_stack([X.flatten(), Y.flatten()])
    points_tensor = torch.tensor(points, dtype=torch.float32, device=gmm.device)
    
    # Evaluate probabilities for both temperatures
    log_probs_low = gmm.log_prob(points_tensor).cpu().numpy()
    probs_low = np.exp(log_probs_low)
    Z_low = probs_low.reshape(X.shape)
    
    log_probs_high = high_temp_gmm.log_prob(points_tensor).cpu().numpy()
    probs_high = np.exp(log_probs_high)
    Z_high = probs_high.reshape(X.shape)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    
    # Titles for the subplots
    titles = [f"Temperature T={1.0}", f"Temperature T={high_temp_value}"]
    data = [Z_low, Z_high]
    
    # Plot each temperature in its own subplot
    for i, (Z, title, ax) in enumerate(zip(data, titles, axes)):
        # Plot contour
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        
        # Plot GMM component means
        marker_size = 80 if gmm.n_mixes < 10 else 40 if gmm.n_mixes < 20 else 20
        ax.scatter(means_np[:, 0], means_np[:, 1], c='red', s=marker_size, marker='x', linewidth=2)
        
        # Set axis limits
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
    
    # Add a common colorbar
    cbar = fig.colorbar(contour, ax=axes, shrink=0.6)
    cbar.set_label('Probability Density')
    
    # Set an overall title
    fig.suptitle(f'GMM Distribution at Different Temperatures ({gmm.n_mixes} modes)', fontsize=16)
    
    return fig


def main():
    """Main function to visualize GMM contour with MCMC samples based on config file."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize GMM configuration with contour plots')
    parser.add_argument('--config', type=str, default='configs/pt/gmm.yaml',
                       help='Path to GMM configuration YAML file (default: configs/pt/gmm.yaml)')
    parser.add_argument('--output-dir', type=str, default='gmm_visualizations',
                       help='Directory to save visualizations (default: gmm_visualizations)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of MCMC samples to generate (default: 1000)')
    parser.add_argument('--no-mcmc', action='store_true',
                       help='Skip MCMC sampling and only show distribution contours')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device to CPU for visualization
    device = "cpu"
    
    # Extract config filename for output naming
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    print(f"Using CPU device for visualization")
    
    # Create GMM from config
    gmm = create_gmm_from_config(config, device=device)
    
    # Get the number of modes from the loaded GMM
    n_modes = gmm.locs.shape[0]
    print(f"Loaded GMM with {n_modes} modes from {args.config}")
    
    # Determine appropriate figure size based on number of modes
    figsize = (18, 6) if n_modes < 10 else (22, 8)
    
    # Get temperature info from config
    pt_config = config.get('pt', {})
    temp_high = float(pt_config.get('temp_high', 50.0))
    
    # First, create a temperature comparison plot (low vs high temp)
    fig_temps = plot_gmm_temp_comparison(gmm, high_temp_value=temp_high, 
                                        resolution=150, figsize=figsize)
    
    # Save the temperature comparison plot
    temp_output_path = os.path.join(args.output_dir, f"{config_name}_temperature_comparison.png")
    fig_temps.savefig(temp_output_path, dpi=300)
    print(f"Temperature comparison plot saved to {temp_output_path}")
    
    # If MCMC sampling is enabled
    if not args.no_mcmc:
        # Get sampling parameters from config
        step_size = pt_config.get('step_size', 0.001)
        
        # Adjust step size for high number of modes
        if n_modes > 10:
            step_size = min(step_size, 0.0005)  # Use smaller step size for complex distributions
        
        # Define replica parameters
        replica_configs = [
            {"step_size": step_size, "seed": 42, "label": f"GMM with {n_modes} modes"}
        ]
        
        # Run sampling for each replica
        all_samples_list = []
        post_burnin_samples_list = []
        replica_labels = []
        
        for config in replica_configs:
            print(f"Running MCMC sampling for {config['label']}...")
            all_samples, post_burnin_samples = run_mcmc_sampling(
                gmm=gmm,
                n_samples=args.samples,
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
            bounds=None,
            resolution=150,
            figsize=figsize
        )
        
        # Save the plot
        output_path = os.path.join(args.output_dir, f"{config_name}_contours_with_samples.png")
        fig.savefig(output_path, dpi=300)
        print(f"Plot with MCMC samples saved to {output_path}")
    
    # Display the plots
    plt.show()


if __name__ == "__main__":
    main() 