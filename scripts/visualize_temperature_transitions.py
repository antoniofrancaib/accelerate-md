#!/usr/bin/env python
"""
Visualization script for temperature transitions in GMM data.

This script analyzes and visualizes the GMM samples at different temperatures
and the transitions between adjacent temperature levels.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.targets.gmm import GMM

# Set up nice plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 10)
plt.rcParams["font.size"] = 12


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_samples_at_temperatures(pairs_data, temps, save_dir='plots'):
    """
    Plot GMM samples at different temperatures.
    
    Args:
        pairs_data (np.ndarray): Array of sample pairs, shape (n_transitions, 2, n_samples, 2)
        temps (np.ndarray): Array of temperatures
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_temperatures = len(temps)
    n_transitions = n_temperatures - 1
    
    # Create a figure for all temperatures
    plt.figure(figsize=(15, 10))
    
    # Plot samples at each temperature
    for i in range(n_temperatures):
        if i < n_transitions:
            # For temperatures 0 to n-2, use the source samples from the i-th transition
            samples = pairs_data[i, 0]
            color = f'C{i}'
            label = f'T={temps[i]:.4f}'
        else:
            # For the last temperature, use the target samples from the last transition
            samples = pairs_data[n_transitions-1, 1]
            color = f'C{i}'
            label = f'T={temps[i]:.4f}'
        
        plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5, color=color, label=label)
    
    plt.title(f'GMM Samples at Different Temperatures')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gmm_temperatures_all.png'), dpi=200)
    plt.close()
    
    # Create individual plots for each temperature
    for i in range(n_temperatures):
        plt.figure(figsize=(8, 8))
        
        if i < n_transitions:
            # For temperatures 0 to n-2, use the source samples from the i-th transition
            samples = pairs_data[i, 0]
            color = f'C{i}'
        else:
            # For the last temperature, use the target samples from the last transition
            samples = pairs_data[n_transitions-1, 1]
            color = f'C{i}'
        
        plt.scatter(samples[:, 0], samples[:, 1], s=20, alpha=0.6, color=color)
        
        plt.title(f'GMM Samples at T={temps[i]:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gmm_temperature_{i}.png'), dpi=200)
        plt.close()


def plot_transition_pairs(pairs_data, temps, transition_idx=0, sample_count=1000, save_dir='plots'):
    """
    Plot sample pairs for a specific temperature transition.
    
    Args:
        pairs_data (np.ndarray): Array of sample pairs, shape (n_transitions, 2, n_samples, 2)
        temps (np.ndarray): Array of temperatures
        transition_idx (int): Index of the transition to plot
        sample_count (int): Number of sample pairs to plot
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get sample pairs for the specified transition
    source_samples = pairs_data[transition_idx, 0, :sample_count]
    target_samples = pairs_data[transition_idx, 1, :sample_count]
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Plot source and target samples
    plt.scatter(source_samples[:, 0], source_samples[:, 1], s=20, alpha=0.6, color='blue', label=f'T={temps[transition_idx]:.4f}')
    plt.scatter(target_samples[:, 0], target_samples[:, 1], s=20, alpha=0.6, color='red', label=f'T={temps[transition_idx+1]:.4f}')
    
    plt.title(f'Temperature Transition {transition_idx}: T={temps[transition_idx]:.4f} → T={temps[transition_idx+1]:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'transition_{transition_idx}_scatter.png'), dpi=200)
    plt.close()
    
    # Create a paired plot with lines connecting corresponding samples
    plt.figure(figsize=(12, 10))
    
    # Plot a subset of pairs with lines connecting them
    n_lines = min(50, sample_count)  # Limit the number of lines for clarity
    
    # Plot samples
    plt.scatter(source_samples[:, 0], source_samples[:, 1], s=20, alpha=0.6, color='blue', label=f'T={temps[transition_idx]:.4f}')
    plt.scatter(target_samples[:, 0], target_samples[:, 1], s=20, alpha=0.6, color='red', label=f'T={temps[transition_idx+1]:.4f}')
    
    # Draw lines connecting corresponding samples
    for i in range(n_lines):
        plt.plot([source_samples[i, 0], target_samples[i, 0]], 
                 [source_samples[i, 1], target_samples[i, 1]], 
                 color='gray', alpha=0.3, linestyle='-', linewidth=1)
    
    plt.title(f'Paired Samples for Transition {transition_idx}: T={temps[transition_idx]:.4f} → T={temps[transition_idx+1]:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'transition_{transition_idx}_paired.png'), dpi=200)
    plt.close()
    
    # Create a heatmap of the displacement vectors
    displacements = np.array(target_samples - source_samples, dtype=np.float64)
    
    plt.figure(figsize=(10, 8))
    
    # Create a 2D histogram of the displacement vectors
    plt.hist2d(displacements[:, 0], displacements[:, 1], bins=50, cmap='viridis')
    plt.colorbar(label='Count')
    
    plt.title(f'Displacement Vectors for Transition {transition_idx}: T={temps[transition_idx]:.4f} → T={temps[transition_idx+1]:.4f}')
    plt.xlabel('Δx')
    plt.ylabel('Δy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'transition_{transition_idx}_displacement_heatmap.png'), dpi=200)
    plt.close()
    
    # Plot the magnitude of displacements
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(displacement_magnitudes, bins=50, alpha=0.7)
    plt.axvline(x=np.mean(displacement_magnitudes), color='red', linestyle='--', 
                label=f'Mean: {np.mean(displacement_magnitudes):.4f}')
    plt.axvline(x=np.median(displacement_magnitudes), color='green', linestyle='--', 
                label=f'Median: {np.median(displacement_magnitudes):.4f}')
    
    plt.title(f'Displacement Magnitudes for Transition {transition_idx}: T={temps[transition_idx]:.4f} → T={temps[transition_idx+1]:.4f}')
    plt.xlabel('Displacement Magnitude')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'transition_{transition_idx}_displacement_magnitudes.png'), dpi=200)
    plt.close()


def analyze_temperature_difficulty(pairs_data, temps, save_dir='plots'):
    """
    Analyze the difficulty of each temperature transition.
    
    Args:
        pairs_data (np.ndarray): Array of sample pairs, shape (n_transitions, 2, n_samples, 2)
        temps (np.ndarray): Array of temperatures
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_transitions = pairs_data.shape[0]
    
    # Calculate metrics for each transition
    mean_displacements = []
    median_displacements = []
    max_displacements = []
    std_displacements = []
    
    for i in range(n_transitions):
        source_samples = pairs_data[i, 0]
        target_samples = pairs_data[i, 1]
        
        displacements = np.array(target_samples - source_samples, dtype=np.float64)
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        
        mean_displacements.append(np.mean(displacement_magnitudes))
        median_displacements.append(np.median(displacement_magnitudes))
        max_displacements.append(np.max(displacement_magnitudes))
        std_displacements.append(np.std(displacement_magnitudes))
    
    # Create temperature difference array
    temp_diffs = [temps[i+1] - temps[i] for i in range(n_transitions)]
    
    # Plot metrics versus temperature difference
    metrics = [
        (mean_displacements, 'Mean Displacement'),
        (median_displacements, 'Median Displacement'),
        (max_displacements, 'Max Displacement'),
        (std_displacements, 'Std Deviation of Displacements')
    ]
    
    plt.figure(figsize=(12, 10))
    
    for values, label in metrics:
        plt.plot(range(n_transitions), values, marker='o', linestyle='-', label=label)
    
    plt.xticks(range(n_transitions), [f'{i}: {temps[i]:.2f}→{temps[i+1]:.2f}' for i in range(n_transitions)], rotation=45)
    plt.title('Displacement Metrics for Each Temperature Transition')
    plt.xlabel('Transition Index: Temperature Range')
    plt.ylabel('Displacement Metric')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'transition_difficulty_metrics.png'), dpi=200)
    plt.close()
    
    # Plot mean displacement versus temperature difference
    plt.figure(figsize=(10, 6))
    plt.scatter(temp_diffs, mean_displacements, s=80, alpha=0.7)
    
    # Add labels for each point
    for i, (x, y) in enumerate(zip(temp_diffs, mean_displacements)):
        plt.annotate(f'{i}: {temps[i]:.2f}→{temps[i+1]:.2f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    plt.title('Mean Displacement vs Temperature Difference')
    plt.xlabel('Temperature Difference')
    plt.ylabel('Mean Displacement')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_displacement_vs_temp_diff.png'), dpi=200)
    plt.close()


def plot_gmm_density(config, save_dir='plots'):
    """
    Plot the GMM density at different temperatures.
    
    Args:
        config (dict): Configuration dictionary with GMM parameters
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract GMM parameters from config
    gmm_config = config['gmm']
    dim = gmm_config['dim']
    n_mixes = gmm_config['n_mixes']
    loc_scaling = gmm_config['loc_scaling']
    
    # Create GMM instance
    gmm = GMM(dim=dim, n_mixes=n_mixes, loc_scaling=loc_scaling)
    
    # Extract temperature range from config
    pt_config = config['pt']
    temp_low = pt_config['temp_low']
    temp_high = pt_config['temp_high']
    total_n_temp = pt_config['total_n_temp']
    
    # Generate temperature array based on schedule
    temp_schedule = pt_config.get('temp_schedule', 'geom')
    
    if temp_schedule == 'geom':
        # Geometric spacing
        temps = np.geomspace(temp_low, temp_high, total_n_temp)
    else:
        # Linear spacing
        temps = np.linspace(temp_low, temp_high, total_n_temp)
    
    # Create grid for density evaluation
    x = np.linspace(-5 * loc_scaling, 5 * loc_scaling, 100)
    y = np.linspace(-5 * loc_scaling, 5 * loc_scaling, 100)
    X, Y = np.meshgrid(x, y)
    
    # Plot density at different temperatures
    for i, temp in enumerate(temps):
        plt.figure(figsize=(8, 8))
        
        # Evaluate density on grid
        Z = np.zeros_like(X)
        for i_x in range(X.shape[0]):
            for i_y in range(X.shape[1]):
                point = np.array([X[i_x, i_y], Y[i_x, i_y]])
                # Use un-tempered density and manually apply temperature
                log_prob = gmm.log_prob(torch.tensor(point, dtype=torch.float32).view(1, -1)).numpy()[0] / temp
                Z[i_x, i_y] = np.exp(log_prob)
        
        # Plot density contour
        plt.contourf(X, Y, Z, 20, cmap='viridis')
        plt.colorbar(label='Density')
        
        plt.title(f'GMM Density at T={temp:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gmm_density_T{i}.png'), dpi=200)
        plt.close()


def visualize_temperature_jumps(pairs_data, temps, save_dir='plots'):
    """
    Create visualizations focusing on the jumps between source and target temperatures.
    
    Args:
        pairs_data (np.ndarray): Array of sample pairs, shape (n_transitions, 2, n_samples, 2)
        temps (np.ndarray): Array of temperatures
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    n_transitions = pairs_data.shape[0]
    
    # Create individual plots for each transition
    for transition_idx in range(n_transitions):
        # Get sample pairs for this transition
        source_samples = pairs_data[transition_idx, 0, :1000]  # Use 1000 samples for clarity
        target_samples = pairs_data[transition_idx, 1, :1000]
        
        # Calculate displacements
        displacements = np.array(target_samples - source_samples, dtype=np.float64)
        
        # 1. Simple scatter plot of source and target samples
        plt.figure(figsize=(12, 10))
        plt.scatter(source_samples[:, 0], source_samples[:, 1], 
                   s=30, alpha=0.6, color='blue', label=f'T={temps[transition_idx]:.4f}')
        plt.scatter(target_samples[:, 0], target_samples[:, 1], 
                   s=30, alpha=0.6, color='red', label=f'T={temps[transition_idx+1]:.4f}')
        
        plt.title(f'Samples for Transition {transition_idx}: T={temps[transition_idx]:.4f} → T={temps[transition_idx+1]:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'transition_{transition_idx}_samples.png'), dpi=200)
        plt.close()
        
        # 2. Plot showing the differences as points in 2D
        plt.figure(figsize=(12, 10))
        plt.scatter(displacements[:, 0], displacements[:, 1], 
                   s=20, alpha=0.6, c=np.linalg.norm(displacements, axis=1), cmap='viridis')
        plt.colorbar(label='Displacement Magnitude')
        
        plt.title(f'Displacement Vectors for Transition {transition_idx}: T={temps[transition_idx]:.4f} → T={temps[transition_idx+1]:.4f}')
        plt.xlabel('Δx (Change in x-coordinate)')
        plt.ylabel('Δy (Change in y-coordinate)')
        plt.grid(True, alpha=0.3)
        
        # Add a point at the origin for reference
        plt.scatter([0], [0], color='red', s=100, marker='x')
        
        # Add mean displacement vector
        mean_dx = np.mean(displacements[:, 0])
        mean_dy = np.mean(displacements[:, 1])
        plt.arrow(0, 0, mean_dx, mean_dy, head_width=0.1, head_length=0.1, 
                 fc='red', ec='red', width=0.02)
        plt.annotate(f'Mean: ({mean_dx:.3f}, {mean_dy:.3f})', 
                    (mean_dx, mean_dy), 
                    textcoords="offset points", 
                    xytext=(10, 10), 
                    ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'transition_{transition_idx}_displacement_points.png'), dpi=200)
        plt.close()
        
        # 3. Density plot of the displacements
        plt.figure(figsize=(12, 10))
        
        # Use a 2D histogram to show the density
        h = plt.hist2d(displacements[:, 0], displacements[:, 1], 
                      bins=50, cmap='viridis', density=True)
        plt.colorbar(h[3], label='Density')
        
        plt.title(f'Displacement Density for Transition {transition_idx}: T={temps[transition_idx]:.4f} → T={temps[transition_idx+1]:.4f}')
        plt.xlabel('Δx (Change in x-coordinate)')
        plt.ylabel('Δy (Change in y-coordinate)')
        plt.grid(True, alpha=0.3)
        
        # Add a point at the origin for reference
        plt.scatter([0], [0], color='red', s=100, marker='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'transition_{transition_idx}_displacement_density.png'), dpi=200)
        plt.close()
    
    # Create a summary plot of all displacement distributions
    plt.figure(figsize=(15, 10))
    
    for transition_idx in range(n_transitions):
        # Get sample pairs for this transition
        source_samples = pairs_data[transition_idx, 0, :1000]  # Use 1000 samples for clarity
        target_samples = pairs_data[transition_idx, 1, :1000]
        
        # Calculate displacements
        displacements = np.array(target_samples - source_samples, dtype=np.float64)
        
        # Plot displacement magnitudes as a histogram
        plt.hist(np.linalg.norm(displacements, axis=1), bins=30, alpha=0.5, 
                label=f'T{transition_idx}: {temps[transition_idx]:.2f} → {temps[transition_idx+1]:.2f}')
    
    plt.title('Displacement Magnitude Distributions for All Transitions')
    plt.xlabel('Displacement Magnitude')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_displacement_magnitudes.png'), dpi=200)
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize temperature transitions in GMM data')
    parser.add_argument('--config', type=str, default='configs/pt/gmm.yaml',
                        help='Path to the GMM configuration file')
    parser.add_argument('--pair-file', type=str, default=None,
                        help='Path to the .npz file containing sample pairs (overrides config)')
    parser.add_argument('--save-dir', type=str, default='plots/transitions',
                        help='Directory to save plots')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--jumps-only', action='store_true',
                        help='Only visualize temperature jumps')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine pair file path
    if args.pair_file:
        pair_file = args.pair_file
    else:
        pair_file = os.path.join(config['io']['save_fold'], 'gmm_pairs.npz')
    
    # Load data
    print(f"Loading data from {pair_file}")
    data = np.load(pair_file, allow_pickle=True)
    
    pairs = data['pairs']
    temps = data['temps']
    
    print(f"Data loaded. Shape of pairs: {pairs.shape}")
    print(f"Temperatures: {temps}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.jumps_only:
        # Only visualize temperature jumps
        print("Visualizing temperature jumps...")
        visualize_temperature_jumps(pairs, temps, save_dir=args.save_dir)
    else:
        # Plot samples at different temperatures
        print("Plotting samples at different temperatures...")
        plot_samples_at_temperatures(pairs, temps, save_dir=args.save_dir)
        
        # Plot transition pairs for each temperature transition
        n_transitions = pairs.shape[0]
        print(f"Plotting {n_transitions} temperature transitions...")
        for i in range(n_transitions):
            print(f"Plotting transition {i}: T={temps[i]:.4f} → T={temps[i+1]:.4f}")
            plot_transition_pairs(pairs, temps, transition_idx=i, save_dir=args.save_dir)
        
        # Analyze temperature transition difficulty
        print("Analyzing temperature transition difficulty...")
        analyze_temperature_difficulty(pairs, temps, save_dir=args.save_dir)
        
        # Add temperature jump visualizations
        print("Visualizing temperature jumps...")
        visualize_temperature_jumps(pairs, temps, save_dir=args.save_dir)
        
        # Plot GMM density at different temperatures
        print("Plotting GMM density at different temperatures...")
        try:
            import torch
            plot_gmm_density(config, save_dir=args.save_dir)
        except:
            print("Skipping GMM density plots due to error (torch might be missing)")
    
    print(f"All plots saved to {args.save_dir}")


if __name__ == '__main__':
    main() 