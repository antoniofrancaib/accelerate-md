"""
Visualization utilities for temperature transition models.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def visualize_temp_mapping(flow, gmm, epoch, temp_high, temp_low, plot_dir, device):
    """
    Generate visualizations of high→low temperature mapping.
    
    Args:
        flow: The trained flow model
        gmm: The base GMM distribution (T=1)
        epoch: Current epoch or label for the visualization
        temp_high: High temperature
        temp_low: Low temperature
        plot_dir: Directory to save plots
        device: Device to use for computation
        
    Returns:
        Path to the saved visualization file
    """
    # Generate samples from both temperatures - use more samples for better visualization
    n_samples = 5000  # Increased from 1000 for denser visualization
    with torch.no_grad():
        # Create high temp GMM
        means = gmm.locs
        hi_scale_trils = torch.sqrt(torch.tensor(temp_high)) * gmm.scale_trils
        
        mix = torch.distributions.Categorical(gmm.cat_probs)
        comp_hi = torch.distributions.MultivariateNormal(
            loc=means,
            scale_tril=hi_scale_trils,
            validate_args=False
        )
        hi_gmm = torch.distributions.MixtureSameFamily(
            mixture_distribution=mix,
            component_distribution=comp_hi,
            validate_args=False
        )
        
        # Sample from both GMMs
        hi_temp_samples = hi_gmm.sample((n_samples,)).cpu().numpy()
        low_temp_samples = gmm.sample((n_samples,)).cpu().numpy()
        
        # Use the flow to map high → low
        hi_tensor = torch.tensor(hi_temp_samples, dtype=torch.float32, device=device)
        mapped_low, _ = flow.inverse(hi_tensor)
        mapped_low = mapped_low.cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define a grid for contour plotting
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # 1. Plot high temperature samples with contours
    axes[0].scatter(hi_temp_samples[:, 0], hi_temp_samples[:, 1], s=5, alpha=0.5)
    axes[0].set_title(f"High Temperature (T={temp_high:.1f})")
    axes[0].set_xlim(-5, 5)  # Smaller range to better see the 5 modes
    axes[0].set_ylim(-5, 5)  # Smaller range to better see the 5 modes
    axes[0].grid(alpha=0.3)
    
    # Add density contours for high temp
    try:
        from scipy.stats import gaussian_kde
        kernel_hi = gaussian_kde(hi_temp_samples.T)
        Z_hi = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_hi[i, j] = kernel_hi([X[i, j], Y[i, j]])
        axes[0].contour(X, Y, Z_hi, levels=10, colors='red', alpha=0.3)
    except Exception as e:
        logger.warning(f"Could not plot contours: {e}")
    
    # 2. Plot flow-mapped samples (high → low) with contours
    axes[1].scatter(mapped_low[:, 0], mapped_low[:, 1], s=5, alpha=0.5)
    axes[1].set_title(f"Flow-Mapped High→Low (Epoch {epoch})")
    axes[1].set_xlim(-5, 5)  # Smaller range to better see the 5 modes
    axes[1].set_ylim(-5, 5)  # Smaller range to better see the 5 modes
    axes[1].grid(alpha=0.3)
    
    # Add density contours for mapped low
    try:
        kernel_mapped = gaussian_kde(mapped_low.T)
        Z_mapped = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_mapped[i, j] = kernel_mapped([X[i, j], Y[i, j]])
        axes[1].contour(X, Y, Z_mapped, levels=10, colors='red', alpha=0.3)
    except Exception as e:
        logger.warning(f"Could not plot contours: {e}")
    
    # 3. Plot true low temperature samples with contours
    axes[2].scatter(low_temp_samples[:, 0], low_temp_samples[:, 1], s=5, alpha=0.5)
    axes[2].set_title(f"True Low Temperature (T={temp_low:.1f})")
    axes[2].set_xlim(-5, 5)  # Smaller range to better see the 5 modes
    axes[2].set_ylim(-5, 5)  # Smaller range to better see the 5 modes
    axes[2].grid(alpha=0.3)
    
    # Add density contours for low temp
    try:
        kernel_low = gaussian_kde(low_temp_samples.T)
        Z_low = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_low[i, j] = kernel_low([X[i, j], Y[i, j]])
        axes[2].contour(X, Y, Z_low, levels=10, colors='red', alpha=0.3)
    except Exception as e:
        logger.warning(f"Could not plot contours: {e}")
    
    plt.tight_layout()
    save_path = plot_dir / f"temp_mapping_5modes_epoch_{epoch}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    return save_path


def verify_bidirectional_mapping(flow, gmm, temp_high, temp_low, plot_dir, device):
    """
    Verify both directions of the mapping: high→low and low→high.
    
    Args:
        flow: The trained flow model
        gmm: The base GMM distribution (T=1)
        temp_high: High temperature
        temp_low: Low temperature 
        plot_dir: Directory to save plots
        device: Device to use for computation
        
    Returns:
        Path to the saved verification file
    """
    n_samples = 5000  # Increased from 1000 for denser visualization
    with torch.no_grad():
        # Create high temp GMM
        means = gmm.locs
        hi_scale_trils = torch.sqrt(torch.tensor(temp_high)) * gmm.scale_trils
        
        mix = torch.distributions.Categorical(gmm.cat_probs)
        comp_hi = torch.distributions.MultivariateNormal(
            loc=means,
            scale_tril=hi_scale_trils,
            validate_args=False
        )
        hi_gmm = torch.distributions.MixtureSameFamily(
            mixture_distribution=mix,
            component_distribution=comp_hi,
            validate_args=False
        )
        
        # Sample from both GMMs
        hi_temp_samples = hi_gmm.sample((n_samples,)).to(device)
        low_temp_samples = gmm.sample((n_samples,)).to(device)
        
        # Direction 1: high → low
        mapped_low, _ = flow.inverse(hi_temp_samples)
        
        # Direction 2: low → high
        mapped_high, _ = flow.forward(low_temp_samples)
        
        # Convert all to numpy for plotting
        hi_samples_np = hi_temp_samples.cpu().numpy()
        low_samples_np = low_temp_samples.cpu().numpy()
        mapped_low_np = mapped_low.cpu().numpy()
        mapped_high_np = mapped_high.cpu().numpy()
    
    # Create figure with subplots in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Row 1: High → Low mapping
    axes[0, 0].scatter(hi_samples_np[:, 0], hi_samples_np[:, 1], s=5, alpha=0.5)
    axes[0, 0].set_title(f"True High Temperature (T={temp_high:.1f})")
    axes[0, 0].set_xlim(-5, 5)  # Smaller range to better see the 5 modes
    axes[0, 0].set_ylim(-5, 5)  # Smaller range to better see the 5 modes
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].scatter(mapped_low_np[:, 0], mapped_low_np[:, 1], s=5, alpha=0.5)
    axes[0, 1].set_title(f"Mapped High→Low via flow.inverse()")
    axes[0, 1].set_xlim(-5, 5)  # Smaller range to better see the 5 modes
    axes[0, 1].set_ylim(-5, 5)  # Smaller range to better see the 5 modes
    axes[0, 1].grid(alpha=0.3)
    
    # Row 2: Low → High mapping
    axes[1, 0].scatter(low_samples_np[:, 0], low_samples_np[:, 1], s=5, alpha=0.5)
    axes[1, 0].set_title(f"True Low Temperature (T={temp_low:.1f})")
    axes[1, 0].set_xlim(-5, 5)  # Smaller range to better see the 5 modes
    axes[1, 0].set_ylim(-5, 5)  # Smaller range to better see the 5 modes
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].scatter(mapped_high_np[:, 0], mapped_high_np[:, 1], s=5, alpha=0.5)
    axes[1, 1].set_title(f"Mapped Low→High via flow.forward()")
    axes[1, 1].set_xlim(-5, 5)  # Smaller range to better see the 5 modes
    axes[1, 1].set_ylim(-5, 5)  # Smaller range to better see the 5 modes
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = plot_dir / "bidirectional_verification_5modes.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    # Calculate statistics for both directions
    # High → Low: How well does mapped_low match the true low_temp distribution?
    # Low → High: How well does mapped_high match the true hi_temp distribution?
    logger.info("Verification statistics:")
    
    # Means
    true_low_mean = low_temp_samples.mean(dim=0)
    mapped_low_mean = mapped_low.mean(dim=0)
    low_mean_error = torch.norm(true_low_mean - mapped_low_mean).item()
    
    true_high_mean = hi_temp_samples.mean(dim=0)
    mapped_high_mean = mapped_high.mean(dim=0)
    high_mean_error = torch.norm(true_high_mean - mapped_high_mean).item()
    
    logger.info(f"High→Low mean error: {low_mean_error:.4f}")
    logger.info(f"Low→High mean error: {high_mean_error:.4f}")
    
    # Variances
    true_low_var = low_temp_samples.var(dim=0).mean().item()
    mapped_low_var = mapped_low.var(dim=0).mean().item()
    
    true_high_var = hi_temp_samples.var(dim=0).mean().item()
    mapped_high_var = mapped_high.var(dim=0).mean().item()
    
    logger.info(f"True low temp variance: {true_low_var:.4f}, Mapped variance: {mapped_low_var:.4f}")
    logger.info(f"True high temp variance: {true_high_var:.4f}, Mapped variance: {mapped_high_var:.4f}")
    
    # Log statistics to wandb
    if WANDB_AVAILABLE:
        wandb.log({
            "high_to_low_mean_error": low_mean_error,
            "low_to_high_mean_error": high_mean_error,
            "true_low_variance": true_low_var,
            "mapped_low_variance": mapped_low_var,
            "true_high_variance": true_high_var,
            "mapped_high_variance": mapped_high_var,
        })
    
    return save_path 