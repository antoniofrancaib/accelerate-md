#!/usr/bin/env python
"""
Training script for RealNVP normalizing flow on GMM data.

This script trains a single RealNVP model on GMM data via maximum likelihood estimation.
It's intended as a simpler approach compared to the temperature transition flows.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.models.flow import RealNVPFlow, create_realnvp_flow
from main.targets.gmm import GMM, plot_contours

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemperedGMMPairDataset(Dataset):
    """
    Dataset of GMM sample pairs from high temperature to low temperature.
    
    Args:
        gmm (GMM): Base GMM distribution (T=1)
        n_samples (int): Number of samples to generate
        temp_high (float): High temperature
        temp_low (float): Low temperature (typically 1.0)
    """
    def __init__(self, gmm, n_samples, temp_high=10.0, temp_low=1.0):
        """Initialize the dataset with high-temperature and low-temperature samples."""
        self.temp_high = temp_high
        self.temp_low = temp_low
        
        # For a proper implementation, we should modify the GMM's variance
        # High temp -> wider variance
        # Low temp -> narrower variance (typically the original GMM)
        with torch.no_grad():
            # Sample from high temp (we'll use scaled variance)
            # This is an approximation - scaling sqrt(temp) is more accurate for GMMs
            # but we need proper tempered sampling which is outside scope
            means = gmm.locs
            hi_scale_trils = torch.sqrt(torch.tensor(temp_high)) * gmm.scale_trils
            
            # Create a temporary high-temp GMM
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
            
            # Sample from both
            self.hi_temp_samples = hi_gmm.sample((n_samples,)).float()
            self.low_temp_samples = gmm.sample((n_samples,)).float()
        
        logger.info(f"Created dataset with {n_samples} sample pairs from T={temp_high:.4f} to T={temp_low:.4f}")
    
    def __len__(self):
        """Return the number of sample pairs in the dataset."""
        return len(self.hi_temp_samples)
    
    def __getitem__(self, idx):
        """Return a (high_temp, low_temp) sample pair at the given index."""
        return self.hi_temp_samples[idx], self.low_temp_samples[idx]


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_realnvp(config, visualize=False):
    """
    Train a RealNVP flow to map from high temperature to low temperature.
    
    Args:
        config (dict): Configuration dictionary
        visualize (bool): Whether to generate visualizations during training
    """
    # Device configuration
    device = torch.device(config.get('device', 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Setup directories
    model_dir = Path(config.get('model_dir', 'models/realnvp_5modes'))
    plot_dir = Path(config.get('plot_dir', 'plots/realnvp_5modes'))
    model_dir.mkdir(exist_ok=True, parents=True)
    if visualize:
        plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Force 5-mode GMM regardless of config
    gmm_config = config['gmm'].copy() 
    gmm_config['n_mixes'] = 5  # Set to 5 modes
    # Set a smaller loc_scaling to place modes closer together
    gmm_config['loc_scaling'] = 0.5  # Smaller scaling to tighten the clusters
    
    # Create GMM from config (only used at T=1)
    gmm = GMM(
        dim=gmm_config['dim'],
        n_mixes=gmm_config['n_mixes'],
        loc_scaling=gmm_config['loc_scaling'],
        device=device
    )
    
    # Override the GMM locations with 5 modes in an ASYMMETRIC pattern
    # This makes the 5 modes visually distinct
    with torch.no_grad():
        # Create asymmetric mode locations deliberately
        # No longer in a perfect circle - varying distances and angles
        locs = torch.zeros((5, 2), device=device)
        
        # Mode 1 - upper right
        locs[0, 0] = 2.5
        locs[0, 1] = 1.7
        
        # Mode 2 - upper left
        locs[1, 0] = -1.8
        locs[1, 1] = 2.0
        
        # Mode 3 - lower left
        locs[2, 0] = -2.0
        locs[2, 1] = -1.5
        
        # Mode 4 - lower right
        locs[3, 0] = 1.4
        locs[3, 1] = -2.2
        
        # Mode 5 - center (slightly offset)
        locs[4, 0] = 0.5
        locs[4, 1] = -0.3
        
        # Update the GMM's locations
        gmm.locs.copy_(locs)
        
        # Make the covariance matrices different for each mode to create asymmetry
        scale_trils = torch.zeros(5, 2, 2, device=device)
        
        # Mode 1 - narrow and elongated horizontally
        scale_trils[0, 0, 0] = 0.3
        scale_trils[0, 1, 1] = 0.2
        
        # Mode 2 - wider and elongated vertically
        scale_trils[1, 0, 0] = 0.3
        scale_trils[1, 1, 1] = 0.5
        
        # Mode 3 - small and circular
        scale_trils[2, 0, 0] = 0.2
        scale_trils[2, 1, 1] = 0.2
        
        # Mode 4 - large and circular
        scale_trils[3, 0, 0] = 0.4
        scale_trils[3, 1, 1] = 0.4
        
        # Mode 5 - medium and elongated diagonally
        scale_trils[4, 0, 0] = 0.3
        scale_trils[4, 1, 0] = 0.1  # off-diagonal term
        scale_trils[4, 1, 1] = 0.3
        
        # Update the GMM's scale matrices - this gives us differently shaped modes
        gmm.scale_trils.copy_(scale_trils)
        
        # Set up 5 asymmetric modes as before
        # But make all weights more balanced to avoid missing modes
        # Update weights to be more balanced - make sure no mode has too small a weight
        gmm.cat_probs.copy_(torch.tensor([0.22, 0.18, 0.20, 0.22, 0.18], device=device))
        
    logger.info(f"Set up GMM with 5 asymmetric modes with varying shapes and balanced weights")
    
    # Create RealNVP model
    model_config = config.get('model', {})
    # Set a higher default for n_couplings to increase model capacity
    if 'n_couplings' not in model_config:
        model_config['n_couplings'] = 10  # Increased from default 6 to 10 for better expressivity
    flow = create_realnvp_flow(model_config).to(device)
    logger.info(f"Created RealNVP with {sum(p.numel() for p in flow.parameters())} parameters")
    
    # Training hyperparameters
    train_config = config.get('training', {})
    n_epochs = train_config.get('n_epochs', 200)  # Increased default from lower value
    batch_size = train_config.get('batch_size', 256)
    learning_rate = train_config.get('learning_rate', 3e-4)  # Reduced from 1e-3 for more stable training
    n_samples = train_config.get('n_samples', 50000)
    val_split = train_config.get('val_split', 0.1)
    patience = train_config.get('patience', 30)  # Increased patience for better convergence
    
    # Temperature settings
    temp_high = train_config.get('temp_high', 10.0)
    temp_low = train_config.get('temp_low', 1.0)
    
    seed = train_config.get('seed', 42)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize wandb if available
    if WANDB:
        # Try to load wandb config from file
        wandb_config_path = "configs/logger/wandb.yaml"
        wandb_config = {}
        if os.path.exists(wandb_config_path):
            try:
                with open(wandb_config_path, 'r') as f:
                    wandb_config = yaml.safe_load(f)
                logger.info(f"Loaded wandb config from {wandb_config_path}")
            except Exception as e:
                logger.warning(f"Error loading wandb config: {e}")
        
        # Initialize wandb with loaded config
        wandb_init_args = {
            "project": wandb_config.get("project", "temp-realnvp"),
            "entity": wandb_config.get("entity", None),
            "config": {
                **config,
                "temp_high": temp_high,
                "temp_low": temp_low,
                "seed": seed,
                "n_samples": n_samples,
                "model_type": "RealNVP",
                "mapping": "high_to_low",
                "n_modes": 5,  # Add info about 5 modes
            },
            "name": f"realnvp_5modes_T{temp_high}_to_T{temp_low}",
            "group": wandb_config.get("group", None),
            "tags": wandb_config.get("tags", []) + ["realnvp", f"T{temp_high}-to-T{temp_low}", "5modes"],
            "mode": "offline" if wandb_config.get("offline", True) else "online"
        }
        
        # Remove None values
        wandb_init_args = {k: v for k, v in wandb_init_args.items() if v is not None}
        
        # Initialize wandb
        wandb.init(**wandb_init_args)
        logger.info(f"Initialized wandb with project={wandb_init_args.get('project')}, mode={wandb_init_args.get('mode')}")
    
    # Create dataset and dataloader with tempered pairs
    dataset = TemperedGMMPairDataset(gmm, n_samples, temp_high=temp_high, temp_low=temp_low)
    
    # Split into training and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    logger.info(f"Created dataloaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(flow.parameters(), lr=learning_rate)
    
    # Use cosine annealing scheduler instead of ReduceLROnPlateau for smoother decay
    total_steps = n_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate / 20  # Minimum LR will be 1/20th of initial LR
    )
    
    # Initial validation loss
    flow.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_high, x_low in val_loader:
            x_high, x_low = x_high.to(device), x_low.to(device)
            
            # High → Low mapping loss: how well does flow.inverse(x_high) match x_low?
            y_low, logdet = flow.inverse(x_high)
            loss = -(gmm.log_prob(y_low) / temp_low + logdet).mean()
            
            val_loss += loss.item() * len(x_high)
    val_loss /= len(val_dataset)
    
    logger.info(f"Initial validation loss: {val_loss:.4f}")
    
    # Training loop
    best_val_loss = val_loss
    best_epoch = -1
    early_stop_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        flow.train()
        train_loss = 0.0
        for x_high, x_low in train_loader:
            x_high, x_low = x_high.to(device), x_low.to(device)
            
            # High → Low mapping loss
            y_low, logdet = flow.inverse(x_high)
            loss = -(gmm.log_prob(y_low) / temp_low + logdet).mean()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step the scheduler in each iteration instead of each epoch
            
            train_loss += loss.item() * len(x_high)
        
        train_loss /= len(train_dataset)
        
        # Validation
        flow.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_high, x_low in val_loader:
                x_high, x_low = x_high.to(device), x_low.to(device)
                
                # High → Low mapping loss
                y_low, logdet = flow.inverse(x_high)
                loss = -(gmm.log_prob(y_low) / temp_low + logdet).mean()
                
                val_loss += loss.item() * len(x_high)
        
        val_loss /= len(val_dataset)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{n_epochs}: "
                   f"Train loss={train_loss:.4f}, "
                   f"Val loss={val_loss:.4f}, "
                   f"LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Log metrics to wandb
        if WANDB:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            
            # Save model
            model_path = model_dir / "realnvp_5modes_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'temp_high': temp_high,
                'temp_low': temp_low,
                'n_modes': 5
            }, model_path)
            
            logger.info(f"Saved best model at epoch {epoch+1} with val_loss={val_loss:.4f}")
            
            # Visualization (every 10 epochs or when we have a new best model)
            if visualize and (epoch % 10 == 0 or early_stop_counter == 0):
                viz_path = visualize_temp_mapping(flow, gmm, epoch, temp_high, temp_low, plot_dir, device)
                
                # Log visualization to wandb
                if WANDB and viz_path:
                    wandb.log({f"temp_mapping_epoch_{epoch}": wandb.Image(str(viz_path))})
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    best_model_path = model_dir / "realnvp_5modes_best.pt"
    checkpoint = torch.load(best_model_path)
    flow.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Training completed. Best loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Final visualization
    if visualize:
        viz_path = visualize_temp_mapping(flow, gmm, "final", temp_high, temp_low, plot_dir, device)
        verify_path = verify_bidirectional_mapping(flow, gmm, temp_high, temp_low, plot_dir, device)
        
        # Log final visualizations to wandb
        if WANDB:
            if viz_path:
                wandb.log({"final_mapping": wandb.Image(str(viz_path))})
            if verify_path:
                wandb.log({"bidirectional_verification": wandb.Image(str(verify_path))})
    
    # Finish wandb run
    if WANDB:
        # Log final best metrics
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.finish()
    
    return flow


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
    if WANDB:
        wandb.log({
            "high_to_low_mean_error": low_mean_error,
            "low_to_high_mean_error": high_mean_error,
            "true_low_variance": true_low_var,
            "mapped_low_variance": mapped_low_var,
            "true_high_variance": true_high_var,
            "mapped_high_variance": mapped_high_var,
        })
    
    return save_path


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train RealNVP to map between temperature levels")
    parser.add_argument("--config", type=str, default="configs/pt/gmm.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-dir", type=str, default="models/realnvp_5modes",
                        help="Directory to save model checkpoints")
    parser.add_argument("--plot-dir", type=str, default="plots/realnvp_5modes",
                        help="Directory to save plots")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--n-samples", type=int, default=50000,
                        help="Number of samples to generate from GMM")
    parser.add_argument("--temp-high", type=float, default=None,
                        help="High temperature for the mapping")
    parser.add_argument("--temp-low", type=float, default=None,
                        help="Low temperature for the mapping (usually 1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations during training")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if CUDA is available")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging even if available")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Disable wandb if requested
    if args.no_wandb:
        global WANDB
        WANDB = False
    
    # Load configuration
    config = load_config(args.config)
    
    # Force CPU if requested
    if args.cpu:
        config['device'] = 'cpu'
    
    # Override with command-line arguments
    config['model_dir'] = args.model_dir
    config['plot_dir'] = args.plot_dir
    
    if 'training' not in config:
        config['training'] = {}
    
    if args.n_epochs:
        config['training']['n_epochs'] = args.n_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.n_samples:
        config['training']['n_samples'] = args.n_samples
    if args.temp_high:
        config['training']['temp_high'] = args.temp_high
    if args.temp_low:
        config['training']['temp_low'] = args.temp_low
    if args.seed:
        config['training']['seed'] = args.seed
    
    # Set wandb mode based on argument (only if wandb is available)
    config['wandb'] = args.wandb and WANDB
    
    # Train the model
    logger.info("Starting temperature-mapping RealNVP training for 5-mode GMM...")
    flow = train_realnvp(config, visualize=args.visualize)
    logger.info("RealNVP training for 5-mode GMM completed!")


if __name__ == "__main__":
    main() 