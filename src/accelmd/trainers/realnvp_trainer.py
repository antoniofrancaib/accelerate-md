"""
Trainer for RealNVP normalizing flow models for temperature transitions.
"""

import os
import logging
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.utils.viz import visualize_temp_mapping, verify_bidirectional_mapping

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def train_realnvp(config, visualize=False, config_paths=None):
    """
    Train a RealNVP flow to map from high temperature to low temperature.
    
    Args:
        config (dict): Configuration dictionary
        visualize (bool): Whether to generate visualizations during training
        config_paths (list): List of paths to configuration files
        
    Returns:
        The trained flow model
    """
    # Device configuration
    device = torch.device(config.get('device', 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Setup directories
    model_dir = Path(config.get('model_dir', 'checkpoints/realnvp_5modes'))
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
    # Import locally to avoid circular imports
    from main.targets.gmm import GMM
    
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
    from src.models.realnvp import create_realnvp_flow
    
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
    if WANDB_AVAILABLE:
        # Try to load wandb config from file
        wandb_config_path = "configs/logger/wandb.yaml"
        wandb_config = {}
        if os.path.exists(wandb_config_path):
            try:
                with open(wandb_config_path, 'r') as f:
                    import yaml
                    wandb_config = yaml.safe_load(f)
                logger.info(f"Loaded wandb config from {wandb_config_path}")
            except Exception as e:
                logger.warning(f"Error loading wandb config: {e}")
        
        # Initialize wandb with loaded config
        wandb_init_args = {
            "project": wandb_config.get("project", "temp-realnvp"),
            "entity": wandb_config.get("entity", None),
            "config": {
                **config,  # merged configuration from YAML files
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
        # Log the configuration files themselves as an artifact for full reproducibility
        try:
            artifact = wandb.Artifact("config_files", type="config")
            for p in (config_paths or []):
                if os.path.exists(p):
                    artifact.add_file(p)
            wandb.log_artifact(artifact)
        except Exception as e:
            logger.warning(f"Could not log config artifact to wandb: {e}")
        logger.info(f"Initialized wandb with project={wandb_init_args.get('project')}, mode={wandb_init_args.get('mode')}")
    
    # Create dataset and dataloader with tempered pairs
    from src.data.tempered_gmm import TemperedGMMPairDataset
    
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
    
    # Create a learning rate scheduler with warmup followed by cosine decay
    # 1. Start with a small learning rate (1e-4)
    # 2. Linear warmup for 5 epochs to target learning rate (3e-4)
    # 3. Cosine annealing for the rest of training
    
    # Set initial and target LRs for warmup
    initial_lr = train_config.get('initial_lr', 1e-4)
    warmup_epochs = train_config.get('warmup_epochs', 5)
    eta_min_factor = train_config.get('eta_min_factor', 0.05)
    
    # Number of steps for each phase
    warmup_steps = warmup_epochs * len(train_loader)
    remaining_steps = (n_epochs - warmup_epochs) * len(train_loader)
    
    # Set initial LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr
    
    # Create warmup scheduler (LinearLR from initial_lr to learning_rate)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=initial_lr / learning_rate,  # start at initial_lr
        end_factor=1.0,  # end at learning_rate
        total_iters=warmup_steps
    )
    
    # Create cosine annealing scheduler for after warmup
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_steps,
        eta_min=learning_rate * eta_min_factor  # Minimum LR as fraction of target LR
    )
    
    # Define max gradient norm for clipping
    max_grad_norm = train_config.get('max_grad_norm', 5.0)
    
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
        for batch_idx, (x_high, x_low) in enumerate(train_loader):
            x_high, x_low = x_high.to(device), x_low.to(device)
            
            # High → Low mapping loss
            y_low, logdet = flow.inverse(x_high)
            loss = -(gmm.log_prob(y_low) / temp_low + logdet).mean()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to improve stability
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_grad_norm)
            optimizer.step()
            
            # Step the appropriate scheduler based on current training step
            global_step = epoch * len(train_loader) + batch_idx
            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            
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
        if WANDB_AVAILABLE:
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
                if WANDB_AVAILABLE and viz_path:
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
        if WANDB_AVAILABLE:
            if viz_path:
                wandb.log({"final_mapping": wandb.Image(str(viz_path))})
            if verify_path:
                wandb.log({"bidirectional_verification": wandb.Image(str(verify_path))})
    
    # Finish wandb run
    if WANDB_AVAILABLE:
        # Log final best metrics
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.finish()
    
    return flow 

# Add CLI interface for direct execution
if __name__ == "__main__":
    import argparse
    import sys
    from src.utils.config import load_config, merge_dicts
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train RealNVP flow for temperature transitions")
    parser.add_argument("--config", type=str, action="append",
                        default=[
                            "configs/pt/gmm.yaml",
                            "configs/pt/flow_training.yaml",
                        ],
                        help="Path(s) to YAML configuration file(s). Can be provided multiple times.")
    parser.add_argument("--model-dir", type=str, default="checkpoints/realnvp_5modes",
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
        # Use local variable instead of modifying the global
        wandb_available = False
    else:
        wandb_available = WANDB_AVAILABLE
    
    # Load configuration
    config = {}
    config_paths = args.config  # list of paths
    for cfg_path in config_paths:
        cfg_part = load_config(cfg_path)
        merge_dicts(config, cfg_part)

    # Keep track of which files were loaded – useful for logging
    config['config_files'] = config_paths
    
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
    config['wandb'] = args.wandb and wandb_available
    
    # Train the model
    logger.info("Starting temperature-mapping RealNVP training for 5-mode GMM...")
    flow = train_realnvp(config, visualize=args.visualize, config_paths=config_paths)
    logger.info("RealNVP training for 5-mode GMM completed!") 