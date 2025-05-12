"""
Trainer for RealNVP normalizing flow models for bidirectional temperature transitions.
"""

import os
import logging
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.accelmd.utils.viz import visualize_temp_mapping, verify_bidirectional_mapping

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

logger = logging.getLogger(__name__)


def train_realnvp(config, visualize=False, config_paths=None):
    """
    Train a RealNVP flow to map bidirectionally between high temperature and low temperature.
    
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
    model_dir = Path(config.get('model_dir', 'checkpoints/realnvp'))
    plot_dir = Path(config.get('plot_dir', 'plots/realnvp'))
    # Check training.model_dir for backward compatibility
    if 'training' in config and 'model_dir' in config['training']:
        model_dir = Path(config['training']['model_dir'])
    model_dir.mkdir(exist_ok=True, parents=True)
    if visualize:
        plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Create GMM from config (only used at T=1)
    # Import locally to avoid circular imports
    from src.accelmd.targets.gmm import GMM
    
    gmm_config = config.get('gmm', {})
    
    # Setup GMM with configured dimensions and mixtures
    gmm = GMM(
        dim=gmm_config.get('dim', 2),
        n_mixes=gmm_config.get('n_mixes', 5),
        loc_scaling=gmm_config.get('loc_scaling', 0.5),
        device=device
    )

    # Always apply custom configuration from the config file
    with torch.no_grad():
        # Apply custom locations if provided
        if 'locations' in gmm_config:
            custom_locs = torch.tensor(gmm_config['locations'], device=device)
            gmm.locs.copy_(custom_locs)
            logger.info(f"Applied {len(gmm_config['locations'])} custom locations to GMM")
            
        # Apply custom scales if provided
        if 'scales' in gmm_config:
            custom_scales = torch.tensor(gmm_config['scales'], device=device)
            gmm.scale_trils.copy_(custom_scales)
            logger.info(f"Applied {len(gmm_config['scales'])} custom scales to GMM")
            
        # Apply custom weights if provided
        if 'weights' in gmm_config:
            custom_weights = torch.tensor(gmm_config['weights'], device=device)
            gmm.cat_probs.copy_(custom_weights)
            logger.info(f"Applied custom weights to GMM: {custom_weights}")
    
    # Get temperature settings from config
    # First check temperature section, then training section (for backwards compatibility)
    temp_config = config.get('temperature', {})
    if not temp_config:
        # For backward compatibility
        if 'pt' in config:
            temp_config = config['pt']
        elif 'training' in config:
            temp_config = config['training']
    
    temp_high = temp_config.get('temp_high', 10.0)
    temp_low = temp_config.get('temp_low', 1.0)
    
    # Get temperature scaling method (default to "sqrt")
    temp_scaling_method = temp_config.get('scaling_method', 'sqrt')
    
    # Create a high-T version of the GMM for the forward (low→high) pass
    hi_gmm = gmm.tempered_version(temp_high, temp_scaling_method)

    logger.info(f"Using temperature scaling method: {temp_scaling_method}")
    logger.info(f"Set up GMM with {gmm.locs.shape[0]} modes for mapping from T={temp_high} to T={temp_low}")
    
    # Create RealNVP model
    from src.accelmd.models.realnvp import create_realnvp_flow
    
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
    
    seed = train_config.get('seed', 42)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize wandb if available and enabled
    if WANDB_AVAILABLE and config.get('wandb', False):
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
                
        # Check if the file exists but couldn't be loaded or if the file doesn't exist
        if not wandb_config:
            logger.warning(f"Could not load wandb config from {wandb_config_path}. Using defaults.")
        
        # Initialize wandb with loaded config
        n_modes = gmm.locs.shape[0]  # Get actual number of modes
        run_name = f"realnvp_{n_modes}modes_T{temp_high}_to_T{temp_low}"
        
        # Get group information from wandb config
        group = wandb_config.get("group", "temperature-mapping")
        
        # Get tags from wandb config (with fallback to empty list)
        tags_from_config = wandb_config.get("tags", [])
        # Ensure tags_from_config is a list
        if not isinstance(tags_from_config, list):
            tags_from_config = [tags_from_config]
        
        # Combine with run-specific tags
        all_tags = tags_from_config + ["realnvp", f"T{temp_high}-to-T{temp_low}", f"{n_modes}modes"]
        
        # Use offline mode only if explicitly specified in config
        is_offline = wandb_config.get("offline", False)
        
        wandb_init_args = {
            "project": wandb_config.get("project", "accelmd"),
            "entity": wandb_config.get("entity", None),
            "config": {
                **config,  # merged configuration from YAML files
                "temp_high": temp_high,
                "temp_low": temp_low,
                "seed": seed,
                "n_samples": n_samples,
                "model_type": "RealNVP",
                "mapping": "bidirectional",
                "n_modes": n_modes,
            },
            "name": run_name,
            "group": group,
            "tags": all_tags,
            "mode": "offline" if is_offline else "online"
        }
        
        # Remove None values
        wandb_init_args = {k: v for k, v in wandb_init_args.items() if v is not None}
        
        logger.info(f"Initializing wandb with: project={wandb_init_args.get('project')}, "
                   f"mode={'offline' if is_offline else 'online'}")
        
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
    else:
        if not WANDB_AVAILABLE:
            logger.info("Weights & Biases (wandb) not available. Install with 'pip install wandb' for experiment tracking.")
        elif not config.get('wandb', False):
            logger.info("Weights & Biases (wandb) not enabled. Use --wandb flag to enable.")
    
    # Create dataset and dataloader with tempered pairs
    from src.accelmd.data.tempered_gmm import TemperedGMMPairDataset
    
    # Create the dataset with the temperature scaling method
    dataset = TemperedGMMPairDataset(
        gmm, 
        n_samples, 
        temp_high=temp_high, 
        temp_low=temp_low,
        temp_scaling_method=temp_scaling_method
    )
    
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
    val_high_to_low_loss = 0.0
    val_low_to_high_loss = 0.0
    val_pair_loss = 0.0
    with torch.no_grad():
        for x_high, x_low in val_loader:
            x_high, x_low = x_high.to(device), x_low.to(device)
            
            # ------------------------------------------------------------------
            #  Bidirectional mapping losses
            # ------------------------------------------------------------------
            # (a) High → Low  (inverse)
            y_low, ld_inv = flow.inverse(x_high)                # inverse pass
            lo_logp = gmm.log_prob(y_low)                       # density at T=1
            high_to_low_loss = -(lo_logp + ld_inv).mean()       # high->low direction loss

            # (b) Low → High  (forward)
            y_high, ld_fwd = flow.forward(x_low)                # forward pass
            hi_logp = hi_gmm.log_prob(y_high) / temp_high       # scale log‑p by 1/T
            low_to_high_loss = -(hi_logp + ld_fwd).mean()       # low->high direction loss

            # (c) Optional pairwise term (stabilises training)
            pair_loss = ((y_low - x_low) ** 2).sum(-1).mean()   # MSE in data space
            pair_loss_term = 0.05 * pair_loss                  # weighted pair loss

            # Total bidirectional loss
            loss = high_to_low_loss + low_to_high_loss + pair_loss_term
            
            # Skip batch if loss is NaN or Inf to avoid destabilising training
            if not torch.isfinite(loss):
                logger.warning(f"Skipping batch {batch_idx} in epoch {epoch} due to non-finite loss")
                continue
            
            val_loss += loss.item() * len(x_high)
            val_high_to_low_loss += high_to_low_loss.item() * len(x_high)
            val_low_to_high_loss += low_to_high_loss.item() * len(x_high)
            val_pair_loss += pair_loss.item() * len(x_high)
    
    val_loss /= len(val_dataset)
    val_high_to_low_loss /= len(val_dataset)
    val_low_to_high_loss /= len(val_dataset)
    val_pair_loss /= len(val_dataset)
    
    logger.info(f"Initial validation loss: {val_loss:.4f} (High→Low: {val_high_to_low_loss:.4f}, Low→High: {val_low_to_high_loss:.4f}, Pair: {val_pair_loss:.4f})")
    
    # Training loop
    best_val_loss = val_loss
    best_epoch = -1
    early_stop_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        flow.train()
        train_loss = 0.0
        train_high_to_low_loss = 0.0
        train_low_to_high_loss = 0.0
        train_pair_loss = 0.0
        for batch_idx, (x_high, x_low) in enumerate(train_loader):
            x_high, x_low = x_high.to(device), x_low.to(device)
            
            # ------------------------------------------------------------------
            #  Bidirectional mapping losses
            # ------------------------------------------------------------------
            # (a) High → Low  (inverse)
            y_low, ld_inv = flow.inverse(x_high)                # inverse pass
            lo_logp = gmm.log_prob(y_low)                       # density at T=1
            high_to_low_loss = -(lo_logp + ld_inv).mean()       # high->low direction loss

            # (b) Low → High  (forward)
            y_high, ld_fwd = flow.forward(x_low)                # forward pass
            hi_logp = hi_gmm.log_prob(y_high) / temp_high       # scale log‑p by 1/T
            low_to_high_loss = -(hi_logp + ld_fwd).mean()       # low->high direction loss

            # (c) Optional pairwise term (stabilises training)
            pair_loss = ((y_low - x_low) ** 2).sum(-1).mean()   # MSE in data space
            pair_loss_term = 0.05 * pair_loss                  # weighted pair loss

            # Total bidirectional loss
            loss = high_to_low_loss + low_to_high_loss + pair_loss_term
            
            # Skip batch if loss is NaN or Inf to avoid destabilising training
            if not torch.isfinite(loss):
                logger.warning(f"Skipping batch {batch_idx} in epoch {epoch} due to non-finite loss")
                continue
            
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
            train_high_to_low_loss += high_to_low_loss.item() * len(x_high)
            train_low_to_high_loss += low_to_high_loss.item() * len(x_high)
            train_pair_loss += pair_loss.item() * len(x_high)
        
        train_loss /= len(train_dataset)
        train_high_to_low_loss /= len(train_dataset)
        train_low_to_high_loss /= len(train_dataset)
        train_pair_loss /= len(train_dataset)
        
        # Validation
        flow.eval()
        val_loss = 0.0
        val_high_to_low_loss = 0.0
        val_low_to_high_loss = 0.0
        val_pair_loss = 0.0
        with torch.no_grad():
            for x_high, x_low in val_loader:
                x_high, x_low = x_high.to(device), x_low.to(device)
                
                # ------------------------------------------------------------------
                #  Bidirectional mapping losses (same as training)
                # ------------------------------------------------------------------
                # (a) High → Low  (inverse)
                y_low, ld_inv = flow.inverse(x_high)                # inverse pass
                lo_logp = gmm.log_prob(y_low)                       # density at T=1
                high_to_low_loss = -(lo_logp + ld_inv).mean()       # high->low direction loss

                # (b) Low → High  (forward)
                y_high, ld_fwd = flow.forward(x_low)                # forward pass
                hi_logp = hi_gmm.log_prob(y_high) / temp_high       # scale log‑p by 1/T
                low_to_high_loss = -(hi_logp + ld_fwd).mean()       # low->high direction loss

                # (c) Optional pairwise term
                pair_loss = ((y_low - x_low) ** 2).sum(-1).mean()   # MSE in data space
                pair_loss_term = 0.05 * pair_loss                  # weighted pair loss

                # Total bidirectional loss
                loss = high_to_low_loss + low_to_high_loss + pair_loss_term
                
                # Skip batch if loss is NaN or Inf
                if not torch.isfinite(loss):
                    continue
                
                val_loss += loss.item() * len(x_high)
                val_high_to_low_loss += high_to_low_loss.item() * len(x_high)
                val_low_to_high_loss += low_to_high_loss.item() * len(x_high)
                val_pair_loss += pair_loss.item() * len(x_high)
        
        val_loss /= len(val_dataset)
        val_high_to_low_loss /= len(val_dataset)
        val_low_to_high_loss /= len(val_dataset)
        val_pair_loss /= len(val_dataset)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{n_epochs}: "
                   f"Train loss={train_loss:.4f} (H→L: {train_high_to_low_loss:.4f}, L→H: {train_low_to_high_loss:.4f}, Pair: {train_pair_loss:.4f}), "
                   f"Val loss={val_loss:.4f} (H→L: {val_high_to_low_loss:.4f}, L→H: {val_low_to_high_loss:.4f}, Pair: {val_pair_loss:.4f}), "
                   f"LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Log metrics to wandb
        if WANDB_AVAILABLE and config.get('wandb', False):
            wandb.log({
                "train/total_loss": train_loss,
                "train/high_to_low_loss": train_high_to_low_loss,
                "train/low_to_high_loss": train_low_to_high_loss,
                "train/pair_loss": train_pair_loss,
                "val/total_loss": val_loss,
                "val/high_to_low_loss": val_high_to_low_loss,
                "val/low_to_high_loss": val_low_to_high_loss,
                "val/pair_loss": val_pair_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            
            # Save model
            n_modes = gmm.locs.shape[0]
            model_path = model_dir / f"realnvp_{n_modes}modes_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'temp_high': temp_high,
                'temp_low': temp_low,
                'n_modes': n_modes
            }, model_path)
            
            logger.info(f"Saved best model at epoch {epoch+1} with val_loss={val_loss:.4f}")
            
            # Visualization (every 10 epochs or when we have a new best model)
            if visualize and (epoch % 10 == 0 or early_stop_counter == 0):
                viz_path = visualize_temp_mapping(flow, gmm, epoch, temp_high, temp_low, plot_dir, device)
                
                # Log visualization to wandb
                if WANDB_AVAILABLE and config.get('wandb', False) and viz_path:
                    wandb.log({f"temp_mapping_epoch_{epoch}": wandb.Image(str(viz_path))})
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    n_modes = gmm.locs.shape[0]
    best_model_path = model_dir / f"realnvp_{n_modes}modes_best.pt"
    checkpoint = torch.load(best_model_path)
    flow.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Training completed. Best loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Final visualization
    if visualize:
        viz_path = visualize_temp_mapping(flow, gmm, "final", temp_high, temp_low, plot_dir, device)
        verify_path = verify_bidirectional_mapping(flow, gmm, temp_high, temp_low, plot_dir, device)
        
        # Log final visualizations to wandb
        if WANDB_AVAILABLE and config.get('wandb', False):
            if viz_path:
                wandb.log({"final_mapping": wandb.Image(str(viz_path))})
            if verify_path:
                wandb.log({"bidirectional_verification": wandb.Image(str(verify_path))})
    
    # Finish wandb run
    if WANDB_AVAILABLE and config.get('wandb', False):
        # Log final best metrics
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.finish()
    
    return flow

# Add CLI interface for direct execution
if __name__ == "__main__":
    import argparse
    import sys
    from src.accelmd.utils.config import load_config, merge_dicts
    
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
                            "configs/pt/realnvp.yaml",
                        ],
                        help="Path(s) to YAML configuration file(s). Can be provided multiple times.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory to save model checkpoints (overrides config)")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory to save plots (overrides config)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Number of epochs to train (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate for Adam optimizer (overrides config)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to generate from GMM (overrides config)")
    parser.add_argument("--temp-high", type=float, default=None,
                        help="High temperature for the mapping (overrides config)")
    parser.add_argument("--temp-low", type=float, default=None,
                        help="Low temperature for the mapping (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (overrides config)")
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
    
    # Override with command-line arguments - only if they're provided
    if args.model_dir:
        config['model_dir'] = args.model_dir
    if args.plot_dir:
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
    if args.seed:
        config['training']['seed'] = args.seed
    
    # Temperature settings may be in different places in the config
    if 'temperature' not in config:
        config['temperature'] = {}
    
    if args.temp_high:
        config['temperature']['temp_high'] = args.temp_high
        # For backward compatibility with old configs
        if 'pt' in config:
            config['pt']['temp_high'] = args.temp_high
    if args.temp_low:
        config['temperature']['temp_low'] = args.temp_low
        # For backward compatibility with old configs
        if 'pt' in config:
            config['pt']['temp_low'] = args.temp_low
    
    # Set wandb mode based on argument (only if wandb is available)
    config['wandb'] = args.wandb and wandb_available
    
    # Train the model
    logger.info(f"Starting temperature-mapping RealNVP training...")
    flow = train_realnvp(config, visualize=args.visualize, config_paths=config_paths)
    logger.info("RealNVP training completed!") 