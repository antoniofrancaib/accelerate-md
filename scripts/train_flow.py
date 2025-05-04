#!/usr/bin/env python
"""
Training script for temperature transition flows.

This script implements the training of normalizing flows for temperature
transitions in a parallel tempering simulation.
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
import time
import yaml
import math
from main.targets.gmm import GMM

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.models.flow import TempTransitionFlow, create_temp_transition_flow

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


class GMMPairDataset(Dataset):
    """
    Dataset of GMM sample pairs from adjacent temperatures.
    
    This dataset loads pairs of samples from the GMM parallel tempering
    simulation, where each pair consists of a sample from temperature k
    and the corresponding sample from temperature k+1.
    
    Args:
        pair_file (str): Path to the .npz file containing sample pairs
        transition_idx (int): Index of the temperature transition to use
    """
    def __init__(self, pair_file, transition_idx):
        """Initialize the dataset with pairs from a specific transition."""
        self.transition_idx = transition_idx
        
        # Load the pairs data
        logger.info(f"Loading pairs from {pair_file}")
        data = np.load(pair_file, allow_pickle=True)
        self.temps = data['temps']
        
        # Extract pairs for the specified transition
        pairs = data['pairs']
        if transition_idx >= pairs.shape[0]:
            raise ValueError(f"Transition index {transition_idx} out of range. Max: {pairs.shape[0] - 1}")
        
        # The pairs array has shape (num_transitions, 2, num_samples, 2)
        # where pairs[k, 0] are the source samples at temperature k
        # and pairs[k, 1] are the target samples at temperature k+1
        source_array = pairs[transition_idx, 0]  # Shape (num_samples, 2)
        target_array = pairs[transition_idx, 1]  # Shape (num_samples, 2)
        
        # Check if we need to convert from object array
        if source_array.dtype == np.dtype('O'):
            logger.info("Converting object arrays to float32")
            # Create a regular numpy array from the object array
            source_array = np.stack([np.array(x, dtype=np.float32) for x in source_array])
            target_array = np.stack([np.array(x, dtype=np.float32) for x in target_array])
        
        # Convert to PyTorch tensors on CPU (will be moved to GPU during training)
        self.sources = torch.tensor(source_array, dtype=torch.float32)
        self.targets = torch.tensor(target_array, dtype=torch.float32)
        
        logger.info(f"Loaded {len(self.sources)} pairs for transition {transition_idx} "
                   f"(T_{transition_idx}={self.temps[transition_idx]:.4f} → "
                   f"T_{transition_idx+1}={self.temps[transition_idx+1]:.4f})")
    
    def __len__(self):
        """Return the number of pairs in the dataset."""
        return len(self.sources)
    
    def __getitem__(self, idx):
        """Return a pair of samples at the given index."""
        return self.sources[idx], self.targets[idx]


def make_pair_loader(pair_file, transition_idx, batch_size, val_split=0.1, 
                     shuffle=True, num_workers=4, seed=42):
    """
    Create a DataLoader for a specific temperature transition.
    
    Args:
        pair_file (str): Path to the .npz file containing sample pairs
        transition_idx (int): Index of the temperature transition to use
        batch_size (int): Batch size for the DataLoader
        val_split (float, optional): Fraction of data to use for validation
        shuffle (bool, optional): Whether to shuffle the data
        num_workers (int, optional): Number of worker processes for the DataLoader
        seed (int, optional): Random seed for reproducible train/val splits
        
    Returns:
        tuple: (train_loader, val_loader) for the specified transition
    """
    # Create the dataset
    dataset = GMMPairDataset(pair_file, transition_idx)
    
    # Split into training and validation sets with fixed seed for reproducibility
    val_size = max(1, int(len(dataset) * val_split))
    if val_size >= len(dataset):
        val_size = len(dataset) - 1  # ensure at least one train sample
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Only pin if CUDA is available
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Only pin if CUDA is available
    )
    
    logger.info(f"Created DataLoader with {len(train_loader)} training batches "
               f"and {len(val_loader)} validation batches (using seed {seed})")
    
    return train_loader, val_loader


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_loss(flow_model, x_src, gmm, temp_next, device):
    """Compute the KL/change-of-variables objective for one batch.

    Loss  =  −E_{x∼π_k} [ log π_{k+1}( f_k(x) )  +  log|det J_{f_k}(x)| ].

    Because the *absolute* normalising constant of π_{k+1} is independent of
    the flow parameters, we can safely omit it.  For a tempered energy model
    like the GMM this is equivalent to dividing the base log-density by the
    target temperature T_{k+1}.

    Args:
        flow_model (nn.Module):   The temperature-transition flow f_k.
        x_src      (Tensor):      Samples from π_k     with shape ``[B,2]``.
        gmm        (GMM):         Analytic GMM target distribution (T=1).
        temp_next  (float | Tensor): Temperature T_{k+1} for the *target* replica.
        device     (torch.device): Device to perform the computation on.

    Returns:
        torch.Tensor: A scalar loss suitable for ``.backward()``.
    """

    # Ensure data are on the correct device and require grad so that gradients
    # can flow back to the input (important for some invertible architectures).
    x_src = x_src.to(device)
    if not x_src.requires_grad:
        x_src = x_src.detach().requires_grad_(True)

    # Forward mapping (x_k  →  y) together with log-determinant of dy/dx.
    y, logdet = flow_model.inverse_and_logdet(x_src)

    # Log-density of y under the analytic GMM at temperature T_{k+1}.
    #   π_T   ∝  p_base^{1/T}
    log_p_tgt = gmm.log_prob(y) / temp_next

    # Final objective (negated expectation).
    loss = -(log_p_tgt + logdet).mean()

    return loss


def train_epoch(flow_model, train_loader, optimizer, gmm, temp_next, device, max_grad_norm=1.0):
    """Train for one epoch."""
    flow_model.train()
    total_loss = 0
    
    for batch_idx, (x, x_target) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = compute_loss(flow_model, x, gmm, temp_next, device)
        loss.backward()
        
        # Clip gradients to prevent explosions
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)


def validate(flow_model, val_loader, gmm, temp_next, device):
    """Compute validation loss."""
    flow_model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, x_target in val_loader:
            loss = compute_loss(flow_model, x, gmm, temp_next, device)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)


def _build_gmm_from_config(cfg: dict, device: torch.device):
    """Helper that instantiates the analytic GMM given the YAML section."""
    gmm_cfg = cfg["gmm"]
    return GMM(
        dim=gmm_cfg["dim"],
        n_mixes=gmm_cfg["n_mixes"],
        loc_scaling=gmm_cfg["loc_scaling"],
        device=device,
    )


def train_flows(config: dict | None = None, *, config_path: str | None = None):
    """Train temperature-transition flows for each adjacent temperature pair.

    One of ``config`` **or** ``config_path`` must be provided.  When ``config``
    is given we use it directly (useful for CLI overrides); otherwise we load
    the YAML from ``config_path``.
    """
    if config is None:
        if config_path is None:
            config_path = "configs/pt/gmm.yaml"
        config = load_config(config_path)
    else:
        # Make a *shallow* copy so later modifications don't mutate the caller
        config = {**config}

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Training hyperparameters from config or defaults
    training_config = config.get('training', {})
    n_epochs = training_config.get('n_epochs', 100)
    batch_size = training_config.get('batch_size', 256)
    lr = training_config.get('learning_rate', 1e-4)
    patience = training_config.get('patience', 20)
    max_grad_norm = training_config.get('max_grad_norm', 1.0)
    val_split = training_config.get('val_split', 0.1)
    seed = training_config.get('seed', 42)
    val_freq = training_config.get('val_freq', 1)  # Validate every N epochs
    
    # I/O settings
    model_dir = Path(training_config.get('model_dir', 'models'))
    model_dir.mkdir(exist_ok=True)
    
    pair_file = config['io'].get('pair_file') or os.path.join(config['io']['save_fold'], 'gmm_pairs.npz')
    
    # Initialize wandb if available
    if WANDB:
        wandb.init(
            project="temp-transition-flows",
            config=config,
            name=f"train_{config['name']}"
        )
    
    # ------------------------------------------------------------------
    #  Build analytic base distribution once (T=1)
    # ------------------------------------------------------------------
    gmm = _build_gmm_from_config(config, device)
    
    # For each temperature transition
    n_transitions = config['pt']['total_n_temp'] - 1
    for k in range(n_transitions):
        logger.info(f"Training flow for temperature transition {k} -> {k+1}")
        
        # Load dataset
        train_loader, val_loader = make_pair_loader(
            pair_file, k, batch_size,
            val_split=val_split,
            seed=seed,
        )
        
        # Temperature of the *target* replica k+1
        temps_array = train_loader.dataset.dataset.temps  # Subset → original dataset
        temp_next = float(temps_array[k + 1])
        
        # Create flow model
        flow_model = create_temp_transition_flow(config).to(device)
        optimizer = optim.Adam(flow_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_path = model_dir / f'flow_{k}.pth'
        
        for epoch in range(n_epochs):
            train_loss = train_epoch(
                flow_model,
                train_loader,
                optimizer,
                gmm,
                temp_next,
                device,
                max_grad_norm,
            )
            
            # Only validate every val_freq epochs (always validate first and last epoch)
            if epoch % val_freq == 0 or epoch == n_epochs - 1:
                val_loss = validate(
                    flow_model,
                    val_loader,
                    gmm,
                    temp_next,
                    device,
                )
                
                # Log metrics with single precision
                metrics = {
                    f'flow_{k}/train_loss': train_loss,
                    f'flow_{k}/val_loss': val_loss,
                    f'flow_{k}/lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                }
                
                if WANDB:
                    wandb.log(metrics)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        flow_model.state_dict(),
                        best_model_path
                    )
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, early_stop_counter={early_stop_counter}"
                )
                
                # Early stopping
                if early_stop_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                # Log only training metrics on non-validation epochs
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
                if WANDB:
                    wandb.log({
                        f'flow_{k}/train_loss': train_loss,
                        'epoch': epoch
                    })
        
        # Load the best model
        logger.info(f"Loading best model from {best_model_path}")
        flow_model.load_state_dict(torch.load(best_model_path))
    
    if WANDB:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train temperature transition flows")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the configuration file")
    parser.add_argument("--pair-file", type=str, default=None,
                        help="Path to the .npz file containing sample pairs (overrides config)")
    parser.add_argument("--transition", type=int, default=0,
                        help="Index of the temperature transition to use")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (overrides config)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory to save model checkpoints (overrides config)")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (just load and display some data)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Test mode - just load and display some data
    if args.test:
        # Create a dataset and loader
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        try:
            # Load raw data first to inspect its structure
            pair_file = args.pair_file or "data/pt/gmm_pairs.npz"
            raw_data = np.load(pair_file, allow_pickle=True)
            logger.info(f"Raw data keys: {list(raw_data.keys())}")
            logger.info(f"Pairs shape: {raw_data['pairs'].shape}")
            logger.info(f"Temperatures: {raw_data['temps']}")
            
            # Create dataset and examine samples
            dataset = GMMPairDataset(pair_file, args.transition)
            
            # Display a few samples
            logger.info("Sample pairs:")
            for i in range(min(5, len(dataset))):
                source, target = dataset[i]
                # Move to device just for display purposes
                source, target = source.to(device), target.to(device)
                logger.info(f"Pair {i}: Source {source.shape} {source.tolist()}, Target {target.shape} {target.tolist()}")
            
            # Create a DataLoader
            batch_size = args.batch_size or 128
            seed = args.seed or 42
            train_loader, val_loader = make_pair_loader(
                pair_file, args.transition, batch_size, seed=seed
            )
            
            # Display a batch
            logger.info("Sample batch:")
            for sources, targets in train_loader:
                # Move batch to device (correct pattern for training)
                sources, targets = sources.to(device), targets.to(device)
                logger.info(f"Batch shapes: Sources {sources.shape}, Targets {targets.shape}")
                break
                
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        # Load/override configuration when a path was supplied so that we can
        # inject CLI overrides *before* training begins.
        cfg = load_config(args.config or "configs/pt/gmm.yaml")
        if 'training' not in cfg:
            cfg['training'] = {}

        if args.batch_size:
            cfg['training']['batch_size'] = args.batch_size
        if args.seed:
            cfg['training']['seed'] = args.seed
        if args.model_dir:
            cfg['training']['model_dir'] = args.model_dir
        if args.pair_file:
            cfg['io']['pair_file'] = args.pair_file

        # Run the training
        train_flows(config=cfg) 