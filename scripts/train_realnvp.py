#!/usr/bin/env python
"""
Training script for RealNVP normalizing flow on GMM data.

This script trains a single RealNVP model on GMM data via maximum likelihood estimation.
It's intended as a simpler approach compared to the temperature transition flows.
"""

import os
import sys
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from src.utils.config import load_config, merge_dicts
from src.trainers.realnvp_trainer import train_realnvp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, logging will be disabled")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train RealNVP to map between temperature levels")
    parser.add_argument("--config", type=str, action="append",
                        default=[
                            "configs/pt/gmm.yaml",
                            "configs/pt/flow_training.yaml",
                        ],
                        help="Path(s) to YAML configuration file(s). Can be provided multiple times.")
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
    wandb_available = WANDB_AVAILABLE
    if args.no_wandb:
        wandb_available = False
    
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


if __name__ == "__main__":
    main() 