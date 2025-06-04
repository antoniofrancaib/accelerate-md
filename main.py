#!/usr/bin/env python
"""
Unified entry-point for the GMM ⇆ RealNVP experiments.

Flags
-----
--train       Train the RealNVP flow (and dump the bidirectional-scatter sanity plot)
--evaluate    Run swap-rate evaluation + all metrics/plots
--run-all     Do both, in that order

Example
-------
python main.py --config configs/gmm.yaml --run-all
"""

from __future__ import annotations
import argparse, logging, shutil, json, copy, sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml

# ---- project imports --------------------------------------------------------
from src.accelmd.utils.config import load_config, build_local_kernel, build_swap_kernel
from src.accelmd.models import MODEL_REGISTRY
from src.accelmd.trainers.realnvp import train_realnvp  # keep for registry
from src.accelmd.trainers.tarflow import train_tarflow  # new backend
from src.accelmd.evaluators import swap_rate
from src.accelmd.metrics import (
    acceptance_autocorrelation,
    moving_average_acceptance,
)
from src.accelmd.targets import build_target

# Import plot generation scripts
from scripts.generate_bidirectional_plots import generate_bidirectional_plot
from scripts.generate_ramachandran_plots import generate_ramachandran_comparison_for_pair

# Unified trainer registry for dynamic backend selection
TRAINER_REGISTRY = {
    "realnvp": train_realnvp,
    "tarflow": train_tarflow,
}

# --------------------------------------------------------------------------- #
#                          small internal helpers                             #
# --------------------------------------------------------------------------- #
_LOG = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def _train(cfg: Dict[str, Any], target):
    """Run training (if not already cached)."""
    _LOG.info("Starting training phase")
    
    # The checkpoint path is now consistent with what trainers produce
    ckpt_expected = Path(cfg["output"]["model_path"])

    if ckpt_expected.is_file():
        _LOG.info("Model checkpoint already present – skipping training.")
        _LOG.info("Using existing model checkpoint: %s", ckpt_expected)
    else:
        _LOG.info("No checkpoint found at %s, running training...", ckpt_expected)
        trainer = TRAINER_REGISTRY[cfg.get("model_type", "realnvp")]
        ckpt_path = trainer(cfg, target)  # does the heavy lifting
        # No need to copy - trainer saves directly to the right location
        _LOG.info("Training completed, checkpoint saved to %s", ckpt_path)

    # Generate training loss plot if enabled
    plots_config = cfg.get("plots_and_metrics", {})
    if plots_config.get("training_loss", True):
        _LOG.info("Generating training loss plot")
        try:
            from scripts.generate_training_plots import generate_training_loss_plot
            
            history_path = Path(cfg["output"]["training_history"])
            plots_dir = Path(cfg["output"]["plots_dir"])
            
            # Generate a meaningful suffix for the plot
            t_low = cfg["pt"].get("temp_low", "low")
            t_high = cfg["pt"].get("temp_high", "high") 
            suffix = f"{t_low:.2f}_{t_high:.2f}"
            
            output_path = plots_dir / f"training_loss_{suffix}.png"
            
            if history_path.exists():
                generate_training_loss_plot(history_path, output_path)
                _LOG.info("Training loss plot generated successfully")
            else:
                _LOG.warning(f"Training history not found at {history_path}. Skipping training loss plot.")
                
        except Exception as e:
            _LOG.error(f"Failed to generate training loss plot: {str(e)}")
            _LOG.warning("Continuing without training loss plot...")

    # Generate plots based on experiment type
    experiment_type = cfg.get("experiment_type", "gmm")
    plots_dir = Path(cfg["output"]["plots_dir"])
    
    if experiment_type == "gmm" and plots_config.get("bidirectional_verification", True):
        # Generate bidirectional verification plot for GMM experiments
        _LOG.info("Generating bidirectional verification plot for GMM experiment")
        generate_bidirectional_plot(
            cfg,
            ckpt_expected,
            plots_dir,
        )
        _LOG.info("Bidirectional verification plot generated")
    elif experiment_type == "aldp":
        # For ALDP experiments, plots will be generated during evaluation
        _LOG.info("ALDP experiment detected - Ramachandran plots will be generated during evaluation")
    else:
        _LOG.warning(f"Unknown experiment type: {experiment_type}. No plots generated during training.")

    _LOG.info("Training phase completed")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def _evaluate(cfg: Dict[str, Any]):
    """Run swap-rate + metrics and produce outputs."""
    _LOG.info("Starting evaluation phase")
    
    # Get plot/metrics configuration
    plots_config = cfg.get("plots_and_metrics", {})
    
    # Build kernels for the evaluation phase
    _LOG.info("Building kernel interfaces...")
    local_kernel = build_local_kernel(cfg)
    swap_kernel = build_swap_kernel(cfg)
    
    if local_kernel and swap_kernel:
        _LOG.info("Using new kernel interfaces for evaluation")
        # Store kernels in config for evaluators to use
        cfg["_kernels"] = {
            "local": local_kernel,
            "swap": swap_kernel
        }
    else:
        _LOG.info("Using legacy evaluation path (no kernel interfaces)")
    
    # -- 1) swap-rate (produces JSON summary) -------------------------------
    if plots_config.get("swap_rate_evaluation", True):
        _LOG.info("Running swap-rate evaluation")
        swap_rate.run(cfg)
        _LOG.info("Swap-rate evaluation completed")
    else:
        _LOG.info("Swap-rate evaluation disabled by configuration")

    # -- 2) legacy metrics (two extra PNGs) ---------------------------------
    if plots_config.get("acceptance_autocorrelation", True) or plots_config.get("moving_average_acceptance", True):
        _LOG.info("Running legacy metrics")
        if plots_config.get("acceptance_autocorrelation", True):
            acceptance_autocorrelation.run(cfg)
        if plots_config.get("moving_average_acceptance", True):
            moving_average_acceptance.run(cfg)
        _LOG.info("Legacy metrics calculation completed")
    else:
        _LOG.info("Legacy metrics disabled by configuration")

    # -- 3) enhanced sampling metrics (vanilla vs flow comparison) ----------
    enhanced_metrics_enabled = any([
        plots_config.get("integrated_autocorr_time", True),
        plots_config.get("effective_sample_size", True),
        plots_config.get("round_trip_time", True)
    ])
    
    if enhanced_metrics_enabled:
        _LOG.info("Running enhanced sampling evaluation metrics")
        try:
            # Import metrics modules
            from src.accelmd.metrics import integrated_autocorr_time
            from src.accelmd.metrics import effective_sample_size  
            from src.accelmd.metrics import round_trip_time
            
            # Run enhanced sampling metrics based on config
            if plots_config.get("integrated_autocorr_time", True):
                _LOG.info("Running integrated autocorrelation time analysis")
                integrated_autocorr_time.run(cfg)
            
            if plots_config.get("effective_sample_size", True):
                _LOG.info("Running effective sample size analysis")
                effective_sample_size.run(cfg)
            
            if plots_config.get("round_trip_time", True):
                _LOG.info("Running round-trip time and exploration analysis")
                round_trip_time.run(cfg)
            
            _LOG.info("Enhanced sampling metrics completed successfully")
            
        except Exception as e:
            _LOG.warning("Enhanced sampling metrics failed: %s", str(e))
            _LOG.warning("Continuing with basic evaluation...")
    else:
        _LOG.info("Enhanced sampling metrics disabled by configuration")

    # -- 4) Generate experiment-specific plots -------------------------------
    experiment_type = cfg.get("experiment_type", "gmm")
    
    if experiment_type == "aldp" and plots_config.get("ramachandran_comparison", True):
        # Generate Ramachandran plots for ALDP experiments
        _LOG.info("Generating Ramachandran comparison plots for ALDP experiment")
        try:
            device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
            t_low = float(cfg["pt"]["temp_low"])
            t_high = float(cfg["pt"]["temp_high"])
            
            flow_model_path = Path(cfg["output"]["model_path"])
            ramachandran_output_dir = Path(cfg["output"]["plots_dir"]) / "ramachandran_comparison"
            
            if flow_model_path.exists():
                generate_ramachandran_comparison_for_pair(
                    cfg, t_low, t_high, flow_model_path, ramachandran_output_dir, device
                )
                _LOG.info("Ramachandran comparison plots generated successfully")
            else:
                _LOG.warning(f"Flow model not found at {flow_model_path}. Skipping Ramachandran plots.")
                
        except Exception as e:
            _LOG.error(f"Failed to generate Ramachandran plots: {str(e)}")
            _LOG.warning("Continuing without Ramachandran plots...")
    
    elif experiment_type == "gmm":
        _LOG.info("GMM experiment detected - bidirectional plots already generated during training")
    
    else:
        _LOG.warning(f"Unknown experiment type: {experiment_type}. No additional plots generated.")

    # Metrics are now stored directly at the expected location via template
    _LOG.info("Metrics calculated → %s", cfg["output"]["metric_json"])
    _LOG.info("Evaluation phase completed")


# --------------------------------------------------------------------------- #
#                                     CLI                                     #
# --------------------------------------------------------------------------- #
def configure_logging(log_file):
    """Configure root logger to write to both file and console."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Create handlers
    file_handler = logging.FileHandler(log_file, mode="w")
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


def main() -> None:
    p = argparse.ArgumentParser(description="Altan RealNVP ⇆ GMM driver")
    p.add_argument("--config", type=str, required=True,
                   help="Path to experiment YAML (e.g. configs/gmm.yaml)")
    p.add_argument("--train",   action="store_true",
                   help="Train a RealNVP model")
    p.add_argument("--evaluate", action="store_true",
                   help="Evaluate model and generate plots")
    p.add_argument("--run-all", action="store_true",
                   help="Do both training and evaluation")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU usage even if GPU is available")
    p.add_argument("--use-legacy-kernels", action="store_true",
                   help="Force use of legacy kernel implementation")
    args = p.parse_args()

    if not (args.train or args.evaluate or args.run_all):
        p.error("Please specify --train, --evaluate or --run-all.")

    # First, load the config to get the log file path
    _LOG.info("Loading config from %s", args.config)
    cfg = load_config(args.config)
    
    # Override kernel usage if requested
    if args.use_legacy_kernels:
        _LOG.info("Legacy kernel mode forced by command line argument")
        cfg.pop("local_kernel", None)
        cfg.pop("swap_kernel", None)
    
    # ─── START experiment dir & logging setup ───
    name = cfg.get("name") or Path(args.config).stem
    # derive_output_paths already put <outputs>/<n> in base_dir
    exp_dir = Path(cfg["output"]["base_dir"])

    # Standard sub-directories (models ≡ checkpoints, metrics ≡ results)
    models_dir   = exp_dir / "models"
    metrics_dir  = exp_dir / "metrics"
    plots_dir    = exp_dir / "plots"

    for d in (models_dir, metrics_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Copy original YAML once (over-write if re-run)
    shutil.copy(args.config, exp_dir / "config.yaml")
    
    # Configure root logging
    log_file = exp_dir / "experiment.log"
    configure_logging(log_file)
    # ─── END experiment dir & logging setup ───
    
    _LOG.info("Experiment directory: %s", exp_dir)
    
    # Configure logging to write to both file and console
    # This will update the root logger, affecting all modules
    _LOG.info("Loaded config:\n%s", yaml.dump(cfg, sort_keys=False))
    
    _LOG.info("Running with options: train=%s, evaluate=%s, run-all=%s", 
              args.train, args.evaluate, args.run_all)
    
    # Use CPU if explicitly requested
    if args.cpu:
        device = torch.device("cpu")
        # Override config device for all downstream code
        cfg["device"] = "cpu"
    else:
        device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    
    if args.run_all:
        temps = cfg["pt"]["temperatures"]
        for i in range(len(temps) - 1):
            t_low, t_high = temps[i], temps[i + 1]

            # ---- per-pair cfg tweaks -------------------------------------
            suffix = f"{t_low:.2f}_{t_high:.2f}"
            cfg["pt"].update({"temp_low": t_low, "temp_high": t_high})

            # unified directories
            cfg["output"].update({
                "checkpoints": str(models_dir),  # for trainer backward-compat
                "model_path": str(models_dir / f"flow_{suffix}.pt"),
                "training_history": str(models_dir / f"training_history_{suffix}.json"),
                "plots_dir": str(plots_dir),
                "results_dir": str(metrics_dir),
                "metric_template": f"swap_rate_flow_{suffix}.json",
                "metric_json": str(metrics_dir / f"swap_rate_flow_{suffix}.json"),
            })

            # ---- TRAIN ---------------------------------------------------
            if args.train or args.run_all:
                _LOG.info("=== Pair %s: Training ===", suffix)
                target = build_target(cfg, device)
                _train(cfg, target)

            # ---- EVALUATE -----------------------------------------------
            if args.evaluate or args.run_all:
                _LOG.info("=== Pair %s: Evaluating ===", suffix)
                _evaluate(cfg)
            
    elif args.train or args.evaluate:
        # Re-enter the loop with run_all=False so we honour the flags
        temps = cfg["pt"]["temperatures"]
        for i in range(len(temps) - 1):
            t_low, t_high = temps[i], temps[i + 1]
            suffix = f"{t_low:.2f}_{t_high:.2f}"

            cfg["pt"].update({"temp_low": t_low, "temp_high": t_high})
            cfg["output"].update({
                "checkpoints": str(models_dir),
                "model_path": str(models_dir / f"flow_{suffix}.pt"),
                "training_history": str(models_dir / f"training_history_{suffix}.json"),
                "plots_dir": str(plots_dir),
                "results_dir": str(metrics_dir),
                "metric_template": f"swap_rate_flow_{suffix}.json",
                "metric_json": str(metrics_dir / f"swap_rate_flow_{suffix}.json"),
            })

            if args.train:
                _LOG.info("=== Pair %s: Training-only ===", suffix)
                target = build_target(cfg, device)
                _train(cfg, target)

            if args.evaluate:
                _LOG.info("=== Pair %s: Evaluation-only ===", suffix)
                _evaluate(cfg)
    
    _LOG.info("Experiment completed successfully")


if __name__ == "__main__":
    # Set up basic logging before config is loaded
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        handlers=[logging.StreamHandler()])
    
    main()
