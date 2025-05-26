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
from src.accelmd.utils.config import load_config
from src.accelmd.models import MODEL_REGISTRY
from src.accelmd.trainers.realnvp import train_realnvp  # keep for registry
from src.accelmd.trainers.tarflow import train_tarflow  # new backend
from src.accelmd.evaluators import swap_rate
from src.accelmd.metrics import (
    acceptance_autocorrelation,
    moving_average_acceptance,
)
from src.accelmd.targets import build_target

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
def _generate_bidirectional_plot(cfg: Dict[str, Any], ckpt_path: Path, out_dir: Path):
    """Re-use the logic from scripts/train.py to make the 2×2 sanity figure.
    
    For dimensions higher than 2, only the first two dimensions are shown in the plot.
    """
    import matplotlib.pyplot as plt  # local import keeps CLI snappy

    device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    t_low  = float(cfg["pt"]["temp_low"])
    t_high = float(cfg["pt"]["temp_high"])
    suffix = f"{t_low:.2f}_{t_high:.2f}"
    
    low_tgt = build_target(cfg, device)
    base_tgt = low_tgt
    low_tgt = base_tgt.tempered_version(t_low)
    high_tgt = base_tgt.tempered_version(t_high)

    # --- load flow -----------------------------------------------------------
    model_type = cfg.get("model_type", "realnvp")
    model_cfg = copy.deepcopy(cfg["trainer"][model_type]["model"])
    if hasattr(low_tgt, 'dim'):
        model_cfg["dim"] = low_tgt.dim
    else:
        model_cfg["dim"] = low_tgt.sample((1,)).shape[-1]
    flow = MODEL_REGISTRY[model_type](model_cfg).to(device)
    flow.load_state_dict(torch.load(ckpt_path, map_location=device))
    flow.eval()

    # --- sample + map --------------------------------------------------------
    N = 5_000
    with torch.no_grad():
        x_hi     = high_tgt.sample((N,)).to(device)
        x_lo     = low_tgt.sample((N,)).to(device)
        x_hi2lo, _ = flow.inverse(x_hi)
        x_lo2hi, _ = flow.forward(x_lo)

    # --- figure --------------------------------------------------------------
    def _scatter(ax, pts, title):
        # For higher dimensions, only plot the first two dimensions
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=.6)
        ax.set_title(title)
        ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.grid(alpha=.3)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    _scatter(ax[0, 0], x_hi.cpu(),    f"True High-T (T={t_high})")
    _scatter(ax[0, 1], x_hi2lo.cpu(), "Mapped High→Low")
    _scatter(ax[1, 0], x_lo.cpu(),    f"True Low-T (T={t_low})")
    _scatter(ax[1, 1], x_lo2hi.cpu(), "Mapped Low→High")
    plt.tight_layout()
    
    # Include temperature suffix in the filename
    out_png = out_dir / f"bidirectional_verification_{suffix}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    _LOG.info("Bidirectional verification figure saved → %s", out_png)


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

    # Always (re-)create the sanity scatter so that plots/ is complete
    _LOG.info("Generating bidirectional verification plot")
    _generate_bidirectional_plot(
        cfg,
        ckpt_expected,
        Path(cfg["output"]["plots_dir"]),
    )
    _LOG.info("Bidirectional verification plot generated")

    # Store a verbatim copy of the YAML that was *actually* used
    with open(Path(cfg["output"]["config_copy"]), "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp)
    _LOG.info("Config snapshot written.")
    _LOG.info("Training phase completed")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def _evaluate(cfg: Dict[str, Any]):
    """Run swap-rate + metrics and produce outputs."""
    _LOG.info("Starting evaluation phase")
    
    # -- 1) swap-rate (produces JSON summary) -------------------------------
    _LOG.info("Running swap-rate evaluation")
    swap_rate.run(cfg)
    _LOG.info("Swap-rate evaluation completed")

    # -- 2) legacy metrics (two extra PNGs) ---------------------------------
    _LOG.info("Running legacy metrics")
    acceptance_autocorrelation.run(cfg)
    moving_average_acceptance.run(cfg)
    _LOG.info("Legacy metrics calculation completed")

    # -- 3) enhanced sampling metrics (vanilla vs flow comparison) ----------
    _LOG.info("Running enhanced sampling evaluation metrics")
    try:
        # Import metrics modules
        from src.accelmd.metrics import integrated_autocorr_time
        from src.accelmd.metrics import effective_sample_size  
        from src.accelmd.metrics import round_trip_time
        
        # Run enhanced sampling metrics
        _LOG.info("Running integrated autocorrelation time analysis")
        integrated_autocorr_time.run(cfg)
        
        _LOG.info("Running effective sample size analysis")
        effective_sample_size.run(cfg)
        
        _LOG.info("Running round-trip time and exploration analysis")
        round_trip_time.run(cfg)
        
        _LOG.info("Enhanced sampling metrics completed successfully")
        
    except Exception as e:
        _LOG.warning("Enhanced sampling metrics failed: %s", str(e))
        _LOG.warning("Continuing with basic evaluation...")

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
    args = p.parse_args()

    if not (args.train or args.evaluate or args.run_all):
        p.error("Please specify --train, --evaluate or --run-all.")

    # First, load the config to get the log file path
    _LOG.info("Loading config from %s", args.config)
    cfg = load_config(args.config)
    
    # ─── START experiment dir & logging setup ───
    name = cfg.get("name") or Path(args.config).stem
    # derive_output_paths already put <outputs>/<name> in base_dir
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
