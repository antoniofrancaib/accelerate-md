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
import argparse, logging, shutil, json, copy
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ­%(levelname)s ­%(name)s ­ %(message)s",
)

def _experiment_dir(cfg: Dict[str, Any]) -> Path:
    """Compute  …/outputs/<experiment-name>  (create if necessary)."""
    name      = cfg["name"]
    base_dir  = cfg.get("output", {}).get("base_dir", "outputs")
    exp_dir   = Path(base_dir) / name
    (exp_dir / "plots").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    return exp_dir


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def _generate_bidirectional_plot(cfg: Dict[str, Any], ckpt_path: Path, out_png: Path):
    """Re-use the logic from scripts/train.py to make the 2×2 sanity figure.
    
    For dimensions higher than 2, only the first two dimensions are shown in the plot.
    """
    import matplotlib.pyplot as plt  # local import keeps CLI snappy

    device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    t_low  = float(cfg["pt"]["temp_low"])
    t_high = float(cfg["pt"]["temp_high"])
    low_tgt  = build_target(cfg, device)
    high_tgt = low_tgt.tempered_version(t_high)

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
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    _LOG.info("Bidirectional verification figure saved → %s", out_png)


def _train(cfg: Dict[str, Any], exp_dir: Path, target):
    """Run training (if not already cached) and copy artefacts into *exp_dir*."""
    print(f"[DEBUG MAIN] Starting training phase")
    t_low  = float(cfg["pt"]["temp_low"])
    t_high = float(cfg["pt"]["temp_high"])
    ckpt_expected = exp_dir / "model.pt"

    if ckpt_expected.is_file():
        _LOG.info("Model checkpoint already present – skipping training.")
        print(f"[DEBUG MAIN] Using existing model checkpoint: {ckpt_expected}")
    else:
        print(f"[DEBUG MAIN] No checkpoint found at {ckpt_expected}, running training...")
        trainer = TRAINER_REGISTRY[cfg.get("model_type", "realnvp")]
        ckpt_path = trainer(cfg, target)  # does the heavy lifting
        shutil.copy2(ckpt_path, ckpt_expected)  # standardised name inside outputs/
        _LOG.info("Checkpoint copied to %s", ckpt_expected)
        print(f"[DEBUG MAIN] Training completed, checkpoint saved to {ckpt_expected}")

    # Always (re-)create the sanity scatter so that plots/ is complete
    print(f"[DEBUG MAIN] Generating bidirectional verification plot")
    _generate_bidirectional_plot(
        cfg,
        ckpt_expected,
        exp_dir / "plots" / "bidirectional_verification.png",
    )
    print(f"[DEBUG MAIN] Bidirectional verification plot generated")

    # Store a verbatim copy of the YAML that was *actually* used
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp)
    _LOG.info("Config snapshot written.")
    print(f"[DEBUG MAIN] Training phase completed")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def _evaluate(cfg: Dict[str, Any], exp_dir: Path):
    """Run swap-rate + metrics and collate outputs inside *exp_dir*."""
    print(f"[DEBUG MAIN] Starting evaluation phase")
    # We point the evaluator sub-modules *into*   outputs/<name>/…
    cfg_eval = copy.deepcopy(cfg)  # avoid polluting the original dict
    
    # Ensure evaluator section exists and has the required fields
    if "evaluator" not in cfg_eval:
        cfg_eval["evaluator"] = {}
    
    # Set plot and results directories
    cfg_eval["evaluator"]["plot_dir"] = str(exp_dir / "plots")
    cfg_eval["evaluator"]["results_dir"] = str(exp_dir)
    
    # Make sure they exist
    Path(cfg_eval["evaluator"]["plot_dir"]).mkdir(exist_ok=True, parents=True)
    Path(cfg_eval["evaluator"]["results_dir"]).mkdir(exist_ok=True, parents=True)

    # -- 1) swap-rate (produces JSON summary) -------------------------------
    print(f"[DEBUG MAIN] Running swap-rate evaluation")
    swap_rate.run(cfg_eval)
    print(f"[DEBUG MAIN] Swap-rate evaluation completed")

    # -- 2) metrics (two extra PNGs) ----------------------------------------
    print(f"[DEBUG MAIN] Running metrics")
    acceptance_autocorrelation.run(cfg_eval)
    moving_average_acceptance.run(cfg_eval)
    print(f"[DEBUG MAIN] Metrics calculation completed")

    # -- 3) copy / rename JSON → metrics.json -------------------------------
    from src.accelmd.evaluators.swap_rate import _pair_suffix
    tl, th = float(cfg_eval["pt"]["temp_low"]), float(cfg_eval["pt"]["temp_high"])
    json_src = exp_dir / f"gmm_swap_rate_{_pair_suffix(tl, th)}.json"
    json_dst = exp_dir / "metrics.json"
    shutil.copy2(json_src, json_dst)
    _LOG.info("Metrics aggregated → %s", json_dst)
    print(f"[DEBUG MAIN] Metrics aggregated to {json_dst}")
    print(f"[DEBUG MAIN] Evaluation phase completed")


# --------------------------------------------------------------------------- #
#                                     CLI                                     #
# --------------------------------------------------------------------------- #
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

    print(f"[DEBUG MAIN] Loading config from {args.config}")
    cfg = load_config(args.config)
    exp_dir = _experiment_dir(cfg)
    print(f"[DEBUG MAIN] Experiment directory: {exp_dir}")
    
    print(f"[DEBUG MAIN] Running with options: train={args.train}, evaluate={args.evaluate}, run-all={args.run_all}")
    
    # Use CPU if explicitly requested
    if args.cpu:
        device = torch.device("cpu")
        # Override config device for all downstream code
        cfg["device"] = "cpu"
    else:
        device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    if args.run_all:
        target = build_target(cfg, device)
        _train(cfg, exp_dir, target)
        _evaluate(cfg, exp_dir)
    elif args.train:
        target = build_target(cfg, device)
        _train(cfg, exp_dir, target)
    elif args.evaluate:
        _evaluate(cfg, exp_dir)
    
    print(f"[DEBUG MAIN] Experiment completed successfully")


if __name__ == "__main__":
    main()
