from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.accelmd.utils.config import load_config

# -----------------------------------------------------------------------------
# Logging setup (will be configured in main when executed as script)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _run_train_flows(cfg_path: str, enable_wandb: bool = False, disable_wandb: bool = False):
    """Train a RealNVP flow **for every adjacent temperature pair**.

    The number of models trained is ``total_n_temp - 1`` as defined in the
    YAML configuration. Each model is trained by cloning the base *cfg* and
    overriding ``pt.temp_low`` / ``pt.temp_high`` for the specific pair before
    calling ``train_realnvp``.
    """

    import copy, numpy as np
    from src.accelmd.trainers.realnvp_trainer import train_realnvp

    base_cfg = load_config(cfg_path)

    # Handle wandb settings globally (each run may still be toggled by cfg)
    if enable_wandb:
        base_cfg["wandb"] = True
    if disable_wandb:
        base_cfg["wandb"] = False

    # Compute temperature schedule
    t_low_global, t_high_global = float(base_cfg["pt"]["temp_low"]), float(base_cfg["pt"]["temp_high"])
    n = int(base_cfg["pt"]["total_n_temp"])
    schedule_type = base_cfg["pt"].get("temp_schedule", "geom")
    temps = (
        np.linspace(t_low_global, t_high_global, n)
        if schedule_type == "linear"
        else np.geomspace(t_low_global, t_high_global, n)
    )

    # Round to 2 decimal places to match checkpoint naming convention
    temps = [round(float(t), 2) for t in temps]

    logger.info("Training flows for temperature pairs: %s", temps)

    for idx in range(len(temps) - 1):
        t_low, t_high = float(temps[idx]), float(temps[idx + 1])
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("config_files", [cfg_path])
        cfg["pt"]["temp_low"] = t_low
        cfg["pt"]["temp_high"] = t_high

        logger.info(
            "[Flow %d/%d] Training RealNVP for T_low=%s → T_high=%s",
            idx + 1,
            len(temps) - 1,
            t_low,
            t_high,
        )

        train_realnvp(cfg)

    logger.info("Flow training for all temperature pairs finished ✓")


def _run_eval_metrics(cfg_path: str, enable_wandb: bool = False, disable_wandb: bool = False):
    """Run the complete evaluation pipeline including swap rates and additional metrics.
    
    This uses the centralized evaluator which:
    1. First runs gmm_swap_rate to generate base data
    2. Then processes that data through all metric modules
    """
    from src.accelmd.evaluators.evaluator import evaluate

    cfg = load_config(cfg_path)
    
    # Handle wandb settings
    if enable_wandb:
        cfg["wandb"] = True
    if disable_wandb:
        cfg["wandb"] = False
        
    logger.info("Starting full evaluation pipeline …")
    evaluate(cfg)
    logger.info("Evaluation pipeline finished ✓")


def _run_eval_swap(cfg_path: str, enable_wandb: bool = False, disable_wandb: bool = False):
    """Run only the extreme-swap evaluator without additional metrics.
    
    For backward compatibility. Consider using eval-metrics instead.
    """
    from src.accelmd.evaluators.gmm_swap_rate import main as eval_main

    cfg = load_config(cfg_path)
    
    # Handle wandb settings
    if enable_wandb:
        cfg["wandb"] = True
    if disable_wandb:
        cfg["wandb"] = False
        
    logger.info("Starting swap-rate evaluation (basic metrics only) …")
    eval_main(cfg_path)
    logger.info("Basic swap-rate evaluation finished ✓")


def main():
    parser = argparse.ArgumentParser(
        description="One-stop runner for temperature-transition experiments on 2-D GMMs"
    )
    
    # Global arguments that apply to all commands
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train-flows ---------------------------------------------------------
    p_train = subparsers.add_parser(
        "train-flows", help="Train RealNVP temperature-transition flows"
    )
    p_train.add_argument(
        "--config", type=str, required=True, help="Path to YAML config (e.g. configs/pt/gmm.yaml)"
    )
    p_train.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p_train.add_argument("--no-wandb", action="store_true", help="Disable wandb logging even if available")

    # --- eval-metrics (new, preferred way) -----------------------------------
    p_eval_full = subparsers.add_parser(
        "eval-metrics", help="Run complete evaluation with all metrics (swap rates, plots, and stats)"
    )
    p_eval_full.add_argument(
        "--config", type=str, required=True, help="Path to YAML config (e.g. configs/pt/gmm.yaml)"
    )
    p_eval_full.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p_eval_full.add_argument("--no-wandb", action="store_true", help="Disable wandb logging even if available")

    # --- eval-swap (legacy) --------------------------------------------------
    p_eval = subparsers.add_parser(
        "eval-swap", help="Evaluate basic extreme-swap acceptance rates only"
    )
    p_eval.add_argument(
        "--config", type=str, required=True, help="Path to YAML config (e.g. configs/pt/gmm.yaml)"
    )
    p_eval.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p_eval.add_argument("--no-wandb", action="store_true", help="Disable wandb logging even if available")

    # --- run-all -------------------------------------------------------------
    p_all = subparsers.add_parser(
        "run-all",
        help="Run both flow training and complete evaluation in sequence",
    )
    p_all.add_argument(
        "--config", type=str, required=True, help="Path to YAML config (e.g. configs/pt/gmm.yaml)"
    )
    p_all.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p_all.add_argument("--no-wandb", action="store_true", help="Disable wandb logging even if available")

    args = parser.parse_args()

    # Configure root logger once here
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    cfg_path = Path(getattr(args, "config", "")).as_posix()
    enable_wandb = getattr(args, "wandb", False)
    disable_wandb = getattr(args, "no_wandb", False)

    if args.command == "train-flows":
        _run_train_flows(cfg_path, enable_wandb, disable_wandb)
    elif args.command == "eval-swap":
        _run_eval_swap(cfg_path, enable_wandb, disable_wandb)
    elif args.command == "eval-metrics":
        _run_eval_metrics(cfg_path, enable_wandb, disable_wandb)
    elif args.command == "run-all":
        _run_train_flows(cfg_path, enable_wandb, disable_wandb)
        # Use the full evaluation pipeline for run-all
        _run_eval_metrics(cfg_path, enable_wandb, disable_wandb)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
