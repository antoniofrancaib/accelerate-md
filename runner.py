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
    """Call the RealNVP trainer using a single YAML config file."""
    from src.accelmd.trainers.realnvp_trainer import train_realnvp

    cfg = load_config(cfg_path)
    cfg.setdefault("config_files", [cfg_path])  # track provenance
    
    # Handle wandb settings
    if enable_wandb:
        cfg["wandb"] = True
    if disable_wandb:
        cfg["wandb"] = False

    logger.info("Starting flow training …")
    train_realnvp(cfg)
    logger.info("Flow training finished ✓")


def _run_eval_swap(cfg_path: str, enable_wandb: bool = False, disable_wandb: bool = False):
    """Run the extreme-swap evaluator."""
    from src.accelmd.evaluators.gmm_swap_rate import main as eval_main

    cfg = load_config(cfg_path)
    
    # Handle wandb settings
    if enable_wandb:
        cfg["wandb"] = True
    if disable_wandb:
        cfg["wandb"] = False
        
    logger.info("Starting swap-rate evaluation …")
    eval_main(cfg_path)
    logger.info("Swap-rate evaluation finished ✓")


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

    # --- eval-swap -----------------------------------------------------------
    p_eval = subparsers.add_parser(
        "eval-swap", help="Evaluate extreme-swap acceptance rates"
    )
    p_eval.add_argument(
        "--config", type=str, required=True, help="Path to YAML config (e.g. configs/pt/gmm.yaml)"
    )
    p_eval.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p_eval.add_argument("--no-wandb", action="store_true", help="Disable wandb logging even if available")

    # --- run-all -------------------------------------------------------------
    p_all = subparsers.add_parser(
        "run-all",
        help="Run both flow training and swap-rate evaluation in sequence",
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
    elif args.command == "run-all":
        _run_train_flows(cfg_path, enable_wandb, disable_wandb)
        _run_eval_swap(cfg_path, enable_wandb, disable_wandb)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
