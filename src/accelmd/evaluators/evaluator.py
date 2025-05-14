"""Run all evaluation metrics for an experiment configuration.

Example usage::

    python -m src.accelmd.evaluators.evaluator \
        --config configs/pt/gmm.yaml \
        --wandb             # or --no-wandb to override

This script first executes gmm_swap_rate.py to generate the base metrics data,
then runs each metric module under ``src.accelmd.evaluators.metrics``.
"""

from __future__ import annotations

# Standard library
import argparse
import logging
from pathlib import Path
from types import ModuleType
from typing import List, Dict, Any

# Local imports
from src.accelmd.utils.config import load_config
from src.accelmd.evaluators import metrics as _metrics_pkg
from src.accelmd.evaluators.gmm_swap_rate import main as run_gmm_swap_rate

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _discover_metric_modules() -> List[ModuleType]:
    """Return a list of metric sub-modules to execute in a deterministic order."""
    names = [
        "summary_rates",
        "moving_average_acceptance",
        "acceptance_autocorrelation",
    ]
    mods: List[ModuleType] = []
    for name in names:
        mod = getattr(_metrics_pkg, name, None)
        if mod is None:
            raise ImportError(
                f"Metric module '{name}' could not be imported from '{_metrics_pkg.__name__}'."
            )
        mods.append(mod)
    return mods


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(cfg: Dict[str, Any], config_path: str) -> None:
    """Run the full evaluation pipeline.
    
    This function:
    1. Runs gmm_swap_rate to generate the base metrics and histories
    2. Executes each metric module that depends on gmm_swap_rate's output
    
    Parameters
    ----------
    cfg : dict
        The loaded configuration dictionary
    config_path : str
        Path to the original config file for passing to gmm_swap_rate
    """
    # ------------------------------------------------------------------
    # 1) Create necessary directories
    # ------------------------------------------------------------------
    # Create directories specified in evaluator section to avoid race conditions
    eval_dirs = [cfg["evaluator"].get(k) for k in ("plot_dir", "results_dir")]
    for d in eval_dirs:
        if d is None:
            continue
        Path(d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Run GMM swap rate evaluation first
    # ------------------------------------------------------------------
    logger.info("Running GMM swap rate evaluation to generate base metrics...")
    run_gmm_swap_rate(config_path)
    
    # ------------------------------------------------------------------
    # 3) Execute all metrics modules
    # ------------------------------------------------------------------
    for mod in _discover_metric_modules():
        logger.info("Running metric: %s.run", mod.__name__)
        if not hasattr(mod, "run"):
            raise AttributeError(f"Metric module '{mod.__name__}' lacks a 'run' function")
        mod.run(cfg)

    print("\N{white heavy check mark} All metrics complete.")


def main() -> None:  # noqa: D401 – imperative API
    # ------------------------------------------------------------------
    # 1) CLI parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Run evaluation metrics for accelerate-md experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging irrespective of YAML setting",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging irrespective of YAML setting",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2) Load YAML configuration
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    # Override wandb flag from CLI if requested
    if args.wandb:
        cfg["wandb"] = True
    if args.no_wandb:
        cfg["wandb"] = False

    # Ensure required top-level sections exist
    if "evaluator" not in cfg:
        raise KeyError("Config is missing required 'evaluator' section")
    if "pt" not in cfg:
        raise KeyError("Config is missing required 'pt' section")
        
    # ------------------------------------------------------------------
    # 3) Run the complete evaluation pipeline
    # ------------------------------------------------------------------
    evaluate(cfg, args.config)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
