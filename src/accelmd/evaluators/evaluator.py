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

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Local imports
from src.accelmd.utils.config import load_config
from src.accelmd.evaluators import metrics as _metrics_pkg
from src.accelmd.evaluators import gmm_swap_rate as _swap_mod

import copy
import numpy as np

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


def _temperature_schedule(pt_cfg: Dict[str, Any]) -> List[float]:
    """Compute list of temperatures according to *pt_cfg*."""
    t_low, t_high = float(pt_cfg["temp_low"]), float(pt_cfg["temp_high"])
    n = int(pt_cfg["total_n_temp"])
    schedule = pt_cfg.get("temp_schedule", "geom")
    if n < 2:
        raise ValueError("total_n_temp must be >= 2")
    if schedule == "linear":
        temps = np.linspace(t_low, t_high, n)
    else:  # default geom
        temps = np.geomspace(t_low, t_high, n)
    # Round to 2 decimal places to match checkpoint naming convention
    return [round(float(t), 2) for t in temps]


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(cfg: Dict[str, Any]) -> None:  # noqa: D401 – imperative API
    """Run the full evaluation pipeline.
    
    This function:
    1. Iterates over each adjacent temperature pair in the schedule
    2. Runs gmm_swap_rate for that pair to generate base metrics and histories
    3. Executes each metric module for that pair
    
    Parameters
    ----------
    cfg : dict
        The loaded configuration dictionary (baseline, extreme temps)
    """
    # 0) Ensure output directories exist
    for key in ("plot_dir", "results_dir"):
        Path(cfg["evaluator"][key]).mkdir(parents=True, exist_ok=True)

    temps = _temperature_schedule(cfg["pt"])
    logger.info("Temperature schedule: %s", temps)

    for i in range(len(temps) - 1):
        t_low, t_high = temps[i], temps[i + 1]
        logger.info("Evaluating pair T_low=%s  →  T_high=%s", t_low, t_high)

        pair_cfg = copy.deepcopy(cfg)
        pair_cfg["pt"]["temp_low"] = float(t_low)
        pair_cfg["pt"]["temp_high"] = float(t_high)

        # 1) Run swap-rate experiment for this pair
        _swap_mod.run(pair_cfg)

        # 2) Run downstream metrics
        for mod in _discover_metric_modules():
            logger.info("  ↳ Running metric: %s.run", mod.__name__)
            if not hasattr(mod, "run"):
                raise AttributeError(
                    f"Metric module '{mod.__name__}' lacks a 'run' function")
            mod.run(pair_cfg)
            
    # No need to call wandb.finish() here as gmm_swap_rate already handles it per-pair
    try:
        # Check for any lingering wandb runs
        if WANDB_AVAILABLE and "wandb" in cfg and cfg["wandb"] and wandb.run is not None:
            logger.warning("Found lingering wandb run at end of evaluation - finishing it")
            wandb.finish()
    except Exception as e:
        logger.warning(f"Error finalizing wandb: {e}")

    print("\N{white heavy check mark} All metrics complete for all temperature pairs.")


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
    evaluate(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
