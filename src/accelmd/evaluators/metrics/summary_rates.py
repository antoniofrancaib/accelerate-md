"""Summary of naive vs flow-based acceptance rates.

Reads the JSON file produced by ``gmm_swap_rate.py`` and extracts the two
scalar acceptance rates.  The metric prints the values to **stdout** and,
optionally, logs them to *wandb* if the experiment configuration enables it.

The function signature is a shared contract across all metrics:

    def run(cfg: dict) -> None

It must be side-effect free apart from I/O (printing, logging and writing).
"""

from __future__ import annotations

# Standard library
import json
import logging
from pathlib import Path
from typing import Any, Dict

# Third-party
import numpy as np  # noqa: F401 – might be useful for future extensions

# Optional Weights-and-Biases
try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

from src.accelmd.evaluators.gmm_swap_rate import _pair_suffix


def _load_results_json(results_dir: Path, t_low: float, t_high: float) -> Dict[str, Any]:
    """Load the JSON summary for a specific temperature pair."""
    json_name = f"gmm_swap_rate_{_pair_suffix(t_low, t_high)}.json"
    json_path = results_dir / json_name
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Could not find results JSON at '{json_path}'. "
            "Has the swap-rate evaluation been run?"
        )

    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data


def run(cfg: Dict[str, Any]) -> None:  # noqa: D401 – imperative API
    """Execute the *summary_rates* metric.

    Parameters
    ----------
    cfg: dict
        Experiment configuration parsed from YAML.  Must contain the keys
        ``evaluator.results_dir`` and optionally ``wandb``.
    """
    # ------------------------------------------------------------------
    # 1) Locate & load the JSON summary
    # ------------------------------------------------------------------
    results_dir = Path(cfg["evaluator"]["results_dir"])
    t_low, t_high = float(cfg["pt"]["temp_low"]), float(cfg["pt"]["temp_high"])
    data = _load_results_json(results_dir, t_low, t_high)

    # ------------------------------------------------------------------
    # 2) Extract acceptance rates
    # ------------------------------------------------------------------
    try:
        naive_rate = float(data["naive_rate"])
        flow_rate = float(data["flow_rate"])
    except KeyError as exc:
        raise KeyError(
            f"Missing key in results JSON: {exc}. "
            "Ensure the swap-rate script stored these fields."
        ) from exc

    # ------------------------------------------------------------------
    # 3) Print to stdout & log via the Python logger
    # ------------------------------------------------------------------
    print("\nSummary Acceptance Rates:\n" "Naive PT: {nr:.4f}\nFlow-based PT: {fr:.4f}".format(nr=naive_rate, fr=flow_rate))
    logger.info("Naive PT acceptance rate = %.4f", naive_rate)
    logger.info("Flow-based PT acceptance rate = %.4f", flow_rate)

    # ------------------------------------------------------------------
    # 4) Optionally log to wandb
    # ------------------------------------------------------------------
    if _WANDB_AVAILABLE and cfg.get("wandb", False):
        try:
            if wandb.run is not None:
                wandb.log({
                    "summary_naive_rate": naive_rate,
                    "summary_flow_rate": flow_rate,
                })
                logger.debug("Logged acceptance rates to wandb.")
            else:
                logger.warning("No active wandb run found. Skipping wandb logging.")
        except Exception as e:
            logger.warning(f"Error logging to wandb: {e}")
