"""Moving-average acceptance rate curves for Naive vs Flow-based PT.

The metric reads the acceptance *histories* stored in
``<results_dir>/gmm_swap_rate.json`` and computes a sliding-window moving
average with a window length of

    window = cfg["pt"].get("swap_interval", 100) * 10

Both curves are plotted in a single Matplotlib figure and saved to
``<plot_dir>/moving_average_acceptance.png``.  The final plot can also be
logged to *wandb* as an image.
"""

from __future__ import annotations

# Standard library
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

# Third-party
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless backend – safe for HPC
import matplotlib.pyplot as plt  # noqa: E402

# Optional Weights-and-Biases
try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_histories(results_dir: Path, t_low: float, t_high: float, cfg: Dict[str, Any]) -> tuple[List[int], List[int]]:
    """Load histories for a specific temperature pair."""
    # Use the metric_json path directly from the config
    json_path = Path(cfg["output"]["metric_json"])
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Results JSON '{json_path}' not found. Run swap-rate evaluation first."
        )
    with open(json_path, "r", encoding="utf-8") as fp:
        data: Dict[str, Any] = json.load(fp)

    try:
        naive_hist = data["naive_hist"]
        flow_hist = data["flow_hist"]
    except KeyError as exc:
        raise KeyError(
            f"Key {exc} missing in results JSON – cannot compute moving average."
        ) from exc

    if not (isinstance(naive_hist, list) and isinstance(flow_hist, list)):
        raise TypeError("Histories must be lists of 0/1 values.")

    return naive_hist, flow_hist


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Simple moving average using convolution.

    Returns an array of length ``len(x) - win + 1``.
    """
    if win < 1:
        raise ValueError("Window length must be >= 1")
    
    # Handle the case where array is too short for the window
    if len(x) < win:
        logger.warning(f"Input array of length {len(x)} is shorter than window size {win}")
        # Adjust window size to match array length
        win = max(1, len(x) // 2)
        logger.warning(f"Adjusted window size to {win}")
    
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: Dict[str, Any]) -> None:  # noqa: D401 – imperative API
    """Compute and plot moving-average acceptance curves."""
    # ------------------------------------------------------------------
    # 1) Load data
    # ------------------------------------------------------------------
    results_dir = Path(cfg["output"]["results_dir"])
    t_low, t_high = float(cfg["pt"]["temp_low"]), float(cfg["pt"]["temp_high"])
    suffix = f"{t_low:.2f}_{t_high:.2f}"
    
    naive_hist, flow_hist = _load_histories(results_dir, t_low, t_high, cfg)

    swap_interval = int(cfg["pt"].get("swap_interval", 100))
    # Calculate window size with a sensible maximum (not more than 25% of data length)
    min_hist_len = min(len(naive_hist), len(flow_hist))
    max_window = max(2, min_hist_len // 4)
    window = min(swap_interval * 10, max_window)
    if window < 2:
        window = 2  # Ensure minimum window size of 2
    logger.debug("Using moving-average window = %d for histories of length %d", window, min_hist_len)

    na_arr = np.asarray(naive_hist, dtype=float)
    fl_arr = np.asarray(flow_hist, dtype=float)

    # Check if arrays are large enough for meaningful analysis
    if min_hist_len < 10:
        logger.warning(f"History arrays are very short ({min_hist_len} elements). Results may not be meaningful.")
        
    # Handle potential zero-length outputs
    try:
        ma_naive = _moving_average(na_arr, window)
        ma_flow = _moving_average(fl_arr, window)
    except Exception as e:
        logger.error(f"Error computing moving average: {e}")
        logger.error(f"Naive history length: {len(na_arr)}, Flow history length: {len(fl_arr)}, Window: {window}")
        raise

    # Check if we got valid results
    if len(ma_naive) == 0 or len(ma_flow) == 0:
        logger.error("Moving average calculation resulted in empty arrays")
        logger.error(f"Original lengths: naive={len(na_arr)}, flow={len(fl_arr)}, window={window}")
        
        # Fall back to a smaller window as a last resort
        fallback_window = max(1, min(len(na_arr), len(fl_arr)) // 3)
        logger.warning(f"Falling back to window size = {fallback_window}")
        ma_naive = _moving_average(na_arr, fallback_window) if len(na_arr) >= fallback_window else np.array([np.mean(na_arr)])
        ma_flow = _moving_average(fl_arr, fallback_window) if len(fl_arr) >= fallback_window else np.array([np.mean(fl_arr)])

    # X-axis: simulation step (centre of the window)
    # Make sure both arrays have the same length for plotting
    common_length = min(len(ma_naive), len(ma_flow))
    if common_length < len(ma_naive):
        logger.warning(f"Truncating naive MA from {len(ma_naive)} to {common_length} points")
        ma_naive = ma_naive[:common_length]
    if common_length < len(ma_flow):
        logger.warning(f"Truncating flow MA from {len(ma_flow)} to {common_length} points")
        ma_flow = ma_flow[:common_length]
        
    steps = np.arange(common_length) * swap_interval + window * 0.5

    # ------------------------------------------------------------------
    # 2) Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(steps, ma_naive, label="Naive PT", color="tab:blue")
    plt.plot(steps, ma_flow, label="Flow-based PT", color="tab:orange")
    plt.xlabel("Simulation step")
    plt.ylabel("Moving-average acceptance")
    plt.ylim(0, 1.05)  # Set explicit y-axis range for acceptance rates (0-1)
    plt.title("Moving-Average Acceptance Rate")
    plt.grid(True, alpha=0.3)  # Add light grid for better readability
    plt.legend()
    plt.tight_layout()

    # Use plot directory from config
    plot_dir = Path(cfg["output"]["plots_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plot filename with correct suffix
    out_path = plot_dir / f"moving_average_acceptance_{suffix}.png"
    plt.savefig(out_path, dpi=200)
    logger.info("Moving-average plot saved to %s", out_path)

    # ------------------------------------------------------------------
    # 3) Optional wandb logging
    # ------------------------------------------------------------------
    if _WANDB_AVAILABLE and cfg.get("wandb", False):
        try:
            if wandb.run is not None:
                wandb.log({"moving_average_acceptance": wandb.Image(str(out_path))})
                logger.debug("Logged moving-average acceptance figure to wandb.")
            else:
                logger.warning("No active wandb run found. Skipping wandb logging.")
        except Exception as e:
            logger.warning(f"Error logging to wandb: {e}")

    plt.close()
