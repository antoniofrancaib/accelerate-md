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

from src.accelmd.evaluators.swap_rate import _pair_suffix

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_histories(results_dir: Path, t_low: float, t_high: float) -> tuple[List[int], List[int]]:
    """Load histories for a specific temperature pair."""
    json_name = f"gmm_swap_rate_{_pair_suffix(t_low, t_high)}.json"
    json_path = results_dir / json_name
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
    results_dir = Path(cfg["evaluator"]["results_dir"])
    t_low, t_high = float(cfg["pt"]["temp_low"]), float(cfg["pt"]["temp_high"])
    naive_hist, flow_hist = _load_histories(results_dir, t_low, t_high)

    swap_interval = int(cfg["pt"].get("swap_interval", 100))
    # Calculate window size with a sensible maximum (not more than 25% of data length)
    max_window = min(len(naive_hist), len(flow_hist)) // 4
    window = min(swap_interval * 10, max_window)
    if window < 2:
        window = 2  # Ensure minimum window size of 2
    logger.debug("Using moving-average window = %d", window)

    na_arr = np.asarray(naive_hist, dtype=float)
    fl_arr = np.asarray(flow_hist, dtype=float)

    if len(na_arr) < window or len(fl_arr) < window:
        raise ValueError(
            "Histories shorter than the requested moving-average window."
        )

    ma_naive = _moving_average(na_arr, window)
    ma_flow = _moving_average(fl_arr, window)

    # X-axis: simulation step (centre of the window)
    steps = np.arange(len(ma_naive)) * swap_interval + window * 0.5

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

    plot_dir = Path(cfg["evaluator"]["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"moving_average_acceptance_{_pair_suffix(t_low, t_high)}.png"
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
