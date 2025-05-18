"""Autocorrelation analysis of acceptance histories.

This metric computes the normalised autocorrelation function (ACF) of the
binary acceptance histories for both *Naive PT* and *Flow-based PT* samples
produced by the swap-rate evaluation.  From the ACF we also estimate the
integrated autocorrelation time (IACT) as

    IACT = 1 + 2 * Σ_{ℓ=1}^{L} ρ(ℓ)

where ρ(ℓ) is the autocorrelation at lag ℓ and *L* equals
``cfg["pt"].get("swap_interval", 100)``.

A figure with both ACF curves is saved to
``<plot_dir>/acceptance_autocorrelation.png`` and the IACTs are printed and
optionally logged to *wandb*.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Optional wandb
try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

from src.accelmd.evaluators.swap_rate import _pair_suffix


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
            f"Key {exc} missing in results JSON – cannot compute autocorrelation."
        ) from exc

    return naive_hist, flow_hist


def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute the normalised autocorrelation for lags 0..max_lag.

    The implementation uses FFT convolution for speed when *x* is long,
    but falls back to a simple direct method for short sequences.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("Empty sequence provided to _autocorrelation().")

    x_mean = x.mean()
    x_var = np.var(x)
    if x_var == 0:
        # All values identical; return rho=1 at lag0 and rho=0 elsewhere.
        rho = np.zeros(max_lag + 1)
        rho[0] = 1.0
        return rho

    # Normalise
    x = x - x_mean

    # Efficient autocorrelation via FFT – compute full then truncate
    # Padding to next power of two improves FFT performance but is optional.
    fft_len = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=fft_len)
    acf = np.fft.irfft(f * np.conjugate(f))[: n]  # length n
    acf /= x_var * np.arange(n, 0, -1)

    return acf[: max_lag + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: Dict[str, Any]) -> None:  # noqa: D401 – imperative API
    """Compute and plot the acceptance autocorrelation functions."""
    results_dir = Path(cfg["evaluator"]["results_dir"])
    t_low, t_high = float(cfg["pt"]["temp_low"]), float(cfg["pt"]["temp_high"])
    naive_hist, flow_hist = _load_histories(results_dir, t_low, t_high)

    swap_interval = int(cfg["pt"].get("swap_interval", 100))
    L = swap_interval  # Max lag for ACF & IACT
    logger.debug("Computing ACF up to lag L=%d", L)

    na_arr = np.asarray(naive_hist, dtype=float)
    fl_arr = np.asarray(flow_hist, dtype=float)

    acf_naive = _autocorrelation(na_arr, L)
    acf_flow = _autocorrelation(fl_arr, L)

    iact_naive = 1.0 + 2.0 * acf_naive[1:].sum()
    iact_flow = 1.0 + 2.0 * acf_flow[1:].sum()

    # ------------------------------------------------------------------
    # 1) Print & log IACTs
    # ------------------------------------------------------------------
    print(
        "\nIntegrated Autocorrelation Time (IACT):\n"
        "Naive PT: {n:.3f}\nFlow-based PT: {f:.3f}".format(n=iact_naive, f=iact_flow)
    )
    logger.info("IACT (Naive PT) = %.3f", iact_naive)
    logger.info("IACT (Flow PT)  = %.3f", iact_flow)

    if _WANDB_AVAILABLE and cfg.get("wandb", False):
        try:
            if wandb.run is not None:
                wandb.log({
                    "iact_naive": iact_naive,
                    "iact_flow": iact_flow,
                })
                logger.debug("Logged IACT values to wandb.")
            else:
                logger.warning("No active wandb run found. Skipping IACT wandb logging.")
        except Exception as e:
            logger.warning(f"Error logging IACT to wandb: {e}")

    # ------------------------------------------------------------------
    # 2) Plot ACF curves
    # ------------------------------------------------------------------
    lags = np.arange(L + 1)
    plt.figure(figsize=(6, 4))
    plt.stem(lags, acf_naive, label="Naive PT", linefmt="tab:blue", markerfmt=" ")
    plt.stem(lags, acf_flow, label="Flow-based PT", linefmt="tab:orange", markerfmt=" ")
    plt.xlabel("Lag ℓ")
    plt.ylabel("ρ(ℓ)")
    plt.title("Acceptance Autocorrelation")
    plt.legend()
    plt.tight_layout()

    plot_dir = Path(cfg["evaluator"]["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"acceptance_autocorrelation_{_pair_suffix(t_low, t_high)}.png"
    plt.savefig(out_path, dpi=200)
    logger.info("Autocorrelation plot saved to %s", out_path)

    if _WANDB_AVAILABLE and cfg.get("wandb", False):
        try:
            if wandb.run is not None:
                wandb.log({"acceptance_autocorrelation": wandb.Image(str(out_path))})
                logger.debug("Logged autocorrelation figure to wandb.")
            else:
                logger.warning("No active wandb run found. Skipping autocorrelation plot wandb logging.")
        except Exception as e:
            logger.warning(f"Error logging autocorrelation plot to wandb: {e}")

    plt.close()
