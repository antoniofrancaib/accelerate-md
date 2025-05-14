from __future__ import annotations

# Standard library
import argparse
import json
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.accelmd.utils.config import load_config
from src.accelmd.targets.gmm import GMM
from src.accelmd.models.realnvp import create_realnvp_flow

logger = logging.getLogger(__name__)


def _build_gmm_from_config(gmm_cfg: dict, device: torch.device) -> GMM:
    """Instantiate a GMM and apply any custom parameters from *gmm_cfg*."""
    gmm = GMM(
        dim=gmm_cfg.get("dim", 2),
        n_mixes=gmm_cfg.get("n_mixes", 5),
        loc_scaling=gmm_cfg.get("loc_scaling", 1.0),
        device=device,
    )

    # Override the randomly-generated parameters with user-provided ones
    with torch.no_grad():
        if "locations" in gmm_cfg:
            gmm.locs.copy_(torch.tensor(gmm_cfg["locations"], device=device))
            logger.info("Applied custom GMM locations from config")
        if "scales" in gmm_cfg:
            gmm.scale_trils.copy_(torch.tensor(gmm_cfg["scales"], device=device))
            logger.info("Applied custom GMM scales from config")
        if "weights" in gmm_cfg:
            gmm.cat_probs.copy_(torch.tensor(gmm_cfg["weights"], device=device))
            logger.info("Applied custom GMM weights from config")
    return gmm


def _attempt_load_flow(cfg: dict, device: torch.device, gmm: GMM) -> torch.nn.Module:
    """Create a RealNVP model from *cfg* and load weights from checkpoint.

    To remain backward compatible with different naming conventions, we try a
    couple of file-name patterns when looking for the checkpoint file.
    """
    ckpt_dir = Path(cfg["evaluator"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pt_cfg = cfg["pt"]
    t_low, t_high = pt_cfg["temp_low"], pt_cfg["temp_high"]

    # Candidate 1 – spec-compliant name
    candidates: List[Path] = [
        ckpt_dir / f"flow_{t_low}_to_{t_high}.pt",
    ]
    # Candidate 2 – trainer's current naming scheme
    n_modes = gmm.locs.shape[0]
    candidates.append(ckpt_dir / f"realnvp_{n_modes}modes_best.pt")

    ckpt_path: Path | None = None
    for cand in candidates:
        if cand.is_file():
            ckpt_path = cand
            break
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Could not find a RealNVP checkpoint in {ckpt_dir}. Tried: {', '.join(str(c) for c in candidates)}"
        )
    logger.info(f"Loading RealNVP weights from {ckpt_path}")

    # Model architecture comes from trainer-section of config
    model_cfg = cfg.get("trainer", {}).get("realnvp", {}).get("model", {})
    flow = create_realnvp_flow(model_cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    # Handle checkpoints that store a dict vs plain state_dict
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    flow.load_state_dict(state_dict)
    flow.eval()
    return flow


def _compute_acceptance_naive(
    x_low: torch.Tensor,
    x_high: torch.Tensor,
    gmm: GMM,
    hi_gmm: torch.distributions.Distribution,
    t_low: float,
    t_high: float,
) -> bool:
    """Acceptance for simple swap proposal (exchange replicas)."""
    # Ensure we're working with batched inputs with correct shapes
    x_low_flat = x_low.view(-1, x_low.shape[-1])
    x_high_flat = x_high.view(-1, x_high.shape[-1])
    
    lp_low_x_low = gmm.log_prob(x_low_flat)
    lp_low_x_high = gmm.log_prob(x_high_flat)

    lp_hi_x_high = hi_gmm.log_prob(x_high_flat) / t_high
    lp_hi_x_low = hi_gmm.log_prob(x_low_flat) / t_high

    log_alpha = (lp_low_x_high + lp_hi_x_low) - (lp_low_x_low + lp_hi_x_high)
    accept = torch.rand(()) .log() < log_alpha
    return bool(accept.item())


def _compute_acceptance_flow(
    x_low: torch.Tensor,
    x_high: torch.Tensor,
    gmm: GMM,
    hi_gmm: torch.distributions.Distribution,
    flow: torch.nn.Module,
    t_high: float,
) -> bool:
    """Acceptance using RealNVP-guided extreme swap."""
    # Ensure we're working with batched inputs with correct shapes
    x_low_flat = x_low.view(-1, x_low.shape[-1])
    x_high_flat = x_high.view(-1, x_high.shape[-1])
    
    # Forward / inverse transforms
    y_high, ld_fwd = flow.forward(x_low_flat)
    y_low, ld_inv = flow.inverse(x_high_flat)

    # Target log densities
    lp_low_y_low = gmm.log_prob(y_low)
    lp_hi_y_high = hi_gmm.log_prob(y_high) / t_high

    lp_low_x_low = gmm.log_prob(x_low_flat)
    lp_hi_x_high = hi_gmm.log_prob(x_high_flat) / t_high

    log_alpha = (
        lp_low_y_low + lp_hi_y_high
        - lp_low_x_low - lp_hi_x_high
        + ld_fwd + ld_inv
    )
    accept = torch.rand(()) .log() < log_alpha
    return bool(accept.item())


def main(config_path: str):
    """Run GMM extreme-swap experiment and report acceptance rates."""
    # ------------------------------------------------------------------
    # 1) Load config & set up directories
    # ------------------------------------------------------------------
    cfg = load_config(config_path)

    # Device handling
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available – falling back to CPU")
        device = torch.device("cpu")

    pt_cfg = cfg["pt"]
    t_low, t_high = float(pt_cfg["temp_low"]), float(pt_cfg["temp_high"])
    n_steps = int(pt_cfg["num_steps"])
    swap_interval = int(pt_cfg["swap_interval"])
    n_attempts = max(1, n_steps // swap_interval)

    # Output directories
    eval_cfg = cfg["evaluator"]
    plot_dir = Path(eval_cfg["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(eval_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Build targets & (optionally) flow model
    # ------------------------------------------------------------------
    gmm = _build_gmm_from_config(cfg["gmm"], device)
    hi_gmm = gmm.tempered_version(t_high)

    # Flow model (may raise if checkpoint not found)
    flow = _attempt_load_flow(cfg, device, gmm)

    # ------------------------------------------------------------------
    # 3) Run the experiment – generate acceptance histories
    # ------------------------------------------------------------------
    naive_hist: List[int] = []
    flow_hist: List[int] = []

    for _ in range(n_attempts):
        # Draw *fresh* samples from each tempered distribution to mimic well-mixed PT chains
        # Add explicit batch dimension [1, 2] instead of just [2]
        x_low = gmm.sample((1,)).to(device)
        x_high = hi_gmm.sample((1,)).to(device)

        # --- Naive PT ---
        naive_accept = _compute_acceptance_naive(
            x_low, x_high, gmm, hi_gmm, t_low, t_high
        )
        naive_hist.append(int(naive_accept))

        # --- Flow-based PT ---
        flow_accept = _compute_acceptance_flow(
            x_low, x_high, gmm, hi_gmm, flow, t_high
        )
        flow_hist.append(int(flow_accept))

    naive_rate = float(np.mean(naive_hist))
    flow_rate = float(np.mean(flow_hist))

    # ------------------------------------------------------------------
    # 4) Pretty print table
    # ------------------------------------------------------------------
    print("""
Swap Pair   |  Naive PT  |  Flow-based T-GePT
------------|------------|-----------------
(T₁↔Tₖ)      |   {na:.2f}     |      {fa:.2f}
""".format(na=naive_rate, fa=flow_rate))

    # ------------------------------------------------------------------
    # 5) Plot acceptance histories
    # ------------------------------------------------------------------
    x_axis = np.arange(n_attempts) * swap_interval
    plt.figure(figsize=(8, 4))
    plt.step(x_axis, naive_hist, where="post", label="Naive PT", alpha=0.8)
    plt.step(x_axis, flow_hist, where="post", label="Flow-based T-GePT", alpha=0.8)
    plt.yticks([0, 1])
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Simulation step")
    plt.ylabel("Swap accepted")
    plt.title("Extreme-swap acceptance on 2-D GMM")
    plt.legend()
    fig_path = plot_dir / "gmm_swap_rate.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved acceptance plot to {fig_path}")

    # ------------------------------------------------------------------
    # 6) JSON summary
    # ------------------------------------------------------------------
    summary = {
        "naive_hist": naive_hist,
        "flow_hist": flow_hist,
        "naive_rate": naive_rate,
        "flow_rate": flow_rate,
    }
    json_path = results_dir / "gmm_swap_rate.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    logger.info(f"Wrote summary to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate extreme-swap acceptance on a 2-D GMM using Naive vs RealNVP-guided PT"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config (e.g. configs/pt/gmm.yaml)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    main(args.config)
