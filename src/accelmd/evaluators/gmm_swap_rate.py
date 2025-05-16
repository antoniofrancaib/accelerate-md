from __future__ import annotations

# Standard library
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

from src.accelmd.utils.config import load_config
from src.accelmd.targets.gmm import GMM
from src.accelmd.models.realnvp import create_realnvp_flow

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

logger = logging.getLogger(__name__)


def _init_wandb(cfg: Dict[str, Any]) -> None:
    """Initialize wandb if enabled in config."""
    if not (WANDB_AVAILABLE and cfg.get("wandb", False)):
        return
    
    # Check if an existing run is active (can happen in a loop)
    if WANDB_AVAILABLE and wandb.run is not None:
        # If we're in a run already (e.g., from previous pairs), finish it
        try:
            wandb.finish()
        except Exception as e:
            logger.warning(f"Error finishing previous wandb run: {e}")
    
    # Get GMM config for logging
    gmm_cfg = cfg.get("gmm", {})
    pt_cfg = cfg.get("pt", {})
    # Determine number of modes robustly even when custom locations are omitted
    if gmm_cfg.get("custom_modes", True) and "locations" in gmm_cfg:
        n_modes = len(gmm_cfg["locations"])
    else:
        n_modes = int(gmm_cfg.get("n_mixes", 0))
    t_low = float(pt_cfg.get("temp_low", 1.0))
    t_high = float(pt_cfg.get("temp_high", 10.0))
    
    # Initialize wandb run
    wandb.init(
        project="accelmd",
        name=f"gmm_eval_{n_modes}modes_T{t_high}_to_T{t_low}",
        config=cfg,
        # Resume if already running from training phase
        resume="allow"
    )


def _build_gmm_from_config(gmm_cfg: dict, device: torch.device) -> GMM:
    """Instantiate a GMM and apply any custom parameters from *gmm_cfg*."""
    gmm = GMM(
        dim=gmm_cfg.get("dim", 2),
        n_mixes=gmm_cfg.get("n_mixes", 5),
        loc_scaling=gmm_cfg.get("loc_scaling", 1.0),
        device=device,
    )

    custom_modes = gmm_cfg.get("custom_modes", True)

    with torch.no_grad():
        if custom_modes:
            # Override the randomly-generated parameters with user-provided ones (if present)
            if "locations" in gmm_cfg:
                gmm.locs.copy_(torch.tensor(gmm_cfg["locations"], device=device))
                logger.info("Applied custom GMM locations from config")
            if "scales" in gmm_cfg:
                gmm.scale_trils.copy_(torch.tensor(gmm_cfg["scales"], device=device))
                logger.info("Applied custom GMM scales from config")
            if "weights" in gmm_cfg:
                gmm.cat_probs.copy_(torch.tensor(gmm_cfg["weights"], device=device))
                logger.info("Applied custom GMM weights from config")
        else:
            # Generate evenly-spaced modes on a circle (2-D specific)
            if gmm_cfg.get("dim", 2) != 2:
                raise ValueError("Uniform mode generation is currently implemented for dim==2 only.")

            n = gmm_cfg.get("n_mixes", 5)
            scale_val = float(gmm_cfg.get("uniform_mode_scale", 0.25))
            
            # Determine mode arrangement
            mode_arrangement = gmm_cfg.get("mode_arrangement", "circle")
            
            if mode_arrangement == "circle":
                # Place modes evenly around a circle
                radius = float(gmm_cfg.get("uniform_mode_radius", 3.0))
                angles = torch.linspace(0, 2 * torch.pi, n + 1, device=device)[:-1]
                locs = torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1)
                
            elif mode_arrangement == "grid":
                # Place modes in a grid pattern
                grid_x_range = gmm_cfg.get("grid_x_range", [-4.0, 4.0])
                grid_y_range = gmm_cfg.get("grid_y_range", [-4.0, 4.0])
                
                # Use specified grid dimensions or calculate them automatically
                if "grid_rows" in gmm_cfg and "grid_cols" in gmm_cfg:
                    rows = int(gmm_cfg.get("grid_rows"))
                    cols = int(gmm_cfg.get("grid_cols"))
                else:
                    # Automatically determine grid dimensions to be approximately square
                    rows = int(np.ceil(np.sqrt(n)))
                    cols = int(np.ceil(n / rows))
                
                # Generate grid positions
                x_points = torch.linspace(grid_x_range[0], grid_x_range[1], cols, device=device)
                y_points = torch.linspace(grid_y_range[0], grid_y_range[1], rows, device=device)
                
                # Create a mesh grid
                grid_x, grid_y = torch.meshgrid(x_points, y_points, indexing='ij')
                grid_positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
                
                # If we have more grid positions than needed, take only the first n
                if grid_positions.shape[0] > n:
                    locs = grid_positions[:n]
                else:
                    # If we need more positions than the grid provides, duplicate some
                    # This shouldn't normally happen if grid dimensions are specified correctly
                    logger.warning(f"Grid dimensions ({rows}x{cols}={rows*cols}) don't provide enough positions for {n} modes. Some modes will overlap.")
                    repeats_needed = int(np.ceil(n / grid_positions.shape[0]))
                    repeated_positions = grid_positions.repeat(repeats_needed, 1)
                    locs = repeated_positions[:n]
            else:
                raise ValueError(f"Unknown mode_arrangement: '{mode_arrangement}'. Use 'circle' or 'grid'.")
                
            gmm.locs.copy_(locs)
            
            # Isotropic covariance with specified scale
            scale_tril = torch.diag(torch.full((2,), scale_val, device=device))
            gmm.scale_trils.copy_(torch.stack([scale_tril] * n))
            
            # Uniform mixture weights
            gmm.cat_probs.copy_(torch.full((n,), 1.0 / n, device=device))

    return gmm


def _attempt_load_flow(cfg: dict, device: torch.device, gmm: GMM) -> torch.nn.Module:
    """Create a RealNVP model from *cfg* and load weights from checkpoint.

    To remain backward compatible with different naming conventions, we try a
    couple of file-name patterns when looking for the checkpoint file.
    """
    ckpt_dir = Path(cfg["evaluator"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pt_cfg = cfg["pt"]
    t_low, t_high = float(pt_cfg["temp_low"]), float(pt_cfg["temp_high"])
    
    # Round to 2 decimals for file matching
    t_low_rounded = round(t_low, 2)
    t_high_rounded = round(t_high, 2)

    # List possible candidates from most specific to least specific
    candidates: List[Path] = []
    
    # Candidate 1: Exact 2 decimal format (new format)
    candidates.append(ckpt_dir / f"flow_{t_low_rounded:.2f}_to_{t_high_rounded:.2f}.pt")
    
    # Candidate 2: Try just the rounded values without format specifier
    candidates.append(ckpt_dir / f"flow_{t_low_rounded}_to_{t_high_rounded}.pt")
    
    # Candidate 3: Historical format with full precision
    candidates.append(ckpt_dir / f"flow_{t_low}_to_{t_high}.pt")
    
    # Candidate 4: Generic fallback
    n_modes = gmm.locs.shape[0]
    candidates.append(ckpt_dir / f"realnvp_{n_modes}modes_best.pt")
    
    # Try direct matches first
    ckpt_path: Path | None = None
    for cand in candidates:
        if cand.is_file():
            ckpt_path = cand
            logger.info(f"Found exact checkpoint match: {cand}")
            break
            
    # If no exact match, try glob patterns
    if ckpt_path is None:
        # Try search by pattern - look for any checkpoints with approximately the right temps
        # Truncate to one decimal to widen the search
        t_low_trunc = int(t_low_rounded * 10) / 10
        t_high_trunc = int(t_high_rounded * 10) / 10
        
        patterns = [
            f"flow_{t_low_trunc}*_to_{t_high_trunc}*.pt",  # e.g. flow_1.0*_to_1.6*.pt
            f"flow_{t_low_rounded}_to_{t_high_rounded}.pt", # Exact rounded match
            "flow_*_to_*.pt"  # Any flow file as a last resort
        ]
        
        logger.info(f"No exact match found. Trying pattern search for t_low={t_low_rounded}, t_high={t_high_rounded}")
        
        for pattern in patterns:
            matches = list(ckpt_dir.glob(pattern))
            if matches:
                # Sort by specificity - shorter filenames are preferred as they likely have fewer decimal places
                matches.sort(key=lambda p: len(p.name))
                ckpt_path = matches[0]
                logger.info(f"Found pattern match {pattern}: {ckpt_path}")
                break

    if ckpt_path is None:
        raise FileNotFoundError(
            f"Could not find a RealNVP checkpoint in {ckpt_dir}. Tried patterns including: "
            f"flow_{t_low_rounded:.2f}_to_{t_high_rounded:.2f}.pt and broader patterns."
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


# ──────────────────────────────────────────────────────────────────
# Helper to format temperatures for filenames
# ──────────────────────────────────────────────────────────────────

def _fmt_temp(t: float) -> str:
    """Format temperature *t* for use in file names (strip trailing zeros)."""
    s = f"{t:.8g}"  # 8 significant digits is plenty
    return s.replace('.', 'p')  # replace dot with 'p' to avoid filesystem issues


# ------------------------------------------------------------------
# New: unified helper to build identifier matching checkpoint names
# ------------------------------------------------------------------


def _pair_suffix(t_low: float, t_high: float) -> str:
    """Return filename-safe suffix *including* the ``flow_`` prefix.

    Examples
    --------
    >>> _pair_suffix(1.0, 1.6681005)
    'flow_1.00_to_1.67'

    The raw ``str`` representation is used to mirror exactly the checkpoint
    names created by ``train_realnvp`` such that post-processing outputs can
    simply prepend their own descriptor (e.g. ``moving_average_acceptance``)
    while sharing the same pair identifier.
    """
    return f"flow_{t_low:.2f}_to_{t_high:.2f}"


# ──────────────────────────────────────────────────────────────────
# Public *run* API (dict-based) – preferred for programmatic use
# ──────────────────────────────────────────────────────────────────


def run(cfg: dict) -> None:  # noqa: D401 – imperative API
    """Run GMM extreme-swap experiment using an in-memory *cfg* dictionary.

    This is the function to call from other Python modules. The old
    ``main(config_path)`` remains for CLI compatibility and simply calls this
    helper after loading the YAML file.
    """

    # ------------------------------------------------------------------
    # 1) Load config & set up directories (cfg already loaded by caller)
    # ------------------------------------------------------------------
    # cfg is already provided

    # Initialize wandb if enabled
    _init_wandb(cfg)

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

    # Log to wandb
    if WANDB_AVAILABLE and cfg.get("wandb", False):
        wandb.log({
            "naive_acceptance_rate": naive_rate,
            "flow_acceptance_rate": flow_rate,
            "temperature_ratio": t_high / t_low,
            "num_attempts": n_attempts,
        })

    # ------------------------------------------------------------------
    # 4) Pretty print table
    # ------------------------------------------------------------------
    print(
        """
Swap Pair   |  Naive PT  |  Flow-based T-GePT
+------------|------------|-----------------
+(T₁↔Tₖ)      |   {na:.2f}     |      {fa:.2f}
""".format(na=naive_rate, fa=flow_rate)
    )

    # ------------------------------------------------------------------
    # 5) JSON summary – pair-specific filename
    # ------------------------------------------------------------------
    summary = {
        "naive_rate": naive_rate,
        "flow_rate": flow_rate,
        "naive_hist": naive_hist,
        "flow_hist": flow_hist,
    }

    # Use pair-specific identifier to avoid overwriting files when looping
    # over multiple temperature pairs.
    json_name = f"gmm_swap_rate_{_pair_suffix(t_low, t_high)}.json"
    json_path = results_dir / json_name
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    logger.info("Wrote summary to %s", json_path)

    # Log a table to wandb
    if WANDB_AVAILABLE and cfg.get("wandb", False):
        data = [[i, naive_hist[i], flow_hist[i]] for i in range(n_attempts)]
        table = wandb.Table(columns=["Step", "Naive PT", "Flow-based PT"], data=data)
        wandb.log({"swap_acceptance_history": table})


# ──────────────────────────────────────────────────────────────────
# Backwards-compatible *main* (path-based) – for CLI usage
# ──────────────────────────────────────────────────────────────────


def main(config_path: str):
    """Run GMM extreme-swap experiment and report acceptance rates."""
    cfg = load_config(config_path)
    run(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate extreme-swap acceptance on a 2-D GMM using Naive vs RealNVP-guided PT"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config (e.g. configs/pt/gmm.yaml)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging even if available")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Load config and potentially override wandb option
    cfg = load_config(args.config)
    if args.wandb:
        cfg["wandb"] = True
    if args.no_wandb:
        cfg["wandb"] = False

    main(args.config)
