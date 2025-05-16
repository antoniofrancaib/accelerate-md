# src/accelmd/trainers/realnvp_trainer.py
"""
Train a bidirectional RealNVP flow that maps between two temperatures of a 2‑D GMM.
All settings come from configs/pt/gmm.yaml.
"""

from __future__ import annotations
import os, logging, json, shutil
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.accelmd.utils.config import load_config
from src.accelmd.data.gmm_dataset import TemperedGMMPairDataset
from src.accelmd.targets.gmm import GMM
from src.accelmd.models.realnvp import create_realnvp_flow

# ───────────────────────────────────────────────────
# Optional wandb
try:
    import wandb
    WANDB = True
except ImportError:  # keep code runnable without it
    WANDB = False
# ───────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ───────────────────────────────────────────────────
# Small helpers
# ───────────────────────────────────────────────────
def _init_wandb(cfg: Dict[str, Any], n_modes: int) -> None:
    """Start a wandb run if WANDB True & cfg['wandb']==True."""
    if not (WANDB and cfg.get("wandb", False)):
        return
    wandb.init(
        project="accelmd",
        name=f"realnvp_{n_modes}modes_T{cfg['pt']['temp_high']}_to_T{cfg['pt']['temp_low']}",
        config=cfg,
    )


def _scheduler(optimiser: optim.Optimizer,
               warmup_epochs: int,
               max_epochs: int,
               steps_per_epoch: int,
               base_lr: float,
               target_lr: float,
               eta_min_factor: float):
    """Two‑phase LR schedule: linear warm‑up then cosine decay."""
    warm_steps = warmup_epochs * steps_per_epoch
    total_steps = max_epochs * steps_per_epoch
    cos_steps = total_steps - warm_steps

    warm = torch.optim.lr_scheduler.LinearLR(optimiser,
                                             start_factor=base_lr / target_lr,
                                             end_factor=1.0,
                                             total_iters=warm_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=cos_steps,
        eta_min=target_lr * eta_min_factor)
    # custom step wrapper
    def step(step_idx: int):
        if step_idx < warm_steps:
            warm.step()
        else:
            cosine.step()
    return step
# ───────────────────────────────────────────────────



def train_realnvp(cfg: Dict[str, Any]) -> Path:
    """Main training routine – returns path to best checkpoint."""
    device = torch.device(cfg.get("device", "cpu"))

    # ───────────────────────────────────────────────────
    # 0. Output dirs
    # ───────────────────────────────────────────────────
    ckpt_dir = Path(cfg["trainer"]["realnvp"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(cfg["evaluator"]["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ───────────────────────────────────────────────────
    # 1. Build target GMM
    # ───────────────────────────────────────────────────
    gmm_cfg = cfg["gmm"]
    gmm = GMM(
        gmm_cfg["dim"],
        gmm_cfg["n_mixes"],
        gmm_cfg["loc_scaling"],
        device=device,
    )

    # Decide whether to use user-provided mode parameters or generate uniform ones
    custom_modes = gmm_cfg.get("custom_modes", True)

    with torch.no_grad():
        if custom_modes:
            # Fall back to previous behaviour but guard against missing keys
            if "locations" in gmm_cfg:
                gmm.locs.copy_(torch.tensor(gmm_cfg["locations"], device=device))
            if "scales" in gmm_cfg:
                gmm.scale_trils.copy_(torch.tensor(gmm_cfg["scales"], device=device))
            if "weights" in gmm_cfg:
                gmm.cat_probs.copy_(torch.tensor(gmm_cfg["weights"], device=device))
        else:
            # ------------------------------------------------------------------
            # Uniform (evenly-spaced) mode generation on a circle in 2-D
            # ------------------------------------------------------------------
            if gmm_cfg["dim"] != 2:
                raise ValueError("Uniform mode generation is currently implemented for dim==2 only.")

            n = gmm_cfg["n_mixes"]
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
                    rows = int(gmm_cfg["grid_rows"])
                    cols = int(gmm_cfg["grid_cols"])
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
            scale_tril = torch.diag(torch.full((gmm_cfg["dim"],), scale_val, device=device))
            gmm.scale_trils.copy_(torch.stack([scale_tril] * n))

            # Uniform mixture weights
            gmm.cat_probs.copy_(torch.full((n,), 1.0 / n, device=device))

    # Temperatures
    t_low = float(cfg["pt"]["temp_low"])
    t_high = float(cfg["pt"]["temp_high"])
    hi_gmm = gmm.tempered_version(t_high, scaling_method="sqrt")

    # ───────────────────────────────────────────────────
    # 2. Dataset
    # ───────────────────────────────────────────────────
    tr_cfg = cfg["trainer"]["realnvp"]["training"]
    n_samples = tr_cfg.get("n_samples", 50_000)
    dataset = TemperedGMMPairDataset(gmm,
                                     n_samples,
                                     temp_high=t_high,
                                     temp_low=t_low,
                                     temp_scaling_method="sqrt")
    val_split = tr_cfg["val_split"]
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(tr_cfg["seed"]))

    loader = DataLoader(train_set,
                        batch_size=tr_cfg["batch_size"],
                        shuffle=True)
    vloader = DataLoader(val_set,
                         batch_size=tr_cfg["batch_size"],
                         shuffle=False)

    # ───────────────────────────────────────────────────
    # 3. Model & optimiser
    # ───────────────────────────────────────────────────
    flow = create_realnvp_flow(cfg["trainer"]["realnvp"]["model"]).to(device)
    
    # Initialize weights with smaller values to improve numerical stability
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            # Use a smaller initialization scale
            torch.nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            
    flow.apply(init_weights)
    logger.info("Model initialized with stabilized weight initialization")

    # Ensure numeric values are converted to appropriate types
    learning_rate = float(tr_cfg["learning_rate"])
    initial_lr = float(tr_cfg.get("initial_lr", learning_rate * 0.3))
    warmup_epochs = int(tr_cfg["warmup_epochs"])
    max_epochs = int(tr_cfg["n_epochs"])
    eta_min_factor = float(tr_cfg["eta_min_factor"])
    max_grad_norm = float(tr_cfg["max_grad_norm"])
    mse_weight = float(tr_cfg.get("mse_weight", 0.05))
    
    opt = optim.Adam(flow.parameters(), lr=learning_rate)

    lr_step = _scheduler(opt,
                         warmup_epochs=warmup_epochs,
                         max_epochs=max_epochs,
                         steps_per_epoch=len(loader),
                         base_lr=initial_lr,
                         target_lr=learning_rate,
                         eta_min_factor=eta_min_factor)

    _init_wandb(cfg, gmm.locs.shape[0])

    # ───────────────────────────────────────────────────
    # 4. Training loop
    # ───────────────────────────────────────────────────
    best_loss = float("inf")
    best_state = None
    step_idx = 0
    patience_counter = 0        # ← NEW: prevents NameError
    for epoch in range(tr_cfg["n_epochs"]):
        flow.train()
        running = 0.0
        for xb_hi, xb_lo in loader:
            xb_hi, xb_lo = xb_hi.to(device), xb_lo.to(device)

            # High→Low
            y_lo, ld_inv = flow.inverse(xb_hi)
            loss_hl = -(gmm.log_prob(y_lo) + ld_inv).mean()

            # Low→High
            y_hi, ld_fwd = flow.forward(xb_lo)
            loss_lh = -(hi_gmm.log_prob(y_hi) / t_high + ld_fwd).mean()

            # small pairwise stabiliser
            mse = mse_weight * ((y_lo - xb_lo) ** 2).sum(-1).mean()

            # Check for NaN values and report which component is causing them
            if torch.isnan(loss_hl):
                gmm_log_prob = gmm.log_prob(y_lo).mean()
                logger.warning(f"NaN in loss_hl: gmm_log_prob={gmm_log_prob.item():.4f}, ld_inv={ld_inv.mean().item():.4f}")
                # Clip extremely negative values to prevent NaN
                loss_hl = -torch.clamp(gmm.log_prob(y_lo) + ld_inv, min=-1e6, max=1e6).mean()

            if torch.isnan(loss_lh):
                hi_log_prob = hi_gmm.log_prob(y_hi).mean()
                logger.warning(f"NaN in loss_lh: hi_log_prob={hi_log_prob.item():.4f}, ld_fwd={ld_fwd.mean().item():.4f}")
                # Clip extremely negative values to prevent NaN
                loss_lh = -torch.clamp(hi_gmm.log_prob(y_hi) / t_high + ld_fwd, min=-1e6, max=1e6).mean()

            if torch.isnan(mse):
                logger.warning(f"NaN in MSE stabilizer")
                mse = torch.tensor(0.0, device=device)

            loss = loss_hl + loss_lh + mse
            
            # Log individual loss components
            if WANDB and cfg.get("wandb", False):
                wandb.log({
                    "loss_hl": loss_hl.item(),
                    "loss_lh": loss_lh.item(),
                    "mse": mse.item(),
                    "total": loss.item(),
                })

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(),
                                           max_grad_norm)
            opt.step()
            lr_step(step_idx)
            step_idx += 1
            running += loss.item() * xb_hi.size(0)

        train_loss = running / len(train_set)

        # ── validation
        flow.eval()
        with torch.no_grad():
            running = 0.0
            for xb_hi, xb_lo in vloader:
                xb_hi, xb_lo = xb_hi.to(device), xb_lo.to(device)
                y_lo, ld_inv = flow.inverse(xb_hi)
                loss_hl = -(gmm.log_prob(y_lo) + ld_inv).mean()
                y_hi, ld_fwd = flow.forward(xb_lo)
                loss_lh = -(hi_gmm.log_prob(y_hi) / t_high + ld_fwd).mean()
                mse = mse_weight * ((y_lo - xb_lo) ** 2).sum(-1).mean()
                
                # Handle NaNs in validation too
                if torch.isnan(loss_hl):
                    loss_hl = torch.tensor(0.0, device=device)
                if torch.isnan(loss_lh):
                    loss_lh = torch.tensor(0.0, device=device)
                if torch.isnan(mse):
                    mse = torch.tensor(0.0, device=device)
                    
                batch_loss = loss_hl + loss_lh + mse
                if torch.isnan(batch_loss):
                    logger.warning("NaN in validation loss, skipping batch")
                    continue
                    
                running += batch_loss.item() * xb_hi.size(0)
        val_loss = running / len(val_set)

        logger.info(f"[{epoch+1}/{tr_cfg['n_epochs']}] "
                    f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = flow.state_dict()
            torch.save(best_state,
                       ckpt_dir /
                       f"flow_{t_low:.2f}_to_{t_high:.2f}.pt")

        # early stop
        if best_loss == val_loss:
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= int(tr_cfg["patience"]):
            logger.info("Early stopping.")
            break

    # Save the best model, or the last one if no good model was found
    logger.info(f"Best val loss: {best_loss:.4f}")
    checkpoint_path = ckpt_dir / f"flow_{t_low:.2f}_to_{t_high:.2f}.pt"

    if best_state is not None:
        torch.save(best_state, checkpoint_path)
    else:
        # If no good checkpoint (everything was NaN), at least save the current weights
        # so the post-training visualization can still run
        logger.warning("No valid checkpoint found - saving final model state")
        torch.save(flow.state_dict(), checkpoint_path)
    
    return checkpoint_path


# ───────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train RealNVP on GMM")
    p.add_argument("--config", default="configs/pt/gmm.yaml",
                   help="Path to YAML experiment file")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable Weights‑and‑Biases logging")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg["wandb"] = False
    ckpt_path = train_realnvp(cfg)
    print(f"✅ trained flow saved to {ckpt_path}")
