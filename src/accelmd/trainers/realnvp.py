# src/accelmd/trainers/realnvp_trainer.py
"""
Train a bidirectional RealNVP flow that maps between two temperatures of a 2‑D GMM.
All settings come from configs/pt/gmm.yaml.
"""

from __future__ import annotations
import os, logging, json, shutil, copy
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.accelmd.utils.config import load_config
from src.accelmd.data.tempered_pair import TemperedPairDataset
from src.accelmd.models.realnvp import create_realnvp_flow
from src.accelmd.utils.gmm_modes import generate_gmm_modes

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



def train_realnvp(cfg: Dict[str, Any], target=None) -> Path:
    """Main training routine – returns path to best checkpoint."""
    logger.info("Starting RealNVP training for target type=%s", cfg['target']['type'])
    device = torch.device(cfg.get("device", "cpu"))
    logger.info("Using device: %s", device)

    # ───────────────────────────────────────────────────
    # 0. Output dirs
    # ───────────────────────────────────────────────────
    ckpt_dir = Path(cfg["output"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Use standardized paths from output config
    plot_dir = Path(cfg["output"]["plots_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directories")

    # ───────────────────────────────────────────────────
    # 1. Prepare target distributions
    # ───────────────────────────────────────────────────
    if target is None:
        from src.accelmd.targets import build_target
        base_tgt = build_target(cfg, torch.device(cfg.get("device", "cpu")))
    else:
        base_tgt = target

    t_low = float(cfg["pt"]["temp_low"])
    t_high = float(cfg["pt"]["temp_high"])

    low_tgt = base_tgt.tempered_version(t_low)
    hi_tgt  = base_tgt.tempered_version(t_high)

    # If the provided target has an attribute 'dim', use it, otherwise infer from a sample
    if hasattr(low_tgt, 'dim'):
        target_dim = low_tgt.dim
    else:
        target_dim = low_tgt.sample((1,)).shape[-1]

    # Determine n_modes (for wandb naming if applicable)
    n_modes = getattr(low_tgt, 'n_mixes', target_dim)

    # ───────────────────────────────────────────────────
    # 2. Dataset
    # ───────────────────────────────────────────────────
    logger.info("Creating dataset...")
    tr_cfg = cfg["trainer"]["realnvp"]["training"]
    n_samples = tr_cfg.get("n_samples", 50_000)
    logger.info("Generating %d data samples", n_samples)
    dataset = TemperedPairDataset(
        low_target=low_tgt,
        high_target=hi_tgt,
        n_samples=n_samples,
        noise_std=float(tr_cfg.get("noise_std", 0.0))
    )
    val_split = tr_cfg.get("val_split", 0.1)  # Default to 10% validation split if not specified
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
    logger.info("Dataset created with %d training and %d validation samples", len(train_set), len(val_set))

    # ───────────────────────────────────────────────────
    # 3. Model & optimiser - ensure dimension compatibility
    # ───────────────────────────────────────────────────
    logger.info("Creating RealNVP model...")
    model_cfg = copy.deepcopy(cfg["trainer"]["realnvp"]["model"])
    model_cfg["dim"] = target_dim  # Ensure the flow model uses the same dimension as GMM
    flow = create_realnvp_flow(model_cfg).to(device)
    logger.info("Created RealNVP flow with %d couplings, dim=%d, hidden_dim=%d", 
                model_cfg['n_couplings'], model_cfg['dim'], model_cfg['hidden_dim'])
    
    # Initialize weights with smaller values to improve numerical stability
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            # Skip final layers in each coupling block which are already zero-initialized
            # Check if this is one of the final layers in a coupling block
            is_final_layer = False
            for coupling in flow.couplings:
                if m is coupling.s_net[-1] or m is coupling.t_net[-1]:
                    is_final_layer = True
                    break
                
            # Only initialize non-final layers
            if not is_final_layer:
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
    logger.info("Optimizer and scheduler created")

    _init_wandb(cfg, n_modes)

    # ───────────────────────────────────────────────────
    # 4. Training loop
    # ───────────────────────────────────────────────────
    logger.info("Starting training loop: %d epochs", max_epochs)
    best_loss = float("inf")
    best_state = None
    step_idx = 0
    patience_counter = 0        # ← NEW: prevents NameError
    for epoch in range(tr_cfg["n_epochs"]):
        flow.train()
        running = 0.0
        logger.info("Starting epoch %d/%d", epoch+1, max_epochs)
        
        batch_count = 0
        for xb_hi, xb_lo in loader:
            xb_hi, xb_lo = xb_hi.to(device), xb_lo.to(device)
            batch_count += 1
            if batch_count % 10 == 0:
                logger.info("Processing batch %d/%d", batch_count, len(loader))

            # High→Low
            y_lo, ld_inv = flow.inverse(xb_hi)
            loss_hl = -(low_tgt.log_prob(y_lo) + ld_inv).mean()

            # Low→High
            y_hi, ld_fwd = flow.forward(xb_lo)
            loss_lh = -(hi_tgt.log_prob(y_hi) + ld_fwd).mean()

            # small pairwise stabiliser
            mse = mse_weight * ((y_lo - xb_lo) ** 2).sum(-1).mean()

            # Check for NaN values and report which component is causing them
            if torch.isnan(loss_hl):
                low_log_prob = low_tgt.log_prob(y_lo).mean()
                logger.warning("NaN in loss_hl: low_log_prob=%f, ld_inv=%f", low_log_prob.item(), ld_inv.mean().item())
                # Clip extremely negative values to prevent NaN
                loss_hl = -torch.clamp(low_tgt.log_prob(y_lo) + ld_inv, min=-1e6, max=1e6).mean()

            if torch.isnan(loss_lh):
                hi_log_prob = hi_tgt.log_prob(y_hi).mean()
                logger.warning("NaN in loss_lh: hi_log_prob=%f, ld_fwd=%f", hi_log_prob.item(), ld_fwd.mean().item())
                # Clip extremely negative values to prevent NaN
                loss_lh = -torch.clamp(hi_tgt.log_prob(y_hi) + ld_fwd, min=-1e6, max=1e6).mean()

            if torch.isnan(mse):
                logger.warning("NaN in MSE stabilizer")
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
        logger.info("Running validation for epoch %d", epoch+1)
        flow.eval()
        with torch.no_grad():
            running = 0.0
            for xb_hi, xb_lo in vloader:
                xb_hi, xb_lo = xb_hi.to(device), xb_lo.to(device)
                y_lo, ld_inv = flow.inverse(xb_hi)
                loss_hl = -(low_tgt.log_prob(y_lo) + ld_inv).mean()
                y_hi, ld_fwd = flow.forward(xb_lo)
                loss_lh = -(hi_tgt.log_prob(y_hi) + ld_fwd).mean()
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

        logger.info("[%d/%d] train=%f  val=%f", epoch+1, tr_cfg['n_epochs'], train_loss, val_loss)
        logger.info("Epoch %d: train=%f, val=%f", epoch+1, train_loss, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = flow.state_dict()
            # Use standardized path from config
            checkpoint_path = Path(cfg["output"]["model_path"])
            torch.save(best_state, checkpoint_path)
            logger.info("New best model saved to %s", checkpoint_path)

        # early stop
        if best_loss == val_loss:
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= int(tr_cfg["patience"]):
            logger.info("Early stopping.")
            logger.info("Early stopping triggered")
            break

    # Save the best model, or the last one if no good model was found
    logger.info("Best val loss: %f", best_loss)
    logger.info("Training completed, best val loss: %f", best_loss)

    if best_state is not None:
        torch.save(best_state, checkpoint_path)
    else:
        # If no good checkpoint (everything was NaN), at least save the current weights
        # so the post-training visualization can still run
        logger.warning("No valid checkpoint found - saving final model state")
        torch.save(flow.state_dict(), checkpoint_path)
    
    logger.info("Final model saved to %s", checkpoint_path)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = train_realnvp(cfg)
    logger.info("✅ trained flow saved to %s", ckpt_path)
    