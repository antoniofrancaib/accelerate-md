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
    gmm = GMM(gmm_cfg["dim"],
              gmm_cfg["n_mixes"],
              gmm_cfg["loc_scaling"],
              device=device)
    with torch.no_grad():
        gmm.locs.copy_(torch.tensor(gmm_cfg["locations"], device=device))
        gmm.scale_trils.copy_(torch.tensor(gmm_cfg["scales"], device=device))
        gmm.cat_probs.copy_(torch.tensor(gmm_cfg["weights"], device=device))

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
    
    # Ensure numeric values are converted to appropriate types
    learning_rate = float(tr_cfg["learning_rate"])
    initial_lr = float(tr_cfg.get("initial_lr", learning_rate * 0.3))
    warmup_epochs = int(tr_cfg["warmup_epochs"])
    max_epochs = int(tr_cfg["n_epochs"])
    eta_min_factor = float(tr_cfg["eta_min_factor"])
    max_grad_norm = float(tr_cfg["max_grad_norm"])
    
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
            mse = 0.05 * ((y_lo - xb_lo) ** 2).sum(-1).mean()

            loss = loss_hl + loss_lh + mse

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
                mse = 0.05 * ((y_lo - xb_lo) ** 2).sum(-1).mean()
                running += (loss_hl + loss_lh + mse).item() * xb_hi.size(0)
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

    logger.info(f"Best val loss: {best_loss:.4f}")
    return ckpt_dir / f"flow_{t_low:.2f}_to_{t_high:.2f}.pt"


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
