from __future__ import annotations
import copy, logging, math
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.accelmd.data.tempered_pair import TemperedPairDataset
from src.accelmd.models.tarflow import create_tarflow_flow
from src.accelmd.utils.gmm_modes import generate_gmm_modes
from src.accelmd.trainers.realnvp import _scheduler, _init_wandb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def train_tarflow(cfg: Dict[str, Any], target=None) -> Path:
    """Train a TarFlow model on the GMM temperature-transition task.

    This mirrors the RealNVP trainer but swaps in the TarFlow architecture and
    allows for independent hyper-parameter blocks under ``trainer.tarflow``.
    Returns the path to the best checkpoint saved on disk.
    """
    device = torch.device(cfg.get("device", "cpu"))
    tr_cfg = cfg["trainer"]["tarflow"]["training"]

    # 1) Output directories --------------------------------------------------
    ckpt_dir = Path(cfg["output"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2) Prepare target ----------------------------------------
    if target is None:
        from src.accelmd.targets import build_target
        target = build_target(cfg, device)

    low_tgt = target
    t_low  = float(cfg["pt"]["temp_low"])
    t_high = float(cfg["pt"]["temp_high"])
    hi_tgt = low_tgt.tempered_version(t_high)

    if hasattr(low_tgt, 'dim'):
        target_dim = low_tgt.dim
    else:
        target_dim = low_tgt.sample((1,)).shape[-1]

    # Determine n_modes for logging/wandb
    n_modes = getattr(low_tgt, 'n_mixes', target_dim)

    # 3) Dataset -------------------------------------------------------------
    ds = TemperedPairDataset(
        low_target=low_tgt,
        high_target=hi_tgt,
        n_samples=tr_cfg["n_samples"],
        noise_std=float(tr_cfg.get("noise_std", 0.0))
    )
    loader = DataLoader(
        ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # 4) Model ---------------------------------------------------------------
    model_cfg = copy.deepcopy(cfg["trainer"]["tarflow"]["model"])
    model_cfg["dim"] = target_dim
    flow = create_tarflow_flow(model_cfg).to(device)

    # Simple weight init for linear layers
    def _init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    flow.apply(_init)

    # 5) Optimiser & LR schedule -------------------------------------------
    opt = optim.Adam(flow.parameters(), lr=tr_cfg["learning_rate"])
    lr_step = _scheduler(
        opt,
        warmup_epochs=tr_cfg.get("warmup_epochs", 5),
        max_epochs=tr_cfg["n_epochs"],
        steps_per_epoch=len(loader),
        base_lr=tr_cfg.get("initial_lr", tr_cfg["learning_rate"] * 0.3),
        target_lr=tr_cfg["learning_rate"],
        eta_min_factor=tr_cfg.get("eta_min_factor", 0.05),
    )

    _init_wandb(cfg, n_modes)

    # 6) Training loop -------------------------------------------------------
    best_loss = math.inf
    best_state = None
    step_idx = 0
    for epoch in range(tr_cfg["n_epochs"]):
        flow.train()
        running = 0.0
        for xb_hi, xb_lo in loader:
            xb_hi, xb_lo = xb_hi.to(device), xb_lo.to(device)

            # Optional Gaussian noise augmentation on low-temperature samples
            noise_std = float(tr_cfg.get("noise_std", 0.0))
            if noise_std > 0.0:
                xb_lo = xb_lo + torch.randn_like(xb_lo) * noise_std

            opt.zero_grad()

            # High → Low (inverse)
            y_lo, ld_inv = flow.inverse(xb_hi)
            loss_hl = -(low_tgt.log_prob(y_lo) + ld_inv).mean()

            # Low → High (forward)
            y_hi, ld_fwd = flow.forward(xb_lo)
            loss_lh = -(hi_tgt.log_prob(y_hi) / t_high + ld_fwd).mean()

            loss = loss_hl + loss_lh

            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), tr_cfg.get("max_grad_norm", 1.0))
            opt.step()
            lr_step(step_idx)
            step_idx += 1
            running += loss.item() * xb_lo.size(0)

        epoch_loss = running / len(ds)
        logger.info(f"[TarFlow][{epoch+1}/{tr_cfg['n_epochs']}] loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(flow.state_dict())

    # 7) Save best checkpoint ----------------------------------------------
    ckpt_path = ckpt_dir / f"tarflow_{t_low}_to_{t_high}.pt"
    torch.save(best_state, ckpt_path)
    logger.info("Saved best TarFlow model → %s", ckpt_path)
    return ckpt_path 