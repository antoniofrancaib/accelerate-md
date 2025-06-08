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
from src.accelmd.data.pt_tempered_pair import PTTemperedPairDataset
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
    
    # Check if PT data path is provided for using equilibrated samples
    pt_data_path = cfg.get("pt_data_path")
    logger.info("PT data path from config: %s", pt_data_path)
    if pt_data_path:
        logger.info("Using equilibrated PT simulation data from: %s", pt_data_path)
        # Find the appropriate temperature pair indices
        # We need to match current t_low and t_high with PT temperature ladder
        import torch as pt_torch
        
        pt_data = pt_torch.load(pt_data_path, map_location='cpu')
        pt_temps = pt_data['temperatures'].tolist()
        
        # Find closest temperature indices
        temp_low_idx = min(range(len(pt_temps)), key=lambda i: abs(pt_temps[i] - t_low))
        temp_high_idx = min(range(len(pt_temps)), key=lambda i: abs(pt_temps[i] - t_high))
        
        # Ensure temp_high_idx > temp_low_idx (high temp has higher index in our setup)
        if temp_high_idx <= temp_low_idx:
            # Find next higher temperature
            temp_high_idx = temp_low_idx + 1
            if temp_high_idx >= len(pt_temps):
                logger.warning("Cannot find appropriate temperature pair in PT data")
                logger.warning("Available temperatures: %s", pt_temps)
                logger.warning("Requested: T_low=%f, T_high=%f", t_low, t_high)
                raise ValueError("Temperature pair not found in PT data")
        
        logger.info("Using PT temperature pair: T_low=%f (idx=%d) → T_high=%f (idx=%d)", 
                   pt_temps[temp_low_idx], temp_low_idx, pt_temps[temp_high_idx], temp_high_idx)
        
        dataset = PTTemperedPairDataset(
            pt_data_path=pt_data_path,
            temp_low_idx=temp_low_idx,
            temp_high_idx=temp_high_idx,
            n_samples=n_samples,
            subsample_factor=tr_cfg.get("pt_subsample_factor", 1)
        )
        logger.info("Created PT dataset with %d equilibrated samples", len(dataset))
    else:
        logger.info("No PT data path provided, generating fresh samples")
        logger.info("Generating %d fresh data samples", n_samples)
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
    
    # Ultra-conservative weight initialization to prevent early NaN instability
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            # Skip final layers in each coupling block which are already zero-initialized
            # Check if this is one of the final layers in a coupling block
            is_final_layer = False
            for coupling in flow.couplings:
                if m is coupling.s_net[-1] or m is coupling.t_net[-1]:
                    is_final_layer = True
                    break
                
            # Only initialize non-final layers with ultra-small values
            if not is_final_layer:
                # Use extremely conservative initialization for numerical stability
                fan_in = m.weight.size(1)
                fan_out = m.weight.size(0)
                # Use much smaller std than Xavier to prevent early instability
                std = 0.001 / np.sqrt(fan_in + fan_out)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            
    flow.apply(init_weights)
    logger.info("Model initialized with ultra-conservative weight initialization")

    # Ensure numeric values are converted to appropriate types
    learning_rate = float(tr_cfg["learning_rate"])
    initial_lr = float(tr_cfg.get("initial_lr", learning_rate * 0.3))
    warmup_epochs = int(tr_cfg["warmup_epochs"])
    max_epochs = int(tr_cfg["n_epochs"])
    eta_min_factor = float(tr_cfg["eta_min_factor"])
    max_grad_norm = float(tr_cfg["max_grad_norm"])
    # mse_weight = float(tr_cfg.get("mse_weight", 0.05))  # Removed MSE term
    
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
    patience_counter = 0
    
    # Track training history for plotting
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": []
    }
    
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

            # High→Low (inverse mapping from high-T to low-T)
            y_lo, ld_inv = flow.inverse(xb_hi)
            
            # ENHANCED NUMERICAL STABILITY CHECKS
            # Check for NaN/inf in flow outputs before computing log probs
            if torch.isnan(y_lo).any() or torch.isinf(y_lo).any():
                logger.warning("NaN/inf detected in flow inverse output, skipping batch")
                continue
            if torch.isnan(ld_inv).any() or torch.isinf(ld_inv).any():
                logger.warning("NaN/inf detected in inverse log determinant, skipping batch")
                continue
                
            # Clip extreme log determinants first
            ld_inv = torch.clamp(ld_inv, min=-500, max=500)
            
            # Compute log probabilities with strict bounds
            log_prob_lo = low_tgt.log_prob(y_lo)
            if torch.isnan(log_prob_lo).any() or torch.isinf(log_prob_lo).any():
                logger.warning("NaN/inf in log_prob_lo, skipping batch")
                continue
            log_prob_lo = torch.clamp(log_prob_lo, min=-500, max=500)
            
            # Combine terms with additional clipping
            combined_hl = log_prob_lo + ld_inv
            combined_hl = torch.clamp(combined_hl, min=-500, max=500)
            loss_hl = -combined_hl.mean()

            # Low→High (forward mapping from low-T to high-T)
            y_hi, ld_fwd = flow.forward(xb_lo)
            
            # Same stability checks for forward direction
            if torch.isnan(y_hi).any() or torch.isinf(y_hi).any():
                logger.warning("NaN/inf detected in flow forward output, skipping batch")
                continue
            if torch.isnan(ld_fwd).any() or torch.isinf(ld_fwd).any():
                logger.warning("NaN/inf detected in forward log determinant, skipping batch")
                continue
                
            # Clip extreme log determinants
            ld_fwd = torch.clamp(ld_fwd, min=-500, max=500)
            
            # Compute log probabilities with strict bounds
            log_prob_hi = hi_tgt.log_prob(y_hi)
            if torch.isnan(log_prob_hi).any() or torch.isinf(log_prob_hi).any():
                logger.warning("NaN/inf in log_prob_hi, skipping batch")
                continue
            log_prob_hi = torch.clamp(log_prob_hi, min=-500, max=500)
            
            # Combine terms with additional clipping
            combined_lh = log_prob_hi + ld_fwd
            combined_lh = torch.clamp(combined_lh, min=-500, max=500)
            loss_lh = -combined_lh.mean()

            # Final NaN check on losses before backprop
            if torch.isnan(loss_hl) or torch.isnan(loss_lh):
                logger.warning("NaN in final losses: loss_hl=%f, loss_lh=%f - skipping batch", 
                              loss_hl.item() if not torch.isnan(loss_hl) else float('nan'),
                              loss_lh.item() if not torch.isnan(loss_lh) else float('nan'))
                continue

            # Compute total loss
            loss = loss_hl + loss_lh
            
            # Additional safety check on total loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/inf in total loss, skipping batch")
                continue
            
            # Log individual loss components
            if WANDB and cfg.get("wandb", False):
                wandb.log({
                    "loss_hl": loss_hl.item(),
                    "loss_lh": loss_lh.item(),
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
            valid_batches = 0
            for xb_hi, xb_lo in vloader:
                xb_hi, xb_lo = xb_hi.to(device), xb_lo.to(device)
                
                # Apply same stability checks as in training
                y_lo, ld_inv = flow.inverse(xb_hi)
                
                # Check for NaN/inf in flow outputs
                if torch.isnan(y_lo).any() or torch.isinf(y_lo).any() or torch.isnan(ld_inv).any() or torch.isinf(ld_inv).any():
                    continue
                    
                # Clip and compute loss_hl
                ld_inv = torch.clamp(ld_inv, min=-500, max=500)
                log_prob_lo = low_tgt.log_prob(y_lo)
                if torch.isnan(log_prob_lo).any() or torch.isinf(log_prob_lo).any():
                    continue
                log_prob_lo = torch.clamp(log_prob_lo, min=-500, max=500)
                combined_hl = torch.clamp(log_prob_lo + ld_inv, min=-500, max=500)
                loss_hl = -combined_hl.mean()
                
                # Forward direction
                y_hi, ld_fwd = flow.forward(xb_lo)
                if torch.isnan(y_hi).any() or torch.isinf(y_hi).any() or torch.isnan(ld_fwd).any() or torch.isinf(ld_fwd).any():
                    continue
                    
                # Clip and compute loss_lh
                ld_fwd = torch.clamp(ld_fwd, min=-500, max=500)
                log_prob_hi = hi_tgt.log_prob(y_hi)
                if torch.isnan(log_prob_hi).any() or torch.isinf(log_prob_hi).any():
                    continue
                log_prob_hi = torch.clamp(log_prob_hi, min=-500, max=500)
                combined_lh = torch.clamp(log_prob_hi + ld_fwd, min=-500, max=500)
                loss_lh = -combined_lh.mean()
                
                # Final validation batch loss check
                batch_loss = loss_hl + loss_lh
                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    continue
                    
                running += batch_loss.item() * xb_hi.size(0)
                valid_batches += 1
        # Handle case where all validation batches were skipped due to NaN
        if valid_batches == 0:
            logger.warning("All validation batches contained NaN/inf - using large penalty")
            val_loss = 1e6  # Large penalty to trigger early stopping
        else:
            val_loss = running / (valid_batches * vloader.batch_size)
            if valid_batches < len(vloader) * 0.5:
                logger.warning("Over 50%% of validation batches skipped due to NaN/inf (%d/%d valid)", 
                              valid_batches, len(vloader))

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

        # Track training history for plotting
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["epochs"].append(epoch+1)

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

    # Save training history to JSON file
    history_path = Path(cfg["output"]["training_history"])
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    logger.info("Training history saved to %s", history_path)

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
    