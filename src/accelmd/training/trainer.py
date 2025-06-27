from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from .losses import bidirectional_nll
from .acceptance_loss import acceptance_loss
from src.accelmd.utils.config import load_config, setup_device, get_temperature_pairs, create_run_config, get_energy_threshold

__all__ = ["PTSwapTrainer"]


class PTSwapTrainer:
    """Simple trainer for PT swap flows (v0)."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        temp_pair: Tuple[int, int],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.temp_pair = temp_pair
        self.device = torch.device(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["training"]["learning_rate"]
        )
        self.clip_grad_norm = config["training"].get("clip_grad_norm", 1.0)
        self.energy_threshold = get_energy_threshold(config)

        # ------------------------------------------------------------------
        # Loss-component weights
        # ------------------------------------------------------------------
        self.recon_weight = config["training"].get("recon_weight", 1.0)
        self.recon_weight_end = config["training"].get("recon_weight_end", 0.0)

        # Acceptance-loss weight schedule (λ_acc)
        self.acc_weight_start = config["training"].get("acc_weight_start", 0.0)
        self.acc_weight_end = config["training"].get("acc_weight_end", 1.0)

        # Annealed NLL weight:  we allow either a fixed `nll_weight` (legacy)
        # or a Timewarp-style geometric warm-up controlled by three knobs.  If
        # the warm-up keys are absent we fall back to the fixed weight so
        # existing configs keep working unchanged.

        t_cfg = config["training"]
        self.nll_weight_start = t_cfg.get("nll_weight_start", t_cfg.get("nll_weight", 0.0))
        self.nll_weight_end = t_cfg.get("nll_weight_end", self.nll_weight_start)
        self.nll_warmup_epochs = t_cfg.get("nll_warmup_epochs", 0)

        # Number of epochs over which we interpolate weights
        self.warmup_epochs = t_cfg.get("warmup_epochs", 0)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config["training"].get("lr_factor", 0.5),
            patience=config["training"].get("lr_patience", 3),
            threshold=5e-3,  # 0.5 % relative improvement
            verbose=True,
            min_lr=config["training"].get("min_lr", 5e-5),
        )
        self.num_epochs = config["training"]["num_epochs"]
        self.early_patience = config["training"].get("early_stopping_patience", 10)

        # Output directory for this pair
        self.output_dir = Path(config["output"]["pair_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)

        self.best_metric = float("inf")
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------
    def train(self) -> Dict:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        start_time = time.time()
        train_hist = []
        val_hist = []
        clip_train_hist = []  # fraction of batches where loss was clipped
        clip_val_hist = []
        for epoch in range(1, self.num_epochs + 1):
            self._current_epoch = epoch
            # Compute current β (NLL weight) via geometric schedule.
            beta = self._current_nll_weight(epoch)

            train_loss, train_nll, train_acc, train_clip_frac = self._run_epoch(
                self.train_loader, training=True
            )
            val_loss, val_nll, val_acc, val_clip_frac = self._run_epoch(
                self.val_loader, training=False
            )

            print(
                f"Epoch {epoch:03d} | loss {train_loss:.3e} (nll {train_nll:.3e} | acc {train_acc:.3e}) "
                f"| val_loss {val_loss:.3e} (nll {val_nll:.3e} | acc {val_acc:.3e})"
            )

            train_hist.append(train_loss)
            val_hist.append(val_loss)
            clip_train_hist.append(train_clip_frac)
            clip_val_hist.append(val_clip_frac)

            # ----------------------------------------------------------
            # β-aware early-stopping:  before warm-up finishes we monitor the
            # total loss; afterwards we switch to the *un-weighted* NLL so
            # reconstruction-weight changes don't fool the criterion.
            if epoch >= self.warmup_epochs:
                early_metric = val_acc
            else:
                early_metric = val_loss

            # Save checkpoint whenever metric strictly improves
            if early_metric < self.best_metric:
                self.best_metric = early_metric
                self._save_checkpoint(epoch)
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # LR scheduler monitors the same metric as early stopping
            self.lr_scheduler.step(early_metric)

        hours = (time.time() - start_time) / 3600.0

        # ----------------------------------------------------------
        # Persist training curve plot
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Loss curve
        fig_path = plots_dir / "loss_curve.png"
        plt.figure(figsize=(6,4))
        epochs = range(1, len(train_hist)+1)
        plt.plot(epochs, train_hist, label="train")
        plt.plot(epochs, val_hist, label="val")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel("loss (total)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

        # Clipping-fraction curve
        clip_path = plots_dir / "clipping_fraction.png"
        plt.figure(figsize=(6,3))
        plt.plot(epochs, clip_train_hist, label="train")
        plt.plot(epochs, clip_val_hist, label="val")
        plt.xlabel("epoch")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel("fraction batches clipped")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.legend()
        plt.savefig(clip_path)
        plt.close()

        return {
            "best_metric": self.best_metric,
            "epochs_trained": epoch,
            "total_time_hours": hours,
        }

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float, float, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        nll_sum = 0.0
        acc_sum = 0.0
        log_det_stats = []  # for monitoring
        n_batches = 0
        clipped_batches = 0
        for batch in loader:
            # Move tensors to device (only coords present for now)
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # --------------------------------------------------------------
            # 1) Bidirectional NLL (always weighted 1.0 for now)
            # 2) Acceptance-loss with scheduled weight λ_acc
            # --------------------------------------------------------------

            with torch.set_grad_enabled(training):
                nll_loss = bidirectional_nll(
                    self.model,
                    batch,
                    energy_threshold=self.energy_threshold,
                )

                # Acceptance loss (Δ clamped only during very early epochs?)
                kB = 0.00831446261815324
                if "temperatures" in self.cfg:
                    temps = self.cfg["temperatures"]["values"]
                    T_low = temps[self.temp_pair[0]]
                    T_high = temps[self.temp_pair[1]]
                    beta_low = 1.0 / (kB * T_low)
                    beta_high = 1.0 / (kB * T_high)
                else:
                    beta_low = beta_high = 1.0

                acc_loss = acceptance_loss(
                    self.model,
                    batch,
                    beta_low=beta_low,
                    beta_high=beta_high,
                    clamp=True,
                    energy_threshold=self.energy_threshold,
                )

            # Current λ_acc schedule
            epoch_idx = self._current_epoch
            lambda_acc = self._interp(self.acc_weight_start, self.acc_weight_end, epoch_idx)

            loss = nll_loss + lambda_acc * acc_loss

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            batch_loss_val = loss.item()
            total_loss += batch_loss_val
            nll_sum += nll_loss.item()
            acc_sum += acc_loss.item()

            # Detect sentinel clipping constant (≈1e4) rather than any large loss
            sentinel = 1e4
            tol = 1e-3
            if (
                abs(batch_loss_val - sentinel) < tol
                or abs(nll_loss.item() - sentinel) < tol
                or abs(acc_loss.item() - sentinel) < tol
            ):
                clipped_batches += 1

            # ---------------- log-det monitoring ----------------
            with torch.no_grad():
                src = batch["source_coords"].to(self.device)
                tgt = batch["target_coords"].to(self.device)
                _, ld_f = self.model.forward(src)
                _, ld_inv = self.model.inverse(tgt)
                log_det_stats.append(ld_f.mean().item())
                log_det_stats.append(ld_inv.mean().item())

            n_batches += 1
        mean_total = total_loss / max(1, n_batches)
        mean_nll = nll_sum / max(1, n_batches)
        mean_acc = acc_sum / max(1, n_batches)

        clip_fraction = clipped_batches / max(1, n_batches)

        if training and log_det_stats:
            import numpy as np
            ld_mean = float(np.mean(log_det_stats))
            ld_std = float(np.std(log_det_stats))
            print(f"    [epoch {self._current_epoch:03d}] log|det J| mean={ld_mean:.3f} ± {ld_std:.3f}")

        return mean_total, mean_nll, mean_acc, clip_fraction

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int):
        path = self.output_dir / "models" / f"best_model_epoch{epoch}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint: {path}") 

    # ------------------------------------------------------------------
    # Helper – geometric warm-up of NLL weight
    def _current_nll_weight(self, epoch: int) -> float:
        if self.warmup_epochs == 0:
            return self.nll_weight_end  # fixed / legacy behaviour

        t = min(epoch, self.warmup_epochs)
        ratio = self.nll_weight_end / max(1e-12, self.nll_weight_start)
        beta = self.nll_weight_start * (ratio ** (t / self.warmup_epochs))
        return beta 

    # ------------------------------------------------------------------
    @property
    def _current_epoch(self):
        """Private helper set by train() loop just before _run_epoch call."""
        return getattr(self, "__cur_epoch", 0)

    @_current_epoch.setter
    def _current_epoch(self, value):
        setattr(self, "__cur_epoch", value)

    def _interp(self, start: float, end: float, epoch: int):
        if self.warmup_epochs == 0:
            return end
        t = min(epoch, self.warmup_epochs)
        return start + (end - start) * (t / self.warmup_epochs) 