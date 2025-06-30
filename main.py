#!/usr/bin/env python3
"""Command-line entry point for training PT swap flows.

Example usage:
    python main.py --config configs/aldp.yaml --temp-pair 0 1 --epochs 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple
import re
import subprocess

import torch
from torch.utils.data import DataLoader

from src.accelmd.utils.config import load_config, setup_device, get_temperature_pairs, create_run_config
from src.accelmd.data.pt_pair_dataset import PTTemperaturePairDataset
from src.accelmd.flows.pt_swap_flow import PTSwapFlow
from src.accelmd.evaluation.swap_acceptance import naive_acceptance, flow_acceptance
from src.accelmd.targets.aldp_boltzmann import AldpBoltzmann
from src.accelmd.training.trainer import PTSwapTrainer


def train_pair(cfg_path: str, pair: Tuple[int, int], epochs_override: int | None = None):
    base_cfg = load_config(cfg_path)
    if epochs_override is not None:
        base_cfg["training"]["num_epochs"] = epochs_override
    device = setup_device(base_cfg)

    from src.accelmd.utils.config import create_run_config
    cfg = create_run_config(base_cfg, pair, device)

    # Dataset & loaders
    dataset = PTTemperaturePairDataset(
        pt_data_path=cfg["data"]["pt_data_path"],
        molecular_data_path=cfg["data"]["molecular_data_path"],
        temp_pair=pair,
        subsample_rate=cfg["data"].get("subsample_rate", 100),
        device="cpu",  # keep on CPU; trainer moves to GPU if needed
    )
    val_split = 0.1
    val_size = int(len(dataset)*val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader_train = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=PTTemperaturePairDataset.collate_fn,
    )
    loader_val = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=PTTemperaturePairDataset.collate_fn,
    )

    # Model
    model_cfg = cfg["model"]
    temps = cfg["temperatures"]["values"]
    sys_cfg = cfg.get("system", {})
    energy_cut = sys_cfg.get("energy_cut")
    energy_max = sys_cfg.get("energy_max")
    model = PTSwapFlow(
        num_atoms=model_cfg["num_atoms"],
        num_layers=model_cfg["flow_layers"],
        hidden_dim=model_cfg["hidden_dim"],
        source_temperature=temps[pair[0]],
        target_temperature=temps[pair[1]],
        target_kwargs={
            "energy_cut": float(energy_cut) if energy_cut is not None else None,
            "energy_max": float(energy_max) if energy_max is not None else None,
        },
        device=device,
    )

    trainer = PTSwapTrainer(
        model=model,
        train_loader=loader_train,
        val_loader=loader_val,
        config=cfg,
        temp_pair=pair,
        device=device,
    )
    summary = trainer.train()
    print(json.dumps(summary, indent=2))

    # Return path to best checkpoint for downstream evaluation
    models_dir = Path(cfg["output"]["pair_dir"]) / "models"
    best_ckpts = list(models_dir.glob("best_model_epoch*.pt"))

    if not best_ckpts:
        return None

    # The trainer saves a checkpoint *each time* validation loss improves, so
    # the **latest modified file** corresponds to the global best model.
    best_ckpt = max(best_ckpts, key=lambda p: p.stat().st_mtime)
    return best_ckpt


def evaluate_pair(cfg_path: str, pair: Tuple[int, int], checkpoint: str, num_samples: int, save_metrics: bool = True):
    base_cfg = load_config(cfg_path)
    device = setup_device(base_cfg)

    cfg = create_run_config(base_cfg, pair, device)

    # Dataset & DataLoader (no split needed for evaluation)
    dataset = PTTemperaturePairDataset(
        pt_data_path=cfg["data"]["pt_data_path"],
        molecular_data_path=cfg["data"]["molecular_data_path"],
        temp_pair=pair,
        subsample_rate=cfg["data"].get("subsample_rate", 100),
        device="cpu",
    )
    batch_size = cfg["training"].get("batch_size", 64)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PTTemperaturePairDataset.collate_fn,
    )

    # Build flow model & load checkpoint
    model_cfg = cfg["model"]
    temps = cfg["temperatures"]["values"]
    sys_cfg = cfg.get("system", {})
    energy_cut = sys_cfg.get("energy_cut")
    energy_max = sys_cfg.get("energy_max")
    model = PTSwapFlow(
        num_atoms=model_cfg["num_atoms"],
        num_layers=model_cfg["flow_layers"],
        hidden_dim=model_cfg["hidden_dim"],
        source_temperature=temps[pair[0]],
        target_temperature=temps[pair[1]],
        target_kwargs={
            "energy_cut": float(energy_cut) if energy_cut is not None else None,
            "energy_max": float(energy_max) if energy_max is not None else None,
        },
        device=device,
    )
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    # Boltzmann bases (cpu)
    base_low = AldpBoltzmann(temperature=temps[pair[0]])
    base_high = AldpBoltzmann(temperature=temps[pair[1]])

    # Limit number of batches to cover roughly num_samples
    max_batches = (num_samples + batch_size - 1) // batch_size

    naive_acc = naive_acceptance(loader, base_low, base_high, max_batches=max_batches)
    flow_acc = flow_acceptance(loader, model, base_low, base_high, device=device, max_batches=max_batches)

    print("\nSwap acceptance estimate (pair %s ↔ %s, %d samples)" % (pair[0], pair[1], num_samples))
    print("    naïve swap : %.4f" % naive_acc)
    print("    flow swap  : %.4f" % flow_acc)

    if save_metrics:
        metrics_dir = Path(cfg["output"]["pair_dir"]) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_path = metrics_dir / "swap_acceptance.json"
        with open(out_path, "w") as fh:
            json.dump({
                "naive_acceptance": naive_acc,
                "flow_acceptance": flow_acc,
                "num_samples": num_samples
            }, fh, indent=2)
        print(f"Metrics saved to {out_path}")


def _generate_rama_plot(cfg_path: str, pair: Tuple[int,int], checkpoint: str):
    """Generate 2×2 Ramachandran grid for the given pair using the helper script.

    The plot is saved under <pair_dir>/plots/rama_grid.png where *pair_dir* is
    dictated by `create_run_config` in utils.config.
    """
    base_cfg = load_config(cfg_path)
    run_cfg = create_run_config(base_cfg, pair, device="cpu")
    plots_dir = Path(run_cfg["output"]["pair_dir"]).expanduser() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "rama_grid.png"

    cmd = [
        "python",
        "scripts/plot_rama_grid.py",
        "--config", cfg_path,
        "--checkpoint", str(checkpoint),
        "--temp-pair", str(pair[0]), str(pair[1]),
        "--n-samples", "20000",
        "--out", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Ramachandran plot generation failed for pair {pair}: {e}")
    else:
        print(f"Ramachandran plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aldp.yaml")
    parser.add_argument("--temp-pair", nargs=2, type=int, help="Indices of temperature pair to train")
    parser.add_argument("--epochs", type=int, help="Override epochs for quick tests")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate swap acceptance instead of training")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for evaluation")
    parser.add_argument("--num-eval-samples", type=int, default=20000, help="Number of evaluation samples")
    args = parser.parse_args()

    if args.evaluate:
        if not args.checkpoint or not args.temp_pair:
            raise ValueError("--checkpoint and --temp-pair are required with --evaluate")
        pair = tuple(args.temp_pair)
        evaluate_pair(args.config, pair, args.checkpoint, num_samples=args.num_eval_samples, save_metrics=False)
        _generate_rama_plot(args.config, pair, args.checkpoint)
        return

    # ------------------------------------------------------------------
    # Copy the YAML config once per invocation into the experiment outputs
    # directory for provenance.
    base_cfg = load_config(args.config)
    base_dir = Path(base_cfg["output"]["base_dir"]).expanduser()
    exp_dir = base_dir / base_cfg["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg_copy_path = exp_dir / "used_config.yaml"
    if not cfg_copy_path.exists():
        import shutil
        shutil.copy(args.config, cfg_copy_path)

    if args.temp_pair:
        pair = tuple(args.temp_pair)
        ckpt_path = train_pair(args.config, pair, epochs_override=args.epochs)
        if ckpt_path is not None:
            evaluate_pair(args.config, tuple(pair), str(ckpt_path), num_samples=20000)
            _generate_rama_plot(args.config, tuple(pair), str(ckpt_path))
    else:
        cfg = load_config(args.config)
        for pair in get_temperature_pairs(cfg):
            ckpt_path = train_pair(args.config, tuple(pair), epochs_override=args.epochs)
            if ckpt_path is not None:
                evaluate_pair(args.config, tuple(pair), str(ckpt_path), num_samples=20000)
                _generate_rama_plot(args.config, tuple(pair), str(ckpt_path))


if __name__ == "__main__":
    main()
