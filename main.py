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

import torch
from torch.utils.data import DataLoader

from src.accelmd.utils.config import load_config, setup_device, get_temperature_pairs, create_run_config
from src.accelmd.utils.ramachandran_plotting import generate_ramachandran_grid
from src.accelmd.data.pt_pair_dataset import PTTemperaturePairDataset
from src.accelmd.flows import PTSwapFlow, PTSwapGraphFlow
from src.accelmd.evaluation.swap_acceptance import naive_acceptance, flow_acceptance
# from src.accelmd.targets.aldp_boltzmann import AldpBoltzmann  # Now using build_target
from src.accelmd.training.trainer import PTSwapTrainer


def build_model(model_cfg: dict, pair: Tuple[int, int], temps: list, target_name: str, 
                target_kwargs: dict, device: str, num_atoms: int = None):
    """Build either PTSwapFlow or PTSwapGraphFlow based on config architecture setting."""
    
    architecture = model_cfg.get("architecture", "simple")
    
    if architecture == "simple":
        # Simple coordinate-to-coordinate flow
        if num_atoms is None:
            raise ValueError("num_atoms must be provided for simple architecture")
            
        return PTSwapFlow(
            num_atoms=num_atoms,
            num_layers=model_cfg["flow_layers"],
            hidden_dim=model_cfg["hidden_dim"],
            source_temperature=temps[pair[0]],
            target_temperature=temps[pair[1]],
            target_name=target_name,
            target_kwargs=target_kwargs,
            device=device,
        )
        
    elif architecture == "graph":
        # Graph-conditioned flow with molecular structure
        graph_cfg = model_cfg.get("graph", {})
        
        return PTSwapGraphFlow(
            num_layers=model_cfg["flow_layers"],
            atom_vocab_size=graph_cfg.get("atom_vocab_size", 4),
            atom_embed_dim=graph_cfg.get("atom_embed_dim", 32),
            hidden_dim=graph_cfg.get("hidden_dim", model_cfg["hidden_dim"]),
            num_mp_layers=graph_cfg.get("num_mp_layers", 2),
            source_temperature=temps[pair[0]],
            target_temperature=temps[pair[1]],
            target_name=target_name,
            target_kwargs=target_kwargs,
            device=device,
        )
        
    elif architecture == "transformer":
        # Transformer-based flow with dimension-agnostic design
        from src.accelmd.flows import PTSwapTransformerFlow
        from src.accelmd.flows.transformer_block import TransformerConfig
        from src.accelmd.flows.rff_position_encoder import RFFPositionEncoderConfig
        
        transformer_cfg = model_cfg.get("transformer", {})
        
        # Create transformer configuration
        transformer_config = TransformerConfig(
            n_head=transformer_cfg.get("n_head", 8),
            dim_feedforward=transformer_cfg.get("dim_feedforward", 2048),
            dropout=0.0,  # No dropout for deterministic likelihood
        )
        
        # Create RFF position encoder configuration
        rff_config = RFFPositionEncoderConfig(
            encoding_dim=transformer_cfg.get("rff_encoding_dim", 64),
            scale_mean=transformer_cfg.get("rff_scale_mean", 1.0),
            scale_stddev=transformer_cfg.get("rff_scale_stddev", 1.0),
        )
        
        return PTSwapTransformerFlow(
            num_layers=model_cfg["flow_layers"],
            atom_vocab_size=transformer_cfg.get("atom_vocab_size", 4),
            atom_embed_dim=transformer_cfg.get("atom_embed_dim", 32),
            transformer_hidden_dim=transformer_cfg.get("transformer_hidden_dim", 128),
            mlp_hidden_layer_dims=transformer_cfg.get("mlp_hidden_layer_dims", [128, 128]),
            num_transformer_layers=transformer_cfg.get("num_transformer_layers", 2),
            source_temperature=temps[pair[0]],
            target_temperature=temps[pair[1]],
            target_name=target_name,
            target_kwargs=target_kwargs,
            transformer_config=transformer_config,
            rff_position_encoder_config=rff_config,
            device=device,
        )
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Must be 'simple', 'graph', or 'transformer'.")


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
        filter_chirality=cfg["data"].get("filter_chirality", False),
        center_coordinates=cfg["data"].get("center_coordinates", True),
    )
    
    # Extract num_atoms dynamically from the dataset
    dynamic_num_atoms = dataset.source_coords.shape[1]  # [N, atoms, 3] -> atoms
    print(f"Dynamically detected {dynamic_num_atoms} atoms from dataset")
    
    # Override config with actual num_atoms from data
    cfg["model"]["num_atoms"] = dynamic_num_atoms
    
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

    # Model - now uses dynamic num_atoms
    model_cfg = cfg["model"]
    temps = cfg["temperatures"]["values"]
    sys_cfg = cfg.get("system", {})
    energy_cut = sys_cfg.get("energy_cut")
    energy_max = sys_cfg.get("energy_max")
    
    # Determine target based on peptide_code
    peptide_code = cfg["peptide_code"].upper()
    if peptide_code == "AX":
        target_name = "aldp"
        target_kwargs_extra = {}
    else:
        target_name = "dipeptide"
        # For dipeptide target, we need PDB path and environment
        pdb_path = f"datasets/timewarp/2AA-1-big/train/{peptide_code}-traj-state0.pdb"
        target_kwargs_extra = {
            "pdb_path": pdb_path,
            "env": "implicit"
        }
    
    # Add system-level energy parameters to target kwargs
    target_kwargs_extra.update({
        "energy_cut": float(energy_cut) if energy_cut is not None else None,
        "energy_max": float(energy_max) if energy_max is not None else None,
    })

    model = build_model(
        model_cfg=model_cfg,
        pair=pair,
        temps=temps,
        target_name=target_name,  # Determined from peptide_code
        target_kwargs=target_kwargs_extra,
        device=device,
        num_atoms=model_cfg["num_atoms"],  # Only used for simple architecture
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
        filter_chirality=cfg["data"].get("filter_chirality", False),
        center_coordinates=cfg["data"].get("center_coordinates", True),
    )
    
    # Extract num_atoms dynamically from the dataset
    dynamic_num_atoms = dataset.source_coords.shape[1]  # [N, atoms, 3] -> atoms
    print(f"Dynamically detected {dynamic_num_atoms} atoms from dataset")
    
    # Override config with actual num_atoms from data
    cfg["model"]["num_atoms"] = dynamic_num_atoms
    
    batch_size = cfg["training"].get("batch_size", 64)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PTTemperaturePairDataset.collate_fn,
    )

    # Build flow model & load checkpoint - now uses dynamic num_atoms
    model_cfg = cfg["model"]
    temps = cfg["temperatures"]["values"]
    sys_cfg = cfg.get("system", {})
    energy_cut = sys_cfg.get("energy_cut")
    energy_max = sys_cfg.get("energy_max")
    
    # Determine target based on peptide_code
    peptide_code = cfg["peptide_code"].upper()
    if peptide_code == "AX":
        target_name = "aldp"
        target_kwargs_extra = {}
    else:
        target_name = "dipeptide"
        # For dipeptide target, we need PDB path and environment
        pdb_path = f"datasets/timewarp/2AA-1-big/train/{peptide_code}-traj-state0.pdb"
        target_kwargs_extra = {
            "pdb_path": pdb_path,
            "env": "implicit"
        }
    
    # Add system-level energy parameters to target kwargs
    target_kwargs_extra.update({
        "energy_cut": float(energy_cut) if energy_cut is not None else None,
        "energy_max": float(energy_max) if energy_max is not None else None,
    })

    model = build_model(
        model_cfg=model_cfg,
        pair=pair,
        temps=temps,
        target_name=target_name,  # Determined from peptide_code
        target_kwargs=target_kwargs_extra,
        device=device,
        num_atoms=model_cfg["num_atoms"],  # Only used for simple architecture
    )
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    # Boltzmann bases (cpu) - use the same target type as the model
    from src.accelmd.targets import build_target
    target_kwargs = target_kwargs_extra.copy()
    target_kwargs.update({
        "energy_cut": float(energy_cut) if energy_cut is not None else None,
        "energy_max": float(energy_max) if energy_max is not None else None,
    })
    
    base_low = build_target(target_name, temperature=temps[pair[0]], device="cpu", **target_kwargs)
    base_high = build_target(target_name, temperature=temps[pair[1]], device="cpu", **target_kwargs)

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
    """Generate 2×2 Ramachandran grid for the given pair using the utility function.

    The plot is saved under <pair_dir>/plots/rama_grid.png where *pair_dir* is
    dictated by `create_run_config` in utils.config.
    """
    base_cfg = load_config(cfg_path)
    run_cfg = create_run_config(base_cfg, pair, device="cpu")
    plots_dir = Path(run_cfg["output"]["pair_dir"]).expanduser() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "rama_grid.png"

    try:
        success = generate_ramachandran_grid(
            config_path=cfg_path,
            checkpoint_path=str(checkpoint),
            temp_pair=pair,
            output_path=str(out_path),
            n_samples=20000,
        )
        if not success:
            print(f"[WARN] Ramachandran plot generation failed for pair {pair}")
        else:
            print(f"Ramachandran plot saved to {out_path}")
    except Exception as e:
        print(f"[WARN] Ramachandran plot generation failed for pair {pair}: {e}")


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
