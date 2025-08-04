#!/usr/bin/env python3
"""Self-contained energy conservation analysis for PT swap flows.

This script analyzes energy conservation by comparing energy distributions
of native configurations vs flow-mapped configurations across temperature pairs.

Usage:
    # Single pair analysis
    python energy_analysis.py --config configs/AA_simple.yaml --pair 0 1 --checkpoint checkpoints/AA_simple/pair_0_1/models/best_model_epoch2787.pt
    
    # Run all analyses for an experiment
    python energy_analysis.py --run-all AA_simple
    python energy_analysis.py --run-all multi_transformer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.accelmd.utils.config import load_config, setup_device, create_run_config
from src.accelmd.data.pt_pair_dataset import PTTemperaturePairDataset
from src.accelmd.targets import build_target
from main import build_model


def load_flow_model(config_path: str, checkpoint_path: str, pair: Tuple[int, int], device: str):
    """Load a trained flow model from checkpoint."""
    base_cfg = load_config(config_path)
    cfg = create_run_config(base_cfg, pair, device)
    
    # Create dataset to get num_atoms
    if base_cfg.get("mode") == "multi":
        # For multi-peptide mode, use first training peptide as representative
        from src.accelmd.utils.config import get_training_peptides
        train_peptides = get_training_peptides(base_cfg)
        representative_peptide = train_peptides[0]
        
        peptide_dir = Path("datasets/pt_dipeptides") / representative_peptide
        pt_data_path = peptide_dir / f"pt_{representative_peptide}.pt"
        molecular_data_path = peptide_dir
        
        dataset = PTTemperaturePairDataset(
            pt_data_path=str(pt_data_path),
            molecular_data_path=str(molecular_data_path),
            temp_pair=pair,
            subsample_rate=cfg["data"].get("subsample_rate", 100),
            device="cpu",
            filter_chirality=cfg["data"].get("filter_chirality", False),
            center_coordinates=cfg["data"].get("center_coordinates", True),
        )
        
        target_name = "dipeptide"
        target_kwargs_extra = {
            "pdb_path": f"datasets/pt_dipeptides/{representative_peptide}/ref.pdb",
            "env": "implicit"
        }
    else:
        # Single-peptide mode
        dataset = PTTemperaturePairDataset(
            pt_data_path=cfg["data"]["pt_data_path"],
            molecular_data_path=cfg["data"]["molecular_data_path"],
            temp_pair=pair,
            subsample_rate=cfg["data"].get("subsample_rate", 100),
            device="cpu",
            filter_chirality=cfg["data"].get("filter_chirality", False),
            center_coordinates=cfg["data"].get("center_coordinates", True),
        )
        
        # Determine target based on peptide_code
        peptide_code = cfg["peptide_code"].upper()
        if peptide_code == "AX":
            target_name = "aldp"
            target_kwargs_extra = {}
        else:
            target_name = "dipeptide"
            pdb_path = f"datasets/pt_dipeptides/{peptide_code}/ref.pdb"
            target_kwargs_extra = {
                "pdb_path": pdb_path,
                "env": "implicit"
            }
    
    # Extract num_atoms dynamically from dataset
    dynamic_num_atoms = dataset.source_coords.shape[1]
    cfg["model"]["num_atoms"] = dynamic_num_atoms
    
    # Build model
    model_cfg = cfg["model"]
    temps = cfg["temperatures"]["values"]
    sys_cfg = cfg.get("system", {})
    energy_cut = sys_cfg.get("energy_cut")
    energy_max = sys_cfg.get("energy_max")
    
    target_kwargs_extra.update({
        "energy_cut": float(energy_cut) if energy_cut is not None else None,
        "energy_max": float(energy_max) if energy_max is not None else None,
    })
    
    model = build_model(
        model_cfg=model_cfg,
        pair=pair,
        temps=temps,
        target_name=target_name,
        target_kwargs=target_kwargs_extra,
        device=device,
        num_atoms=dynamic_num_atoms,
    )
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, dataset, cfg, target_kwargs_extra


def sample_configurations(dataset: PTTemperaturePairDataset, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample configurations from the dataset."""
    loader = DataLoader(
        dataset,
        batch_size=min(64, num_samples),
        shuffle=True,
        collate_fn=PTTemperaturePairDataset.collate_fn,
    )
    
    source_coords_list = []
    target_coords_list = []
    
    total_collected = 0
    for batch in loader:
        if total_collected >= num_samples:
            break
            
        batch_size = batch["source_coords"].shape[0]
        remaining = num_samples - total_collected
        
        if remaining < batch_size:
            source_coords_list.append(batch["source_coords"][:remaining])
            target_coords_list.append(batch["target_coords"][:remaining])
            total_collected += remaining
        else:
            source_coords_list.append(batch["source_coords"])
            target_coords_list.append(batch["target_coords"])
            total_collected += batch_size
    
    source_coords = torch.cat(source_coords_list, dim=0)
    target_coords = torch.cat(target_coords_list, dim=0)
    
    return source_coords, target_coords


def compute_energies(coords: torch.Tensor, target_dist, device: str) -> np.ndarray:
    """Compute potential energies for a batch of coordinates."""
    coords = coords.to(device)
    
    with torch.no_grad():
        coords_flat = coords.view(coords.shape[0], -1)
        log_probs = target_dist.log_prob(coords_flat)
        
        # Convert log probabilities back to energies
        beta = target_dist.beta
        energies = -log_probs / beta
    
    return energies.cpu().numpy()


def map_through_flow(model, coords: torch.Tensor, atom_types: torch.Tensor, 
                    adj_list: torch.Tensor, direction: str = "forward") -> torch.Tensor:
    """Map coordinates through the flow."""
    device = next(model.parameters()).device
    coords = coords.to(device)
    atom_types = atom_types.to(device)
    adj_list = adj_list.to(device)
    
    with torch.no_grad():
        model_type = model.__class__.__name__
        
        if model_type == "PTSwapFlow":
            mapped_coords = model.sample_proposal(
                source_coords=coords,
                direction=direction
            )
        elif model_type == "PTSwapGraphFlow":
            mapped_coords = model.sample_proposal(
                source_coords=coords,
                atom_types=atom_types,
                adj_list=adj_list,
                direction=direction
            )
        elif model_type == "PTSwapTransformerFlow":
            if direction == "forward":
                mapped_coords, _ = model.forward(
                    coordinates=coords,
                    atom_types=atom_types,
                    adj_list=adj_list,
                    masked_elements=None,
                    reverse=False,
                    return_log_det=False
                )
            else:
                mapped_coords, _ = model.forward(
                    coordinates=coords,
                    atom_types=atom_types,
                    adj_list=adj_list,
                    masked_elements=None,
                    reverse=True,
                    return_log_det=False
                )
        else:
            if direction == "forward":
                mapped_coords, _ = model.forward(coords)
            else:
                mapped_coords, _ = model.inverse(coords)
    
    return mapped_coords.cpu()


def create_energy_plot(
    native_low_energies: np.ndarray,
    native_high_energies: np.ndarray,
    mapped_high_energies: np.ndarray,
    temps: List[float],
    pair: Tuple[int, int],
    output_path: str
) -> Dict:
    """Create energy conservation plot with smooth density curves."""
    from scipy.stats import gaussian_kde
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    low_temp, high_temp = temps[pair[0]], temps[pair[1]]
    
    # Determine common energy range
    all_energies = np.concatenate([native_low_energies, native_high_energies, mapped_high_energies])
    energy_min, energy_max = all_energies.min(), all_energies.max()
    energy_grid = np.linspace(energy_min, energy_max, 1000)
    
    # Create KDE for each distribution
    kde_low = gaussian_kde(native_low_energies)
    kde_high = gaussian_kde(native_high_energies)
    kde_mapped = gaussian_kde(mapped_high_energies)
    
    # Evaluate densities
    density_low = kde_low(energy_grid)
    density_high = kde_high(energy_grid)
    density_mapped = kde_mapped(energy_grid)
    
    # Plot smooth density curves
    ax.fill_between(energy_grid, density_low, alpha=0.4, color='blue', 
                    label=f'Native T={low_temp:.1f}K')
    ax.plot(energy_grid, density_low, color='blue', linewidth=2.5)
    
    ax.fill_between(energy_grid, density_high, alpha=0.4, color='green', 
                    label=f'Native T={high_temp:.1f}K')
    ax.plot(energy_grid, density_high, color='green', linewidth=2.5)
    
    ax.fill_between(energy_grid, density_mapped, alpha=0.4, color='red', 
                    label=f'Flow(T={low_temp:.1f}K → T={high_temp:.1f}K)')
    ax.plot(energy_grid, density_mapped, color='red', linewidth=2.5)
    
    ax.set_xlabel('Energy (kJ/mol)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Energy Conservation: T={low_temp:.1f}K → T={high_temp:.1f}K', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Compute and display statistics
    high_diff = mapped_high_energies - native_high_energies
    mae = np.mean(np.abs(high_diff))
    rmse = np.sqrt(np.mean(high_diff**2))
    
    stats_text = f'MAE: {mae:.1f} kJ/mol\nRMSE: {rmse:.1f} kJ/mol'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'low_temp': float(low_temp),
        'high_temp': float(high_temp),
        'high_energy_mae': float(mae),
        'high_energy_rmse': float(rmse),
    }


def analyze_single_pair(
    config_path: str,
    checkpoint_path: str,
    pair: Tuple[int, int],
    num_samples: int = 2000,
    output_dir: Optional[str] = None
) -> Dict:
    """Analyze energy conservation for a single temperature pair."""
    print(f"Analyzing pair {pair[0]}-{pair[1]}...")
    
    # Setup
    base_cfg = load_config(config_path)
    device = setup_device(base_cfg)
    
    # Load model and dataset
    model, dataset, cfg, target_kwargs = load_flow_model(config_path, checkpoint_path, pair, device)
    
    # Sample configurations
    source_coords, target_coords = sample_configurations(dataset, num_samples)
    
    # Get molecular data
    atom_types = dataset.atom_types.unsqueeze(0).expand(source_coords.shape[0], -1)
    adj_list = dataset.adj_list
    
    # Build target distributions
    temps = cfg["temperatures"]["values"]
    sys_cfg = cfg.get("system", {})
    energy_cut = sys_cfg.get("energy_cut")
    energy_max = sys_cfg.get("energy_max")
    
    target_kwargs.update({
        "energy_cut": float(energy_cut) if energy_cut is not None else None,
        "energy_max": float(energy_max) if energy_max is not None else None,
    })
    
    # Determine target type
    if base_cfg.get("mode") == "multi":
        target_name = "dipeptide"
    else:
        peptide_code = cfg["peptide_code"].upper()
        target_name = "aldp" if peptide_code == "AX" else "dipeptide"
    
    target_low = build_target(target_name, temperature=temps[pair[0]], device=device, **target_kwargs)
    target_high = build_target(target_name, temperature=temps[pair[1]], device=device, **target_kwargs)
    
    # Compute energies
    native_low_energies = compute_energies(source_coords, target_low, device)
    native_high_energies = compute_energies(target_coords, target_high, device)
    
    # Map through flow and compute mapped energies
    mapped_high_coords = map_through_flow(model, source_coords, atom_types, adj_list, "forward")
    mapped_high_energies = compute_energies(mapped_high_coords, target_high, device)
    
    # Create output directory
    if output_dir is None:
        output_dir = f"experiments/energy/plots/{base_cfg['experiment_name']}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create plot and get statistics
    plot_path = output_path / f"pair_{pair[0]}_{pair[1]}_energy_comparison.png"
    stats = create_energy_plot(
        native_low_energies, native_high_energies, mapped_high_energies,
        temps, pair, str(plot_path)
    )
    
    # Save statistics
    stats_path = output_path / f"pair_{pair[0]}_{pair[1]}_energy_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  MAE: {stats['high_energy_mae']:.1f} kJ/mol")
    print(f"  Plot: {plot_path}")
    
    return stats


def run_all_analyses(experiment: str):
    """Run energy analysis for all pairs in an experiment."""
    print(f"Running all analyses for {experiment}...")
    
    # Define experiment configurations
    if experiment == "AA_simple":
        config_path = "configs/AA_simple.yaml"
        pairs = [(0,1), (1,2), (2,3), (3,4)]
        checkpoints = [
            "checkpoints/AA_simple/pair_0_1/models/best_model_epoch2787.pt",
            "checkpoints/AA_simple/pair_1_2/models/best_model_epoch1231.pt",
            "checkpoints/AA_simple/pair_2_3/models/best_model_epoch1301.pt",
            "checkpoints/AA_simple/pair_3_4/models/best_model_epoch773.pt",
        ]
    elif experiment == "multi_transformer":
        config_path = "configs/multi_transformer.yaml"
        pairs = [(0,1), (1,2), (2,3), (3,4)]
        checkpoints = [
            "checkpoints/multi_transformer/pair_0_1/models/best_model_epoch331.pt",
            "checkpoints/multi_transformer/pair_1_2/models/best_model_epoch325.pt",
            "checkpoints/multi_transformer/pair_2_3/models/best_model_epoch257.pt",
            "checkpoints/multi_transformer/pair_3_4/models/best_model_epoch254.pt",
        ]
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    all_stats = []
    for pair, checkpoint in zip(pairs, checkpoints):
        stats = analyze_single_pair(config_path, checkpoint, pair)
        all_stats.append(stats)
    
    # Print summary
    print(f"\n=== {experiment} Summary ===")
    for i, (pair, stats) in enumerate(zip(pairs, all_stats)):
        print(f"Pair {pair[0]}-{pair[1]}: MAE = {stats['high_energy_mae']:.1f} kJ/mol")
    
    avg_mae = np.mean([s['high_energy_mae'] for s in all_stats])
    print(f"Average MAE: {avg_mae:.1f} kJ/mol")


def main():
    parser = argparse.ArgumentParser(description='Energy conservation analysis for PT swap flows')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--checkpoint', help='Path to checkpoint file')
    parser.add_argument('--pair', nargs=2, type=int, help='Temperature pair indices')
    parser.add_argument('--num-samples', type=int, default=2000, help='Number of samples')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--run-all', choices=['AA_simple', 'multi_transformer'], 
                       help='Run all analyses for an experiment')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_analyses(args.run_all)
    elif args.config and args.checkpoint and args.pair:
        pair = tuple(args.pair)
        analyze_single_pair(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            pair=pair,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
    else:
        parser.error("Either use --run-all or provide --config, --checkpoint, and --pair")


if __name__ == "__main__":
    main()