#!/usr/bin/env python3
"""
Exact replication of main.py evaluation to debug acceptance rate differences.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.accelmd.utils.config import load_config, create_run_config, setup_device
from src.accelmd.evaluation.swap_acceptance import naive_acceptance, flow_acceptance
from src.accelmd.data.pt_pair_dataset import PTTemperaturePairDataset
from src.accelmd.targets import build_target
from torch.utils.data import DataLoader
from main import build_model


def replicate_main_evaluation(config_path: str, checkpoint_path: str, pair: tuple, num_samples: int = 1000):
    """Exactly replicate what main.py does for evaluation."""
    
    print(f"=== Replicating main.py evaluation ===")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Pair: {pair}")
    print(f"Samples: {num_samples}")
    
    # Step 1: Load config exactly like main.py
    base_cfg = load_config(config_path)
    device = setup_device(base_cfg)
    cfg = create_run_config(base_cfg, pair, device)
    
    print(f"Device: {device}")
    print(f"Temperatures: {cfg['temperatures']['values']}")
    
    # Step 2: Create dataset exactly like main.py
    from src.accelmd.utils.config import is_multi_peptide_mode
    
    if is_multi_peptide_mode(base_cfg):
        print("Using multi-peptide mode dataset creation")
        # Multi-peptide mode: build peptide-specific dataset like main.py
        peptide_code = "AA"  # We're testing AA peptide
        peptide_dir = Path("datasets/pt_dipeptides") / peptide_code
        pt_data_path = str(peptide_dir / f"pt_{peptide_code}.pt")
        molecular_data_path = str(peptide_dir)
    else:
        print("Using single-peptide mode dataset creation")
        # Single-peptide mode: use config paths
        pt_data_path = cfg["data"]["pt_data_path"]
        molecular_data_path = cfg["data"]["molecular_data_path"]
    
    print(f"PT data path: {pt_data_path}")
    print(f"Molecular data path: {molecular_data_path}")
    
    dataset = PTTemperaturePairDataset(
        pt_data_path=pt_data_path,
        molecular_data_path=molecular_data_path,
        temp_pair=pair,
        subsample_rate=cfg["data"].get("subsample_rate", 100),
        device="cpu",
        filter_chirality=cfg["data"].get("filter_chirality", False),
        center_coordinates=cfg["data"].get("center_coordinates", True),
    )
    
    # Extract num_atoms dynamically from the dataset (like main.py)
    dynamic_num_atoms = dataset.source_coords.shape[1]
    print(f"Dynamically detected {dynamic_num_atoms} atoms from dataset")
    cfg["model"]["num_atoms"] = dynamic_num_atoms
    
    batch_size = cfg["training"].get("batch_size", 64)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PTTemperaturePairDataset.collate_fn,
    )
    
    # Step 3: Build model exactly like main.py
    model_cfg = cfg["model"]
    temps = cfg["temperatures"]["values"]
    sys_cfg = cfg.get("system", {})
    energy_cut = sys_cfg.get("energy_cut")
    energy_max = sys_cfg.get("energy_max")
    
    # Determine target based on peptide_code (like main.py)
    if is_multi_peptide_mode(base_cfg):
        # Multi-peptide mode: peptide is AA
        peptide_code = "AA"
    else:
        # Single-peptide mode: read from config
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
    
    # Add system-level energy parameters to target kwargs (like main.py)
    target_kwargs_extra.update({
        "energy_cut": float(energy_cut) if energy_cut is not None else None,
        "energy_max": float(energy_max) if energy_max is not None else None,
    })
    
    print(f"Target name: {target_name}")
    print(f"Target kwargs: {target_kwargs_extra}")
    
    # Build model using the exact same function as main.py
    model = build_model(
        model_cfg=model_cfg,
        pair=pair,
        temps=temps,
        target_name=target_name,
        target_kwargs=target_kwargs_extra,
        device=device,
        num_atoms=model_cfg["num_atoms"],
    )
    
    # Load checkpoint exactly like main.py
    import torch
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Step 4: Build targets exactly like main.py
    target_kwargs = target_kwargs_extra.copy()
    target_kwargs.update({
        "energy_cut": float(energy_cut) if energy_cut is not None else None,
        "energy_max": float(energy_max) if energy_max is not None else None,
    })
    
    base_low = build_target(target_name, temperature=temps[pair[0]], device="cpu", **target_kwargs)
    base_high = build_target(target_name, temperature=temps[pair[1]], device="cpu", **target_kwargs)
    
    # Step 5: Compute acceptance exactly like main.py
    max_batches = (num_samples + batch_size - 1) // batch_size
    
    naive_acc = naive_acceptance(loader, base_low, base_high, max_batches=max_batches)
    flow_acc = flow_acceptance(loader, model, base_low, base_high, device=device, max_batches=max_batches)
    
    print(f"\nSwap acceptance estimate for {peptide_code} (pair {pair[0]} ↔ {pair[1]}, {num_samples} samples)")
    print("    naïve swap : %.4f" % naive_acc)
    print("    flow swap  : %.4f" % flow_acc)
    
    return naive_acc, flow_acc


def main():
    """Test both simple and transformer models."""
    
    print("Testing Simple Flow Model:")
    print("-" * 50)
    simple_naive, simple_flow = replicate_main_evaluation(
        config_path="configs/AA_simple_01.yaml",
        checkpoint_path="outputs/AA_simple/pair_0_1/models/best_model_epoch2986.pt",
        pair=(0, 1),
        num_samples=1000
    )
    
    print("\n\nTesting Transformer Flow Model:")
    print("-" * 50)
    transformer_naive, transformer_flow = replicate_main_evaluation(
        config_path="configs/multi_transformer_01.yaml",
        checkpoint_path="outputs/multi_transformer/pair_0_1/models/best_model_epoch48.pt",
        pair=(0, 1),
        num_samples=1000
    )
    
    print("\n" + "="*60)
    print("COMPARISON WITH EXPECTED VALUES:")
    print("="*60)
    print(f"Simple Flow:")
    print(f"  Expected: naive=0.3807, flow=0.5149")
    print(f"  Got:      naive={simple_naive:.4f}, flow={simple_flow:.4f}")
    print(f"  Match:    naive={'✓' if abs(simple_naive - 0.3807) < 0.01 else '✗'}, flow={'✓' if abs(simple_flow - 0.5149) < 0.01 else '✗'}")
    
    print(f"\nTransformer Flow:")
    print(f"  Expected: naive=0.3807, flow=0.8277")
    print(f"  Got:      naive={transformer_naive:.4f}, flow={transformer_flow:.4f}")
    print(f"  Match:    naive={'✓' if abs(transformer_naive - 0.3807) < 0.01 else '✗'}, flow={'✓' if abs(transformer_flow - 0.8277) < 0.01 else '✗'}")


if __name__ == "__main__":
    main()