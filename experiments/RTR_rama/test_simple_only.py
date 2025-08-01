#!/usr/bin/env python3
"""
Test script to verify the new simple models work properly.
Only tests simple models, not transformer models.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.accelmd.evaluation.swap_acceptance import naive_acceptance, flow_acceptance
from src.accelmd.data.pt_pair_dataset import PTTemperaturePairDataset
from src.accelmd.targets import build_target
from torch.utils.data import DataLoader


def test_simple_model(pair: tuple):
    """Test a single simple model for a temperature pair."""
    
    print(f"\n{'='*50}")
    print(f"Testing Simple Model for Pair {pair}")
    print(f"{'='*50}")
    
    # Model configuration
    config_path = str(project_root / "configs/AA_simple.yaml")
    epoch_map = {(0, 1): 2986, (1, 2): 1231, (2, 3): 931, (3, 4): 892}
    
    epoch = epoch_map.get(pair)
    if epoch is None:
        print(f"No epoch mapping for pair {pair}")
        return None
        
    model_path = str(project_root / f"outputs/AA_simple/pair_{pair[0]}_{pair[1]}/models/best_model_epoch{epoch}.pt")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    print(f"Model path: {model_path}")
    
    # Load configuration exactly like main.py
    from src.accelmd.utils.config import load_config, create_run_config, setup_device, is_multi_peptide_mode
    from main import build_model
    
    base_cfg = load_config(config_path)
    device = setup_device(base_cfg)
    cfg = create_run_config(base_cfg, pair, device)
    
    # Create dataset exactly like main.py
    if is_multi_peptide_mode(base_cfg):
        # This shouldn't happen for simple models, but just in case
        peptide_code = "AA"
        peptide_dir = project_root / "datasets/pt_dipeptides" / peptide_code
        pt_data_path = str(peptide_dir / f"pt_{peptide_code}.pt")
        molecular_data_path = str(peptide_dir)
    else:
        # Single-peptide mode (expected for simple models)
        pt_data_path = str(project_root / cfg["data"]["pt_data_path"])
        molecular_data_path = str(project_root / cfg["data"]["molecular_data_path"])
        peptide_code = cfg["peptide_code"].upper()
    
    print(f"Peptide: {peptide_code}")
    print(f"PT data: {pt_data_path}")
    print(f"Molecular data: {molecular_data_path}")
    
    dataset = PTTemperaturePairDataset(
        pt_data_path=pt_data_path,
        molecular_data_path=molecular_data_path,
        temp_pair=pair,
        subsample_rate=cfg["data"].get("subsample_rate", 100),
        device="cpu",
        filter_chirality=cfg["data"].get("filter_chirality", False),
        center_coordinates=cfg["data"].get("center_coordinates", True),
    )
    
    print(f"Dataset loaded: {len(dataset)} samples, {dataset.source_coords.shape[1]} atoms")
    
    # Update config with dynamic num_atoms
    dynamic_num_atoms = dataset.source_coords.shape[1]
    cfg["model"]["num_atoms"] = dynamic_num_atoms
    
    batch_size = cfg["training"].get("batch_size", 64)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PTTemperaturePairDataset.collate_fn,
    )
    
    # Build model exactly like main.py
    model_cfg = cfg["model"]
    temps = cfg["temperatures"]["values"]
    sys_cfg = cfg.get("system", {})
    energy_cut = sys_cfg.get("energy_cut")
    energy_max = sys_cfg.get("energy_max")
    
    # Target configuration
    if peptide_code == "AX":
        target_name = "aldp"
        target_kwargs_extra = {}
    else:
        target_name = "dipeptide"
        pdb_path = str(project_root / f"datasets/pt_dipeptides/{peptide_code}/ref.pdb")
        target_kwargs_extra = {
            "pdb_path": pdb_path,
            "env": "implicit"
        }
    
    target_kwargs_extra.update({
        "energy_cut": float(energy_cut) if energy_cut is not None else None,
        "energy_max": float(energy_max) if energy_max is not None else None,
    })
    
    print(f"Building model with target: {target_name}")
    print(f"Temperature pair: {temps[pair[0]]:.1f}K ↔ {temps[pair[1]]:.1f}K")
    
    # Build and load model
    model = build_model(
        model_cfg=model_cfg,
        pair=pair,
        temps=temps,
        target_name=target_name,
        target_kwargs=target_kwargs_extra,
        device=device,
        num_atoms=model_cfg["num_atoms"],
    )
    
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Build targets exactly like main.py
    target_kwargs = target_kwargs_extra.copy()
    base_low = build_target(target_name, temperature=temps[pair[0]], device="cpu", **target_kwargs)
    base_high = build_target(target_name, temperature=temps[pair[1]], device="cpu", **target_kwargs)
    
    # Compute acceptance rates exactly like main.py
    max_batches = (1000 + batch_size - 1) // batch_size  # 1000 samples like main.py
    
    print("Computing acceptance rates...")
    naive_acc = naive_acceptance(loader, base_low, base_high, max_batches=max_batches)
    flow_acc = flow_acceptance(loader, model, base_low, base_high, device=device, max_batches=max_batches)
    
    print(f"Results for pair {pair}:")
    print(f"  Naive swap: {naive_acc:.4f}")
    print(f"  Flow swap:  {flow_acc:.4f}")
    
    if flow_acc > 0:
        speedup = flow_acc / naive_acc if naive_acc > 0 else float('inf')
        print(f"  Speedup:    {speedup:.2f}x")
        if flow_acc > naive_acc:
            print("  ✅ SUCCESS: Flow improves over naive!")
        else:
            print("  ⚠️  WARNING: Flow worse than naive")
    else:
        print("  ❌ FAILURE: Flow acceptance is zero!")
    
    return naive_acc, flow_acc


def main():
    """Test all simple models."""
    print("Testing New Simple Models")
    print("=" * 60)
    
    # Test all temperature pairs
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    results = {}
    
    for pair in pairs:
        result = test_simple_model(pair)
        if result:
            naive_acc, flow_acc = result
            results[pair] = {"naive": naive_acc, "flow": flow_acc}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF SIMPLE MODEL PERFORMANCE")
    print(f"{'='*60}")
    
    for pair in pairs:
        if pair in results:
            naive = results[pair]["naive"]
            flow = results[pair]["flow"]
            if naive > 0 and flow > 0:
                speedup = flow / naive
                status = "✅" if flow > naive else "⚠️"
            else:
                speedup = 0.0
                status = "❌"
            
            print(f"Pair {pair}: naive={naive:.4f}, flow={flow:.4f}, speedup={speedup:.2f}x {status}")
        else:
            print(f"Pair {pair}: FAILED TO EVALUATE ❌")


if __name__ == "__main__":
    main()