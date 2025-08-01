#!/usr/bin/env python3
"""
Simple test to verify model loading and basic flow acceptance computation.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.accelmd.flows.pt_swap_flow import PTSwapFlow
from src.accelmd.flows.pt_swap_transformer_flow import PTSwapTransformerFlow
from src.accelmd.evaluation.swap_acceptance import flow_acceptance, naive_acceptance
from src.accelmd.data.pt_pair_dataset import PTTemperaturePairDataset
from torch.utils.data import DataLoader


def load_model(model_type: str, pair: Tuple[int, int], device: str = "cpu"):
    """Load a specific model."""
    
    if model_type == "simple":
        path = f"outputs/AA_simple/pair_{pair[0]}_{pair[1]}/models/best_model_epoch2986.pt"
        if not os.path.exists(path):
            print(f"Simple model not found: {path}")
            return None
            
        model = PTSwapFlow(
            num_atoms=23,  # AA dipeptide has 23 atoms
            num_layers=8,
            hidden_dim=512,  # From config
            source_temperature=300.0 + pair[0] * 100,  # Dummy temperatures
            target_temperature=300.0 + pair[1] * 100,
            target_name="dipeptide",
            target_kwargs={
                "pdb_path": "datasets/pt_dipeptides/AA/ref.pdb",
                "env": "implicit"
            },
            device=device
        )
        
    elif model_type == "transformer":
        path = f"outputs/multi_transformer/pair_{pair[0]}_{pair[1]}/models/best_model_epoch48.pt"
        if not os.path.exists(path):
            print(f"Transformer model not found: {path}")
            return None
            
        model = PTSwapTransformerFlow(
            num_layers=8,
            atom_vocab_size=4,
            atom_embed_dim=32,
            transformer_hidden_dim=128,
            mlp_hidden_layer_dims=[128, 128],
            num_transformer_layers=2,
            source_temperature=300.0 + pair[0] * 100,
            target_temperature=300.0 + pair[1] * 100,
            target_name="dipeptide",
            target_kwargs={
                "pdb_path": "datasets/pt_dipeptides/AA/ref.pdb",
                "env": "implicit"
            },
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Loading {model_type} model for pair {pair}: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model


def test_acceptance_computation(model, model_name: str, device: str = "cpu"):
    """Test acceptance computation for a model."""
    print(f"\nTesting {model_name} acceptance computation...")
    
    # Create a small test dataset
    try:
        dataset = PTTemperaturePairDataset(
            pt_data_path="datasets/pt_dipeptides/AA/pt_AA.pt",
            molecular_data_path="datasets/pt_dipeptides/AA/",
            temp_pair=(0, 1),
            subsample_rate=1000,  # Use more subsampling for speed
            device="cpu"  # Keep data on CPU
        )
        
        loader = DataLoader(dataset, batch_size=16, shuffle=False, 
                          collate_fn=dataset.collate_fn)
        
        print(f"Dataset size: {len(dataset)} samples")
        
        # Test naive acceptance
        naive_acc = naive_acceptance(
            loader=loader,
            base_low=model.base_low,
            base_high=model.base_high,
            max_batches=5  # Just test a few batches
        )
        
        print(f"Naive acceptance: {naive_acc:.4f}")
        
        # Test flow acceptance
        flow_acc = flow_acceptance(
            loader=loader,
            model=model,
            base_low=model.base_low,
            base_high=model.base_high,
            device=device,
            max_batches=5  # Just test a few batches
        )
        
        print(f"Flow acceptance: {flow_acc:.4f}")
        print(f"Flow speedup: {flow_acc / naive_acc:.2f}x")
        
        return naive_acc, flow_acc
        
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        return None, None


def main():
    """Main test function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test pair (0, 1)
    pair = (0, 1)
    
    print("="*60)
    print("Testing model loading and basic acceptance computation")
    print("="*60)
    
    # Test simple model
    simple_model = load_model("simple", pair, device)
    if simple_model is not None:
        naive_acc_simple, flow_acc_simple = test_acceptance_computation(
            simple_model, "Simple Flow", device
        )
    
    # Test transformer model
    transformer_model = load_model("transformer", pair, device)
    if transformer_model is not None:
        naive_acc_transformer, flow_acc_transformer = test_acceptance_computation(
            transformer_model, "Transformer Flow", device
        )
    
    print("\n" + "="*60)
    print("Summary:")
    if simple_model is not None and flow_acc_simple is not None:
        print(f"Simple Flow - Naive: {naive_acc_simple:.4f}, Flow: {flow_acc_simple:.4f}")
    if transformer_model is not None and flow_acc_transformer is not None:
        print(f"Transformer Flow - Naive: {naive_acc_transformer:.4f}, Flow: {flow_acc_transformer:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()