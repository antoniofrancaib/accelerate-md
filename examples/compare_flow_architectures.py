#!/usr/bin/env python3
"""Demonstration script comparing flow architectures for dissertation.

This script demonstrates the two different architectures:
1. Simple coordinate-to-coordinate sequential flow (PTSwapFlow)
2. Graph-conditioned flow with molecular structure (PTSwapGraphFlow)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import time

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.accelmd.flows import PTSwapFlow, PTSwapGraphFlow


def create_demo_config() -> tuple:
    """Create configurations for both architectures using config-style format."""
    
    # Simple flow configuration (matches config.yaml format)
    simple_config = {
        "architecture": "simple",
        "flow_layers": 6,
        "hidden_dim": 128,
    }
    
    # Graph-conditioned flow configuration (matches config.yaml format)
    graph_config = {
        "architecture": "graph", 
        "flow_layers": 6,
        "hidden_dim": 128,
        "graph": {
            "atom_vocab_size": 4,  # H, C, N, O
            "atom_embed_dim": 32,
            "graph_embed_dim": 64,
            "node_feature_dim": 64,
            "attention_lengthscales": [1.0, 2.0, 4.0],  # Multi-head attention
        }
    }
    
    return simple_config, graph_config


def generate_molecular_data(batch_size: int = 8) -> Dict[str, torch.Tensor]:
    """Generate synthetic molecular data for testing."""
    print(f"Generating molecular data for {batch_size} molecules...")
    
    # Create ALDP-like molecular data
    num_atoms = 22
    
    # Random coordinates (simulating molecular conformations)
    coordinates = torch.randn(batch_size, num_atoms, 3) * 2.0  # Å scale
    
    # Atom types (simplified: H=0, C=1, N=2, O=3)
    atom_types = torch.randint(0, 4, (batch_size, num_atoms))
    
    # Simple chain connectivity (each atom connected to next)
    adj_list = torch.stack([
        torch.arange(num_atoms - 1),
        torch.arange(1, num_atoms)
    ], dim=1)  # [21, 2] for ALDP-like chain
    
    # Edge batch indices (same edges for all molecules)
    edge_batch_idx = torch.cat([
        torch.full((num_atoms - 1,), i) for i in range(batch_size)
    ])
    
    # No masking for fixed-size molecules
    masked_elements = torch.zeros(batch_size, num_atoms, dtype=torch.bool)
    
    return {
        "coordinates": coordinates,
        "atom_types": atom_types,
        "adj_list": adj_list,
        "edge_batch_idx": edge_batch_idx,
        "masked_elements": masked_elements,
    }


def benchmark_architecture(model: nn.Module, data: Dict[str, torch.Tensor], 
                         architecture_name: str, num_runs: int = 10) -> Dict[str, float]:
    """Benchmark architecture performance."""
    print(f"\nBenchmarking {architecture_name}...")
    
    coords = data["coordinates"]
    batch_size = coords.shape[0]
    
    # Warmup
    if isinstance(model, PTSwapGraphFlow):
        _, _ = model.forward(
            coordinates=coords,
            atom_types=data["atom_types"],
            adj_list=data["adj_list"],
            edge_batch_idx=data["edge_batch_idx"],
            masked_elements=data["masked_elements"],
            reverse=False,
        )
    else:
        _, _ = model.forward(coords)
    
    # Benchmark forward pass
    forward_times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        if isinstance(model, PTSwapGraphFlow):
            output_coords, log_det = model.forward(
                coordinates=coords,
                atom_types=data["atom_types"],
                adj_list=data["adj_list"], 
                edge_batch_idx=data["edge_batch_idx"],
                masked_elements=data["masked_elements"],
                reverse=False,
            )
        else:
            output_coords, log_det = model.forward(coords)
            
        forward_times.append(time.time() - start_time)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Compute statistics
    forward_time_mean = np.mean(forward_times) * 1000  # Convert to ms
    forward_time_std = np.std(forward_times) * 1000
    
    # Test invertibility
    if isinstance(model, PTSwapGraphFlow):
        coords_reconstructed, log_det_inv = model.forward(
            coordinates=output_coords,
            atom_types=data["atom_types"],
            adj_list=data["adj_list"],
            edge_batch_idx=data["edge_batch_idx"],
            masked_elements=data["masked_elements"],
            reverse=True,
        )
    else:
        coords_reconstructed, log_det_inv = model.inverse(output_coords)
    
    reconstruction_error = torch.mean((coords - coords_reconstructed) ** 2).item()
    jacobian_error = torch.mean((log_det + log_det_inv) ** 2).item()
    
    results = {
        "forward_time_ms": forward_time_mean,
        "forward_time_std_ms": forward_time_std,
        "num_parameters": num_params,
        "reconstruction_error": reconstruction_error,
        "jacobian_error": jacobian_error,
        "log_det_magnitude": torch.mean(torch.abs(log_det)).item(),
    }
    
    print(f"  Forward time: {forward_time_mean:.2f} ± {forward_time_std:.2f} ms")
    print(f"  Parameters: {num_params:,}")
    print(f"  Reconstruction error: {reconstruction_error:.2e}")
    print(f"  Jacobian error: {jacobian_error:.2e}")
    
    return results


def demonstrate_swap_proposals(simple_model: PTSwapFlow, graph_model: PTSwapGraphFlow,
                             data: Dict[str, torch.Tensor]) -> None:
    """Demonstrate swap proposal generation."""
    print("\nGenerating PT swap proposals...")
    
    coords = data["coordinates"][:4]  # Use first 4 molecules
    
    # Simple flow proposals
    with torch.no_grad():
        simple_proposals = simple_model.sample_proposal(coords, direction="forward")
    
    # Graph flow proposals  
    with torch.no_grad():
        graph_proposals = graph_model.sample_proposal(
            source_coords=coords,
            atom_types=data["atom_types"][:4],
            adj_list=data["adj_list"],
            edge_batch_idx=data["edge_batch_idx"][:4*21],  # 4 molecules × 21 edges
            masked_elements=data["masked_elements"][:4],
            direction="forward",
        )
    
    # Compute displacement statistics
    simple_displacements = torch.norm(simple_proposals - coords, dim=-1).mean(dim=-1)
    graph_displacements = torch.norm(graph_proposals - coords, dim=-1).mean(dim=-1)
    
    print(f"Simple flow - avg displacement per atom: {simple_displacements.mean():.3f} ± {simple_displacements.std():.3f} Å")
    print(f"Graph flow - avg displacement per atom: {graph_displacements.mean():.3f} ± {graph_displacements.std():.3f} Å")
    
    # Check for diversity in proposals
    simple_diversity = torch.var(simple_displacements).item()
    graph_diversity = torch.var(graph_displacements).item()
    
    print(f"Simple flow - displacement diversity: {simple_diversity:.4f}")
    print(f"Graph flow - displacement diversity: {graph_diversity:.4f}")


def main():
    """Main demonstration function."""
    print("=== PT Swap Flow Architecture Comparison ===")
    print("Comparing two architectures for dissertation:")
    print("1. Simple coordinate-to-coordinate sequential flow")
    print("2. Graph-conditioned flow with molecular structure\n")
    
    # Setup
    torch.manual_seed(42)  # For reproducibility
    simple_config, graph_config = create_demo_config()
    data = generate_molecular_data(batch_size=16)
    
    # Initialize models using config-based architecture selection
    print("Initializing models...")
    
    # Import build_model function from main.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_module", "main.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    # Common parameters for both models
    common_params = {
        "pair": (0, 1),
        "temps": [1.0, 1.5],
        "target_name": "aldp",
        "target_kwargs": {},
        "device": "cpu",
    }
    
    simple_model = main_module.build_model(
        model_cfg=simple_config,
        num_atoms=22,  # Required for simple architecture
        **common_params
    )
    
    graph_model = main_module.build_model(
        model_cfg=graph_config,
        **common_params
    )
    
    print(f"Simple model: {simple_model}")
    print(f"Graph model: {graph_model}")
    
    # Benchmark both architectures
    simple_results = benchmark_architecture(simple_model, data, "Simple Sequential Flow")
    graph_results = benchmark_architecture(graph_model, data, "Graph-Conditioned Flow")
    
    # Compare results
    print("\n=== Architecture Comparison ===")
    print(f"Parameter ratio (Graph/Simple): {graph_results['num_parameters'] / simple_results['num_parameters']:.2f}x")
    print(f"Speed ratio (Graph/Simple): {graph_results['forward_time_ms'] / simple_results['forward_time_ms']:.2f}x")
    print(f"Reconstruction quality (Simple): {simple_results['reconstruction_error']:.2e}")
    print(f"Reconstruction quality (Graph): {graph_results['reconstruction_error']:.2e}")
    
    # Demonstrate swap proposals
    demonstrate_swap_proposals(simple_model, graph_model, data)
    
    # Architecture insights for dissertation
    print("\n=== Insights for Dissertation ===")
    print("Simple Sequential Flow:")
    print("  + Fewer parameters, faster inference")
    print("  + Simpler architecture, easier to analyze")
    print("  - No molecular structure awareness")
    print("  - Fixed atom count, less transferable")
    
    print("\nGraph-Conditioned Flow:")
    print("  + Molecular structure awareness via graph embedding")
    print("  + Distance-based attention captures spatial interactions")
    print("  + Variable molecule sizes supported")
    print("  + Transferable to different peptides")
    print("  - More parameters, slower inference")
    print("  - More complex architecture")
    
    print(f"\nExpected outcome: Graph-conditioned flow should generate")
    print(f"more physically realistic swap proposals due to molecular")
    print(f"structure awareness, leading to higher acceptance rates.")


if __name__ == "__main__":
    main() 