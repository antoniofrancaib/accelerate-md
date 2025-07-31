#!/usr/bin/env python3
"""
Round Trip Performance Analysis for PT Swap Flows

This script loads trained flow models and evaluates their performance
in enhanced parallel tempering simulations by tracking round trips.

Usage:
    python figures/round_trip_analysis.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import yaml
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.accelmd.flows import PTSwapFlow, PTSwapGraphFlow, PTSwapTransformerFlow
from src.accelmd.data import PTTemperaturePairDataset
from src.accelmd.targets import build_target


@dataclass
class PTReplica:
    """A single replica in the parallel tempering simulation."""
    temp_idx: int
    temperature: float
    coordinates: torch.Tensor  # [N, 3]
    replica_id: int  # Track original identity for round trips
    

class EnhancedPTSimulator:
    """Enhanced Parallel Tempering with flow-based swap proposals."""
    
    def __init__(
        self,
        temperatures: List[float],
        initial_coords: torch.Tensor,  # [N, 3]
        atom_types: torch.Tensor,     # [N]
        adj_list: torch.Tensor,       # [E, 2]
        flow_models: Dict[Tuple[int, int], torch.nn.Module],
        target_name: str = "aldp",
        device: str = "cpu"
    ):
        self.temperatures = temperatures
        self.n_temps = len(temperatures)
        self.atom_types = atom_types
        self.adj_list = adj_list
        self.flow_models = flow_models
        self.device = device
        
        # Initialize replicas
        self.replicas = []
        for i, temp in enumerate(temperatures):
            replica = PTReplica(
                temp_idx=i,
                temperature=temp,
                coordinates=initial_coords.clone(),
                replica_id=i  # Track original ID for round trips
            )
            self.replicas.append(replica)
        
        # Build Boltzmann targets for each temperature
        self.targets = {}
        for i, temp in enumerate(temperatures):
            if target_name == "aldp":
                # Use aldp target for standard alanine dipeptide
                self.targets[i] = build_target(
                    "aldp", 
                    temperature=temp, 
                    device="cpu"
                )
            else:
                # Use dipeptide target with PDB file for AA
                pdb_path = "datasets/pt_dipeptides/AA/ref.pdb"
                self.targets[i] = build_target(
                    "dipeptide",
                    temperature=temp,
                    device="cpu",
                    pdb_path=pdb_path,
                    env="implicit"
                )
        
        # Round trip tracking
        self.round_trip_counts = [0] * len(temperatures)
        self.replica_positions = list(range(len(temperatures)))  # Which temp each replica is at
        
    def energy(self, coords: torch.Tensor, temp_idx: int) -> float:
        """Compute potential energy for coordinates at given temperature."""
        coords_flat = coords.view(1, -1)  # [1, N*3]
        log_p = self.targets[temp_idx].log_prob(coords_flat)
        energy = -log_p.item() / self.targets[temp_idx].beta
        return energy
    
    def attempt_swap(self, i: int, j: int, use_flow: bool = True) -> bool:
        """Attempt to swap replicas at temperatures i and j."""
        if abs(i - j) != 1:  # Only adjacent swaps
            return False
            
        replica_i = self.replicas[i]
        replica_j = self.replicas[j]
        
        if use_flow and (i, j) in self.flow_models:
            # Enhanced swap using flow
            flow = self.flow_models[(i, j)]
            
            # Prepare inputs for flow
            coords_i = replica_i.coordinates.unsqueeze(0)  # [1, N, 3]
            coords_j = replica_j.coordinates.unsqueeze(0)  # [1, N, 3]
            
            atom_types_batch = self.atom_types.unsqueeze(0)  # [1, N]
            adj_list_batch = self.adj_list  # [E, 2]
            edge_batch_idx = torch.zeros(self.adj_list.shape[0], dtype=torch.long)  # [E]
            
            try:
                # Forward: low temp → high temp
                if hasattr(flow, 'forward'):
                    if 'PTSwapGraphFlow' in str(type(flow)) or 'PTSwapTransformerFlow' in str(type(flow)):
                        # Graph/Transformer flow interface
                        prop_j, log_det_f = flow.forward(
                            coordinates=coords_i,
                            atom_types=atom_types_batch,
                            adj_list=adj_list_batch,
                            edge_batch_idx=edge_batch_idx,
                            reverse=False
                        )
                        prop_i, log_det_inv = flow.forward(
                            coordinates=coords_j,
                            atom_types=atom_types_batch,
                            adj_list=adj_list_batch,
                            edge_batch_idx=edge_batch_idx,
                            reverse=True
                        )
                    else:
                        # Simple flow interface
                        prop_j, log_det_f = flow.forward(coords_i)
                        prop_i, log_det_inv = flow.inverse(coords_j)
                else:
                    # Fallback to transform method
                    prop_j, log_det_f = flow.transform(coords_i, reverse=False)
                    prop_i, log_det_inv = flow.transform(coords_j, reverse=True)
                
                # Extract coordinates
                prop_i = prop_i.squeeze(0)  # [N, 3]
                prop_j = prop_j.squeeze(0)  # [N, 3]
                log_det = log_det_f.item() + log_det_inv.item()
                
            except Exception as e:
                print(f"Flow failed for swap {i}↔{j}: {e}")
                # Fall back to naive swap
                prop_i = replica_j.coordinates
                prop_j = replica_i.coordinates
                log_det = 0.0
        else:
            # Naive swap
            prop_i = replica_j.coordinates
            prop_j = replica_i.coordinates
            log_det = 0.0
        
        # Compute energies
        E_i_old = self.energy(replica_i.coordinates, i)
        E_j_old = self.energy(replica_j.coordinates, j)
        E_i_new = self.energy(prop_i, i)
        E_j_new = self.energy(prop_j, j)
        
        # Metropolis acceptance
        beta_i = self.targets[i].beta
        beta_j = self.targets[j].beta
        
        delta_E = (beta_i * E_i_new + beta_j * E_j_new) - (beta_i * E_i_old + beta_j * E_j_old)
        log_acc = -delta_E + log_det
        
        if log_acc > 0 or np.random.random() < np.exp(log_acc):
            # Accept swap
            replica_i.coordinates = prop_i
            replica_j.coordinates = prop_j
            
            # Update replica tracking for round trips
            old_pos_i = self.replica_positions[replica_i.replica_id]
            old_pos_j = self.replica_positions[replica_j.replica_id]
            self.replica_positions[replica_i.replica_id] = old_pos_j
            self.replica_positions[replica_j.replica_id] = old_pos_i
            
            return True
        
        return False
    
    def check_round_trips(self):
        """Check for completed round trips and update counts."""
        for replica_id in range(self.n_temps):
            current_pos = self.replica_positions[replica_id]
            # Simple round trip: replica returns to its starting position
            if current_pos == replica_id:
                # Could implement more sophisticated round trip detection here
                pass
    
    def run_simulation(self, n_steps: int, swap_interval: int = 10) -> List[int]:
        """Run PT simulation and return cumulative round trip counts."""
        round_trip_history = []
        total_round_trips = 0
        
        for step in range(n_steps):
            # Attempt swaps between adjacent temperatures
            if step % swap_interval == 0:
                for i in range(self.n_temps - 1):
                    self.attempt_swap(i, i + 1)
            
            # Check for round trips (simplified: count when replica returns to origin)
            if step % (swap_interval * 5) == 0:  # Check every 5 swap attempts
                for replica_id in range(self.n_temps):
                    if self.replica_positions[replica_id] == replica_id:
                        total_round_trips += 1
                        # Reset position tracking for this replica
                        self.replica_positions[replica_id] = -1  # Mark as counted
            
            round_trip_history.append(total_round_trips)
        
        return round_trip_history


def load_model_checkpoint(model_path: str, model_type: str, config: dict) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    if model_type == "simple":
        # Check checkpoint to infer correct parameters
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # Infer num_atoms from mask size (mask has shape [num_atoms * 3])
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        mask_size = state_dict["flow.layers.0.mask"].shape[0]
        num_atoms = mask_size // 3
        
        # Infer hidden_dim from network weights
        hidden_dim = state_dict["flow.layers.0.net._layers.0.bias"].shape[0]
        
        print(f"  Inferred: {num_atoms} atoms, {hidden_dim} hidden_dim")
        
        model = PTSwapFlow(
            num_atoms=num_atoms,
            num_layers=config.get("flow_layers", 8),
            hidden_dim=hidden_dim,
            source_temperature=300.0,
            target_temperature=448.6,
            target_name="aldp",
            device="cpu"
        )
        
        # Load state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    elif model_type == "graph":
        model = PTSwapGraphFlow(
            num_layers=config.get("flow_layers", 8),
            atom_vocab_size=4,
            atom_embed_dim=64,
            hidden_dim=config.get("hidden_dim", 256),
            num_mp_layers=3,
            source_temperature=300.0,
            target_temperature=448.6,
            target_name="aldp",
            device="cpu"
        )
    elif model_type == "transformer":
        model = PTSwapTransformerFlow(
            num_layers=config.get("flow_layers", 8),
            atom_vocab_size=4,
            atom_embed_dim=32,
            transformer_hidden_dim=128,
            source_temperature=300.0,
            target_temperature=448.6,
            target_name="aldp",
            device="cpu"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def load_aa_data():
    """Load AA dataset for testing."""
    data_dir = Path("datasets/pt_dipeptides/AA")
    
    # Load molecular structure
    atom_types = torch.load(data_dir / "atom_types.pt", map_location="cpu", weights_only=False)
    adj_list = torch.load(data_dir / "adj_list.pt", map_location="cpu", weights_only=False)
    
    # Load a sample configuration
    pt_data = torch.load(data_dir / "pt_AA.pt", map_location="cpu", weights_only=False)
    if isinstance(pt_data, dict) and "trajectory" in pt_data:
        traj = pt_data["trajectory"]
    else:
        traj = pt_data
    
    # Get a sample configuration from T0
    if traj.ndim == 4:  # [n_temps, n_chains, n_steps, n_coords]
        sample_coords = traj[0, 0, 0, :].view(-1, 3)  # First sample from T0
    else:
        sample_coords = traj[0, 0, :].view(-1, 3)
    
    return sample_coords, atom_types, adj_list


def create_round_trip_figure(results: Dict[str, List[int]], n_steps: int):
    """Create the round trip performance figure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Time axis (simulation iterations)
    time_points = np.arange(n_steps)
    
    # Colors and styles
    colors = {
        "baseline": "gray",
        "simple": "blue", 
        "graph": "green",
        "transformer": "red"
    }
    
    styles = {
        "baseline": "--",
        "simple": "-",
        "graph": "-", 
        "transformer": "-"
    }
    
    labels = {
        "baseline": "PT Baseline",
        "simple": "Simple Flow",
        "graph": "Graph Flow",
        "transformer": "Transformer Flow"
    }
    
    # Plot lines
    for method, round_trips in results.items():
        ax.plot(
            time_points, 
            round_trips,
            color=colors[method],
            linestyle=styles[method],
            linewidth=2.5 if method != "baseline" else 2.0,
            label=labels[method],
            alpha=0.9
        )
    
    # Calculate and annotate slopes (round trip rates)
    for method, round_trips in results.items():
        if len(round_trips) > 100:
            # Calculate slope over last half of simulation
            x_data = time_points[-len(round_trips)//2:]
            y_data = round_trips[-len(round_trips)//2:]
            if len(y_data) > 1:
                slope = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
                rate_per_1000 = slope * 1000
                
                # Annotate slope
                mid_x = x_data[len(x_data)//2]
                mid_y = y_data[len(y_data)//2]
                ax.annotate(
                    f'{rate_per_1000:.2f}/1000 steps',
                    xy=(mid_x, mid_y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    color=colors[method],
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                )
    
    # Inset box with final counts and improvements
    final_counts = {method: round_trips[-1] if round_trips else 0 for method, round_trips in results.items()}
    baseline_final = final_counts.get("baseline", 1)  # Avoid division by zero
    
    inset_text = "Final Round Trip Counts:\n"
    for method in ["baseline", "simple", "graph", "transformer"]:
        if method in final_counts:
            count = final_counts[method]
            if method == "baseline":
                inset_text += f"PT Baseline: {count}\n"
            else:
                fold_improvement = count / baseline_final if baseline_final > 0 else float('inf')
                inset_text += f"{labels[method]}: {count} ({fold_improvement:.1f}×)\n"
    
    # Add inset box
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    ax.text(0.02, 0.98, inset_text.strip(), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Formatting
    ax.set_xlabel("Simulation Time (iterations)", fontsize=12)
    ax.set_ylabel("Cumulative Round Trips", fontsize=12)
    ax.set_title("Round Trip Performance Over Time", fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    ax.set_xlim(0, n_steps)
    max_trips = max(max(trips) if trips else 0 for trips in results.values())
    ax.set_ylim(0, max_trips * 1.1)
    
    plt.tight_layout()
    return fig


def main():
    """Main analysis script."""
    print("=== Round Trip Performance Analysis ===")
    
    # Model checkpoint paths
    model_paths = {
        "simple": "outputs/AA_simple_01/pair_0_1/models/best_model_epoch992.pt",
        "graph": "outputs/multi_graph_01/pair_0_1/models/best_model_epoch67.pt", 
        "transformer": "outputs/multi_transformer_01/pair_0_1/models/best_model_epoch73.pt"
    }
    
    # Load AA data
    print("Loading AA molecular data...")
    sample_coords, atom_types, adj_list = load_aa_data()
    print(f"Loaded molecule: {len(atom_types)} atoms, {len(adj_list)} bonds")
    
    # Temperature ladder (from config)
    temperatures = [300.0, 448.6046433448792, 670.82040309906, 1003.1104803085328, 1500.0]
    
    # Default config for model loading
    default_config = {
        "flow_layers": 8,
        "hidden_dim": 256
    }
    
    # Load models
    print("Loading trained models...")
    models = {}
    for model_type, path in model_paths.items():
        if Path(path).exists():
            print(f"Loading {model_type} model from {path}")
            try:
                model = load_model_checkpoint(path, model_type, default_config)
                models[model_type] = model
                print(f"✓ Successfully loaded {model_type} model")
            except Exception as e:
                print(f"✗ Failed to load {model_type} model: {e}")
        else:
            print(f"✗ Model checkpoint not found: {path}")
    
    if not models:
        print("No models loaded successfully. Exiting.")
        return
    
    # Simulation parameters
    n_steps = 10000  # Shorter for testing
    swap_interval = 10
    
    print(f"\nRunning simulations ({n_steps} steps each)...")
    
    results = {}
    
    # Baseline PT (no flows)
    print("Running baseline PT simulation...")
    baseline_sim = EnhancedPTSimulator(
        temperatures=temperatures,
        initial_coords=sample_coords,
        atom_types=atom_types,
        adj_list=adj_list,
        flow_models={},  # No flows
        target_name="dipeptide"  # Use dipeptide target for AA
    )
    results["baseline"] = baseline_sim.run_simulation(n_steps, swap_interval)
    print(f"Baseline: {results['baseline'][-1]} round trips")
    
    # Enhanced PT with each flow model
    for model_type, model in models.items():
        print(f"Running {model_type} flow simulation...")
        flow_models = {(0, 1): model}  # Only enhance the 0↔1 swap
        
        enhanced_sim = EnhancedPTSimulator(
            temperatures=temperatures,
            initial_coords=sample_coords,
            atom_types=atom_types,
            adj_list=adj_list,
            flow_models=flow_models,
            target_name="dipeptide"  # Use dipeptide target for AA
        )
        results[model_type] = enhanced_sim.run_simulation(n_steps, swap_interval)
        print(f"{model_type.title()}: {results[model_type][-1]} round trips")
    
    # Create figure
    print("\nCreating round trip performance figure...")
    fig = create_round_trip_figure(results, n_steps)
    
    # Save figure
    output_path = Path("figures/round_trip_performance.png")
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    # Show figure
    plt.show()
    
    print("=== Analysis Complete ===")


if __name__ == "__main__":
    main() 