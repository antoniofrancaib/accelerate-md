#!/usr/bin/env python3
"""
Round-trip comparison between vanilla PT, simple-flow PT, and transformer-flow PT.

This script runs three parallel tempering simulations:
1. Vanilla PT - standard replica exchange
2. Simple-flow PT - using RealNVP flows for swap proposals  
3. Transformer-flow PT - using transformer flows for swap proposals

Each flow-enhanced PT uses the appropriate trained model for each temperature pair.
The output is a plot showing cumulative round-trips vs steps for all three methods.
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.accelmd.samplers.pt.sampler import ParallelTempering
from src.accelmd.samplers.pt.dyn_wrapper import DynSamplerWrapper
from src.accelmd.targets.dipeptide_potential import DipeptidePotentialCart
from src.accelmd.flows.pt_swap_flow import PTSwapFlow
from src.accelmd.flows.pt_swap_transformer_flow import PTSwapTransformerFlow
from src.accelmd.utils.config import load_config


class RoundTripTracker:
    """Simple round-trip tracker for PT simulations."""
    
    def __init__(self, n_temps: int, n_chains: int):
        self.n_temps = n_temps
        self.n_chains = n_chains
        self.round_trips = 0  # Total round trips across all chains
        self.chain_positions = torch.zeros(n_chains, dtype=torch.long)  # Which temp each chain is at
        self.chain_max_reached = torch.zeros(n_chains, dtype=torch.long)  # Max temp reached per chain
        
    def update_after_swap(self, temp_i: int, temp_j: int, swapped_chains: torch.Tensor):
        """Update tracking after swaps between temperatures i and j."""
        # Handle case where swapped_chains might be shorter than n_chains
        n_swapped = min(len(swapped_chains), self.n_chains)
        
        # Update chain positions for swapped chains
        for chain_idx in range(n_swapped):
            if swapped_chains[chain_idx]:
                # This chain was swapped between temp_i and temp_j
                if self.chain_positions[chain_idx] == temp_i:
                    self.chain_positions[chain_idx] = temp_j
                elif self.chain_positions[chain_idx] == temp_j:
                    self.chain_positions[chain_idx] = temp_i
        
        # Update max temperature reached and count round trips
        for chain_idx in range(self.n_chains):
            current_temp = self.chain_positions[chain_idx]
            
            # Update max temp reached
            if current_temp > self.chain_max_reached[chain_idx]:
                self.chain_max_reached[chain_idx] = current_temp
            
            # Count round trip if back to temp 0 after visiting higher temps
            if current_temp == 0 and self.chain_max_reached[chain_idx] >= self.n_temps - 1:
                self.round_trips += 1
                self.chain_max_reached[chain_idx] = 0  # Reset for next round trip


class FlowEnhancedPT:
    """Parallel tempering with flow-enhanced swap proposals."""
    
    def __init__(
        self,
        base_pt,  # Could be DynSamplerWrapper or ParallelTempering
        flow_models: Dict[Tuple[int, int], torch.nn.Module],
        temperatures: torch.Tensor,
        device: str = "cpu"
    ):
        self.base_pt = base_pt
        self.flow_models = flow_models
        self.temperatures = temperatures
        self.device = device
        
        # Access the underlying sampler if wrapped
        if hasattr(base_pt, 'sampler'):
            self.sampler = base_pt.sampler
        else:
            self.sampler = base_pt
            
        # Initialize round-trip tracking
        n_temps, n_chains = self.sampler.x.shape[:2]
        self.round_trip_tracker = RoundTripTracker(n_temps, n_chains)
        
    def sample(self):
        """Perform one PT step with flow-enhanced swaps."""
        # Do the MD/MCMC steps first
        new_samples, acc, *_ = self.base_pt.sample()
        
        # Then override swaps with flow-enhanced proposals
        # We need to intercept the swap step and replace it
        self._flow_enhanced_swap()
        
        return new_samples, acc
    
    def _flow_enhanced_swap(self):
        """Perform flow-enhanced replica swaps."""
        n_temp = len(self.temperatures)
        
        # Try swaps between adjacent temperatures
        for i in range(n_temp - 1):
            temp_pair = (i, i + 1)
            
            if temp_pair not in self.flow_models:
                # Fallback to vanilla swap if no model available
                accept_mask = self._vanilla_swap(i, i + 1)
                self.round_trip_tracker.update_after_swap(i, i + 1, accept_mask)
                continue
                
            flow_model = self.flow_models[temp_pair]
            accept_mask = self._flow_swap(flow_model, i, i + 1)
            self.round_trip_tracker.update_after_swap(i, i + 1, accept_mask)
    
    def _vanilla_swap(self, temp_i: int, temp_j: int) -> torch.Tensor:
        """Perform vanilla replica swap between two temperatures."""
        beta_i = 1.0 / self.temperatures[temp_i]
        beta_j = 1.0 / self.temperatures[temp_j]
        
        # Get coordinates
        x_i = self.sampler.x[temp_i]  # [n_chains, dim] or [dim]
        x_j = self.sampler.x[temp_j]  # [n_chains, dim] or [dim]
        
        # Handle the case where coordinates might be flattened
        if x_i.ndim == 1:
            x_i = x_i.unsqueeze(0)
            x_j = x_j.unsqueeze(0)
        
        # Compute energies
        U_i = -self.sampler.energy_func(x_i)
        U_j = -self.sampler.energy_func(x_j)
        
        # Vanilla swap acceptance
        log_acc = (beta_i - beta_j) * (U_i - U_j)
        acc_prob = torch.minimum(torch.ones_like(log_acc), torch.exp(log_acc))
        
        # Accept/reject each chain independently
        accept_mask = torch.rand_like(acc_prob) < acc_prob
        
        # Apply accepted swaps
        for chain_idx in range(x_i.shape[0]):
            if accept_mask[chain_idx]:
                # Swap coordinates
                temp = self.sampler.x[temp_i, chain_idx].clone()
                self.sampler.x[temp_i, chain_idx] = self.sampler.x[temp_j, chain_idx]
                self.sampler.x[temp_j, chain_idx] = temp
        
        return accept_mask
    
    def _flow_swap(self, flow_model: torch.nn.Module, temp_i: int, temp_j: int) -> torch.Tensor:
        """Perform flow-enhanced swap between two temperatures."""
        beta_low = 1.0 / self.temperatures[temp_i]
        beta_high = 1.0 / self.temperatures[temp_j]
        
        # Get coordinates for this temperature pair
        x_low = self.sampler.x[temp_i]   # Should be [n_chains, dim]
        x_high = self.sampler.x[temp_j]  # Should be [n_chains, dim]
        
        # Handle the case where coordinates might be flattened
        if x_low.ndim == 1:
            # Reshape to [1, dim] if it's a single chain
            x_low = x_low.unsqueeze(0)
            x_high = x_high.unsqueeze(0)
        
        # Ensure correct dtype (flows expect float32)
        x_low = x_low.float()
        x_high = x_high.float()
        
        n_chains = x_low.shape[0]
        n_atoms = x_low.shape[1] // 3
        
        # Reshape to [n_chains, n_atoms, 3] for flow
        x_low_3d = x_low.view(n_chains, n_atoms, 3)
        x_high_3d = x_high.view(n_chains, n_atoms, 3)
        
        try:
            if isinstance(flow_model, PTSwapTransformerFlow):
                # Need atom types for transformer flow
                atom_types = self._get_atom_types(n_chains, n_atoms)
                
                y_high, log_det_f = flow_model.forward(
                    coordinates=x_low_3d,
                    atom_types=atom_types,
                    reverse=False,
                    return_log_det=True
                )
                y_low, log_det_inv = flow_model.forward(
                    coordinates=x_high_3d,
                    atom_types=atom_types,
                    reverse=True,
                    return_log_det=True
                )
            else:
                # Simple flow - need to handle return values correctly
                flow_result_f = flow_model.forward(x_low_3d)
                flow_result_inv = flow_model.inverse(x_high_3d)
                
                if isinstance(flow_result_f, tuple):
                    y_high, log_det_f = flow_result_f
                else:
                    y_high = flow_result_f
                    log_det_f = torch.zeros(n_chains, device=self.device)
                
                if isinstance(flow_result_inv, tuple):
                    y_low, log_det_inv = flow_result_inv
                else:
                    y_low = flow_result_inv
                    log_det_inv = torch.zeros(n_chains, device=self.device)
            
            # Reshape back to flat coordinates
            y_high_flat = y_high.view(n_chains, -1)
            y_low_flat = y_low.view(n_chains, -1)
            
            # Compute energies
            U_x_low = -self.sampler.energy_func(x_low)
            U_x_high = -self.sampler.energy_func(x_high)
            U_y_low = -self.sampler.energy_func(y_low_flat)
            U_y_high = -self.sampler.energy_func(y_high_flat)
            
            # Flow-enhanced acceptance probability
            log_acc = (
                -beta_low * U_y_low - beta_high * U_y_high +
                beta_low * U_x_low + beta_high * U_x_high +
                log_det_f + log_det_inv
            )
            
            acc_prob = torch.minimum(torch.ones_like(log_acc), torch.exp(log_acc))
            
            # Accept/reject each chain independently
            accept_mask = torch.rand_like(acc_prob) < acc_prob
            
            # Apply accepted swaps
            for chain_idx in range(n_chains):
                if accept_mask[chain_idx]:
                    # Swap coordinates
                    self.sampler.x[temp_i, chain_idx] = y_low_flat[chain_idx]
                    self.sampler.x[temp_j, chain_idx] = y_high_flat[chain_idx]
            
            return accept_mask
            
        except Exception as e:
            print(f"Flow swap failed for pair ({temp_i}, {temp_j}), falling back to vanilla: {e}")
            # Fallback to vanilla swap
            return self._vanilla_swap(temp_i, temp_j)
    
    def _get_atom_types(self, n_chains: int, n_atoms: int) -> torch.Tensor:
        """Generate atom types matching the AA dipeptide pattern."""
        # AA dipeptide atom types from the actual dataset:
        # tensor([0, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 3, 0, 1, 2, 1, 2, 1, 1, 1, 2, 3, 3])
        if n_atoms == 23:  # AA dipeptide
            aa_pattern = torch.tensor([0, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 3, 0, 1, 2, 1, 2, 1, 1, 1, 2, 3, 3], dtype=torch.long)
            atom_types = aa_pattern.unsqueeze(0).repeat(n_chains, 1).to(self.device)
        else:
            # Fallback pattern for other peptides
            atom_types = torch.zeros(n_chains, n_atoms, dtype=torch.long, device=self.device)
            for i in range(n_atoms):
                if i % 4 == 0:
                    atom_types[:, i] = 1  # C
                elif i % 4 == 1:
                    atom_types[:, i] = 0  # H  
                elif i % 4 == 2:
                    atom_types[:, i] = 2  # N
                else:
                    atom_types[:, i] = 3  # O
                
        return atom_types


def load_flow_models(model_paths: Dict[str, Dict[Tuple[int, int], str]], peptide: str = "AA", device: str = "cpu"):
    """Load flow models for each temperature pair."""
    models = {"simple": {}, "transformer": {}}
    
    # Load simple flow models
    for pair, path in model_paths["simple"].items():
        if os.path.exists(path):
            print(f"Loading simple flow model for pair {pair}: {path}")
            
            # Create simple flow model (matching AA_simple_01.yaml config)
            model = PTSwapFlow(
                num_atoms=23,  # AA dipeptide has 23 atoms
                num_layers=8,
                hidden_dim=512,  # From config: hidden_dim: 512
                source_temperature=1.0,  # Will be updated based on pair
                target_temperature=1.5,
                target_name="dipeptide",
                target_kwargs={
                    "pdb_path": f"datasets/pt_dipeptides/{peptide}/ref.pdb",
                    "env": "implicit"
                },
                device=device
            )
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint)
            model.eval()
            
            models["simple"][pair] = model
        else:
            print(f"Warning: Simple flow model not found: {path}")
    
    # Load transformer flow models  
    for pair, path in model_paths["transformer"].items():
        if os.path.exists(path):
            print(f"Loading transformer flow model for pair {pair}: {path}")
            
            # Create transformer flow model
            model = PTSwapTransformerFlow(
                num_layers=8,
                atom_vocab_size=4,
                atom_embed_dim=32,
                transformer_hidden_dim=128,
                mlp_hidden_layer_dims=[128, 128],
                num_transformer_layers=2,
                source_temperature=1.0,  # Will be updated based on pair
                target_temperature=1.5,
                target_name="dipeptide",
                target_kwargs={
                    "pdb_path": f"datasets/pt_dipeptides/{peptide}/ref.pdb",
                    "env": "implicit"
                },
                device=device
            )
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint)
            model.eval()
            
            models["transformer"][pair] = model
        else:
            print(f"Warning: Transformer flow model not found: {path}")
    
    return models


def run_vanilla_pt(config: Dict) -> List[int]:
    """Run vanilla parallel tempering."""
    print("Running vanilla PT...")
    
    # Set up target and initial coordinates
    target = DipeptidePotentialCart(
        pdb_path=f"datasets/pt_dipeptides/{config['peptide']}/ref.pdb",
        n_threads=1,
        device=config["device"]
    )
    
    # Get minimized initial coordinates
    state = target.context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    pos_array = np.array([[v.x, v.y, v.z] for v in minimized_positions])
    
    x_init = torch.tensor(pos_array, device=config["device"]).view(1, -1)
    x_init = x_init.unsqueeze(0).repeat(config["total_n_temp"], config["num_chains"], 1)
    
    # Set up PT sampler
    pt = ParallelTempering(
        x=x_init,
        energy_func=lambda x: -target.log_prob(x),
        step_size=torch.tensor([config["step_size"]] * (config["total_n_temp"] * config["num_chains"]), 
                              device=config["device"]).unsqueeze(-1),
        swap_interval=config["swap_interval"],
        temperatures=config["temperatures"],
        mh=True,
        device=config["device"]
    )
    
    pt_wrapped = DynSamplerWrapper(pt, per_temp=True, total_n_temp=config["total_n_temp"], 
                                  target_acceptance_rate=0.6, alpha=0.25)
    
    # Set up round trip tracking
    tracker = RoundTripTracker(config["total_n_temp"], config["num_chains"])
    round_trip_history = []
    
    # Run simulation
    progress_bar = tqdm(range(config["num_steps"]), desc="Vanilla PT")
    for i in progress_bar:
        new_samples, acc, *_ = pt_wrapped.sample()
        
        # Extract swap information from the sampler
        # For vanilla PT, we simulate swaps by checking swap rates
        if hasattr(pt, 'swap_rates') and pt.swap_rates:
            for temp_i in range(len(pt.swap_rates)):
                # Simulate successful swaps based on swap rate
                n_swaps = int(pt.swap_rates[temp_i] * config["num_chains"])
                if n_swaps > 0:
                    # Create mock swap mask
                    swap_mask = torch.zeros(config["num_chains"], dtype=torch.bool)
                    swap_mask[:n_swaps] = True
                    tracker.update_after_swap(temp_i, temp_i + 1, swap_mask)
        
        round_trip_history.append(tracker.round_trips)
        
        if i % 1000 == 0:
            progress_bar.set_postfix_str(f"Round trips: {tracker.round_trips}")
    
    return round_trip_history


def run_flow_enhanced_pt(config: Dict, flow_models: Dict[Tuple[int, int], torch.nn.Module], 
                        method_name: str) -> List[int]:
    """Run flow-enhanced parallel tempering."""
    print(f"Running {method_name} PT...")
    
    # Set up target and initial coordinates (same as vanilla)
    target = DipeptidePotentialCart(
        pdb_path=f"datasets/pt_dipeptides/{config['peptide']}/ref.pdb",
        n_threads=1,
        device=config["device"]
    )
    
    state = target.context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    pos_array = np.array([[v.x, v.y, v.z] for v in minimized_positions])
    
    x_init = torch.tensor(pos_array, device=config["device"]).view(1, -1)
    x_init = x_init.unsqueeze(0).repeat(config["total_n_temp"], config["num_chains"], 1)
    
    # Set up base PT sampler
    base_pt = ParallelTempering(
        x=x_init,
        energy_func=lambda x: -target.log_prob(x),
        step_size=torch.tensor([config["step_size"]] * (config["total_n_temp"] * config["num_chains"]), 
                              device=config["device"]).unsqueeze(-1),
        swap_interval=config["swap_interval"],
        temperatures=config["temperatures"],
        mh=True,
        device=config["device"]
    )
    
    base_pt = DynSamplerWrapper(base_pt, per_temp=True, total_n_temp=config["total_n_temp"], 
                               target_acceptance_rate=0.6, alpha=0.25)
    
    # Wrap with flow enhancement
    flow_pt = FlowEnhancedPT(base_pt, flow_models, config["temperatures"], config["device"])
    
    round_trip_history = []
    
    # Run simulation
    progress_bar = tqdm(range(config["num_steps"]), desc=f"{method_name} PT")
    for i in progress_bar:
        new_samples, acc, *_ = flow_pt.sample()
        
        round_trip_history.append(flow_pt.round_trip_tracker.round_trips)
        
        if i % 1000 == 0:
            progress_bar.set_postfix_str(f"Round trips: {flow_pt.round_trip_tracker.round_trips}")
    
    return round_trip_history


def plot_round_trip_comparison(vanilla_history: List[int], simple_history: List[int], 
                             transformer_history: List[int], output_path: str = "round_trip_comparison.png"):
    """Plot the round-trip comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = np.arange(len(vanilla_history))
    
    ax.plot(steps, transformer_history, 'k-', linewidth=2, label='Transformer-flow PT')
    ax.plot(steps, simple_history, 'b-', linewidth=2, label='Simple-flow PT')  
    ax.plot(steps, vanilla_history, 'r-', linewidth=2, label='Vanilla PT')
    
    ax.set_xlabel('Steps', fontsize=14)
    ax.set_ylabel('Number of round trips', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {output_path}")


def main():
    """Main execution function."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    config = {
        "peptide": "AA",
        "temp_schedule": "geom",
        "temp_low": 1.0,
        "temp_high": 5.0,
        "total_n_temp": 5,
        "num_chains": 10,
        "num_steps": 1000,  # Short test run for debugging
        "step_size": 0.0001,
        "swap_interval": 100,
        "device": device
    }
    
    # Set up temperatures
    if config["temp_schedule"] == 'geom':
        temperatures = torch.from_numpy(
            np.geomspace(config["temp_low"], config["temp_high"], config["total_n_temp"])
        ).float().to(device)
    else:
        temperatures = torch.linspace(
            config["temp_low"], config["temp_high"], config["total_n_temp"]
        ).float().to(device)
    
    config["temperatures"] = temperatures
    
    # Define model paths
    model_paths = {
        "simple": {
            (0, 1): "outputs/AA_simple/pair_0_1/models/best_model_epoch2986.pt",
            (1, 2): "outputs/AA_simple/pair_1_2/models/best_model_epoch929.pt", 
            (2, 3): "outputs/AA_simple/pair_2_3/models/best_model_epoch925.pt",
            (3, 4): "outputs/AA_simple/pair_3_4/models/best_model_epoch926.pt"
        },
        "transformer": {
            (0, 1): "outputs/multi_transformer/pair_0_1/models/best_model_epoch48.pt",
            (1, 2): "outputs/multi_transformer/pair_1_2/models/best_model_epoch94.pt",
            (2, 3): "outputs/multi_transformer/pair_2_3/models/best_model_epoch70.pt", 
            (3, 4): "outputs/multi_transformer/pair_3_4/models/best_model_epoch84.pt"
        }
    }
    
    # Load flow models
    models = load_flow_models(model_paths, config["peptide"], device)
    
    # Run all three PT simulations
    print("Starting round-trip comparison...")
    
    # 1. Vanilla PT
    vanilla_history = run_vanilla_pt(config)
    
    # 2. Simple-flow PT
    simple_history = run_flow_enhanced_pt(config, models["simple"], "Simple-flow")
    
    # 3. Transformer-flow PT
    transformer_history = run_flow_enhanced_pt(config, models["transformer"], "Transformer-flow")
    
    # Plot comparison
    plot_round_trip_comparison(vanilla_history, simple_history, transformer_history,
                             "experiments/RTR_rama/round_trip_comparison.png")
    
    # Print final statistics
    print(f"\nFinal round-trip counts after {config['num_steps']} steps:")
    print(f"Vanilla PT: {vanilla_history[-1]}")
    print(f"Simple-flow PT: {simple_history[-1]}")
    print(f"Transformer-flow PT: {transformer_history[-1]}")
    
    # Calculate speedup
    if vanilla_history[-1] > 0:
        simple_speedup = simple_history[-1] / vanilla_history[-1]
        transformer_speedup = transformer_history[-1] / vanilla_history[-1]
        print(f"\nSpeedup vs vanilla:")
        print(f"Simple-flow PT: {simple_speedup:.2f}x")
        print(f"Transformer-flow PT: {transformer_speedup:.2f}x")


if __name__ == "__main__":
    main()