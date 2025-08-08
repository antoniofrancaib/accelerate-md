#!/usr/bin/env python3
"""Flow-enhanced Parallel Tempering simulation with Ramachandran analysis.

This script runs PT simulations using trained normalizing flows for enhanced
swap proposals, generating Ramachandran plots at regular intervals.

Usage:
    conda activate accelmd && python experiments/rama/run_pt_flow_simulation.py

Edit the config dictionary at the bottom to change:
- peptide name ("AA", "AK", etc.)
- architecture ("simple", "graph", "transformer") 
- simulation parameters (steps, temperatures, etc.)
"""

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.accelmd.samplers.pt.sampler import ParallelTempering
from src.accelmd.samplers.pt.dyn_wrapper import DynSamplerWrapper
from src.accelmd.targets.dipeptide_potential import DipeptidePotentialCart
from src.accelmd.utils.plot_utils import plot_Ramachandran
from src.accelmd.flows import PTSwapFlow, PTSwapGraphFlow, PTSwapTransformerFlow
from src.accelmd.flows.transformer_block import TransformerConfig
from src.accelmd.flows.rff_position_encoder import RFFPositionEncoderConfig
import mdtraj as md


class FlowParallelTempering(ParallelTempering):
    """PT sampler with flow-enhanced swap proposals."""
    
    def __init__(self, flow_dict, target, molecular_data_path, *args, **kwargs):
        """
        Parameters
        ----------
        flow_dict : dict
            Maps temperature pair tuples (i,j) to trained flow models
        target : DipeptidePotentialCart
            Boltzmann target for energy evaluation
        molecular_data_path : str
            Path to molecular data directory containing atom_types.pt
        """
        super().__init__(*args, **kwargs)
        self.flows = flow_dict
        self.target = target
        
        # Load molecular data for flows that need it
        self._load_molecular_data(molecular_data_path)
    
    def _load_molecular_data(self, molecular_data_path):
        """Load atom types for flow models."""
        from pathlib import Path
        
        # Load atom types
        atom_types_path = Path(molecular_data_path) / "atom_types.pt"
        if atom_types_path.exists():
            try:
                self.atom_types = torch.load(atom_types_path, map_location="cpu", weights_only=True)
            except Exception:
                self.atom_types = torch.load(atom_types_path, map_location="cpu")
        else:
            # Create default atom types if file doesn't exist
            # Assume all atoms are carbon (type 1) for now
            n_atoms = self.target.n_atoms
            self.atom_types = torch.ones(n_atoms, dtype=torch.long)
            print(f"Warning: atom_types.pt not found, using default carbon atoms")
        
    def _attempt_swap(self, idx_a, idx_b):
        """Enhanced swap with flow proposals when available."""
        temp_a, temp_b = self.temperatures[idx_a], self.temperatures[idx_b]
        
        # Get chain slices for each temperature
        chains_per_temp = self.x.shape[0] // self.num_temperatures
        slice_a = slice(idx_a * chains_per_temp, (idx_a + 1) * chains_per_temp)
        slice_b = slice(idx_b * chains_per_temp, (idx_b + 1) * chains_per_temp)
        
        chain_a = self.x[slice_a]  # [n_chains, dim]
        chain_b = self.x[slice_b]  # [n_chains, dim]
        
        # Check if we have a flow for this temperature pair
        pair = (min(idx_a, idx_b), max(idx_a, idx_b))
        pair_key = f"pair_{pair[0]}_{pair[1]}"
        
        if pair_key in self.flows and self.flows[pair_key] is not None:
            # Flow-enhanced swap
            flow = self.flows[pair_key]
            device = next(flow.parameters()).device
            
            # Reshape to [B, N, 3] and move to flow device
            n_atoms = self.target.n_atoms
            chain_a_3d = chain_a.view(-1, n_atoms, 3).float().to(device)
            chain_b_3d = chain_b.view(-1, n_atoms, 3).float().to(device)
            
            # Apply flow transformations
            with torch.no_grad():
                # Prepare atom types for all chains in batch
                n_chains = chain_a_3d.shape[0]
                atom_types_batch = self.atom_types.unsqueeze(0).repeat(n_chains, 1).to(device)
                
                # Forward: low temp -> high temp proposal
                if idx_a < idx_b:  # a is lower temperature
                    proposal_b, log_det_f = flow.forward(
                        chain_a_3d, atom_types=atom_types_batch, return_log_det=True
                    )
                    proposal_a, log_det_inv = flow.inverse(
                        chain_b_3d, atom_types=atom_types_batch
                    )
                else:  # b is lower temperature  
                    proposal_a, log_det_f = flow.forward(
                        chain_b_3d, atom_types=atom_types_batch, return_log_det=True
                    )
                    proposal_b, log_det_inv = flow.inverse(
                        chain_a_3d, atom_types=atom_types_batch
                    )
                
                # Reshape back and move to CPU for energy evaluation
                proposal_a = proposal_a.view(-1, chain_a.shape[-1]).cpu()
                proposal_b = proposal_b.view(-1, chain_b.shape[-1]).cpu()
                
                # Compute energies using unscaled energy function
                energy_a = self.base_energy(chain_a)
                energy_b = self.base_energy(chain_b)
                energy_prop_a = self.base_energy(proposal_a)
                energy_prop_b = self.base_energy(proposal_b)
                
                # Flow-enhanced acceptance probability (Boltzmann-Generator formula)
                log_accept = (
                    (1.0 / temp_a - 1.0 / temp_b) * (energy_prop_b - energy_prop_a) +
                    (1.0 / temp_b - 1.0 / temp_a) * (energy_b - energy_a) +
                    log_det_f.cpu() + log_det_inv.cpu()
                )
                
                new_chain_a = proposal_a
                new_chain_b = proposal_b
                
        else:
            # Vanilla swap (no flow available)
            energy_a = self.base_energy(chain_a)
            energy_b = self.base_energy(chain_b)
            log_accept = (1.0 / temp_a - 1.0 / temp_b) * (energy_b - energy_a)
            new_chain_a = chain_b  # Standard coordinate exchange
            new_chain_b = chain_a
        
        # Accept or reject
        accept_prob = torch.minimum(torch.ones_like(log_accept), log_accept.exp())
        accept = (torch.rand_like(log_accept) <= accept_prob).unsqueeze(-1)
        
        # Apply swaps where accepted
        self.x[slice_a] = torch.where(accept, new_chain_a, chain_a)
        self.x[slice_b] = torch.where(accept, new_chain_b, chain_b)
        
        return accept_prob.mean().item()


def load_experiment_config():
    """Load checkpoint paths from centralized config."""
    config_path = project_root / "configs" / "experiments.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_flows(architecture, exp_config, model_config, target_name, pdb_path, temperatures, device):
    """Load trained flow models for each temperature pair."""
    flows = {}
    
    # Get checkpoint paths
    if architecture == "transformer":
        ckpt_section = exp_config["checkpoints"]["transformer"]["multi_peptide"]
    elif architecture == "graph":
        ckpt_section = exp_config["checkpoints"]["graph"]["multi_peptide"] 
    elif architecture == "simple":
        # For simple architecture, we'd need peptide-specific checkpoints
        # For now, assume multi_peptide section exists
        ckpt_section = exp_config["checkpoints"]["simple"].get("multi_peptide", {})
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Load each flow
    for pair_key, ckpt_path in ckpt_section.items():
        if ckpt_path is None:
            print(f"No checkpoint for {pair_key}, skipping...")
            continue
            
        # Parse pair indices
        pair_indices = tuple(map(int, pair_key.split("_")[1:]))
        i, j = pair_indices
        
        # Build flow model
        target_kwargs = {
            "pdb_path": pdb_path,
            "env": "implicit"
        }
        
        if architecture == "simple":
            # Determine number of atoms from target
            temp_target = DipeptidePotentialCart(pdb_path=pdb_path, device="cpu")
            n_atoms = temp_target.n_atoms
            
            flow = PTSwapFlow(
                num_atoms=n_atoms,
                num_layers=model_config["flow_layers"],
                hidden_dim=model_config["hidden_dim"],
                source_temperature=temperatures[i].item(),
                target_temperature=temperatures[j].item(),
                target_name=target_name,
                target_kwargs=target_kwargs,
                device=device,
            )
            
        elif architecture == "graph":
            graph_cfg = model_config.get("graph", {})
            flow = PTSwapGraphFlow(
                num_layers=model_config["flow_layers"],
                atom_vocab_size=graph_cfg.get("atom_vocab_size", 4),
                atom_embed_dim=graph_cfg.get("atom_embed_dim", 32),
                hidden_dim=graph_cfg.get("hidden_dim", model_config["hidden_dim"]),
                source_temperature=temperatures[i].item(),
                target_temperature=temperatures[j].item(),
                target_name=target_name,
                target_kwargs=target_kwargs,
                device=device,
            )
            
        elif architecture == "transformer":
            transformer_cfg = model_config.get("transformer", {})
            
            transformer_config = TransformerConfig(
                n_head=transformer_cfg.get("n_head", 8),
                dim_feedforward=transformer_cfg.get("dim_feedforward", 2048),
                dropout=0.0,
            )
            
            rff_config = RFFPositionEncoderConfig(
                encoding_dim=transformer_cfg.get("rff_encoding_dim", 64),
                scale_mean=transformer_cfg.get("rff_scale_mean", 1.0),
                scale_stddev=transformer_cfg.get("rff_scale_stddev", 1.0),
            )
            
            flow = PTSwapTransformerFlow(
                num_layers=model_config["flow_layers"],
                atom_vocab_size=transformer_cfg.get("atom_vocab_size", 4),
                atom_embed_dim=transformer_cfg.get("atom_embed_dim", 32),
                transformer_hidden_dim=transformer_cfg.get("transformer_hidden_dim", 128),
                mlp_hidden_layer_dims=transformer_cfg.get("mlp_hidden_layer_dims", [128, 128]),
                num_transformer_layers=transformer_cfg.get("num_transformer_layers", 2),
                source_temperature=temperatures[i].item(),
                target_temperature=temperatures[j].item(),
                target_name=target_name,
                target_kwargs=target_kwargs,
                transformer_config=transformer_config,
                rff_position_encoder_config=rff_config,
                device=device,
            )
        
        # Load checkpoint
        ckpt_full_path = project_root / ckpt_path
        if not ckpt_full_path.exists():
            print(f"Checkpoint not found: {ckpt_full_path}")
            continue
            
        state_dict = torch.load(ckpt_full_path, map_location=device)
        flow.load_state_dict(state_dict)
        flow.eval()
        
        flows[pair_key] = flow
        print(f"Loaded {architecture} flow for {pair_key}")
    
    return flows


def main(config):
    """Run flow-enhanced PT simulation with Ramachandran analysis."""
    print(f"Starting flow-enhanced PT simulation for {config['name']} using {config['architecture']} architecture")
    
    # Load experiment configuration
    exp_config = load_experiment_config()
    
    # Load model configuration
    model_config_path = project_root / exp_config["configs"][config["architecture"]]["config_file"]
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    
    # Set up temperatures
    if config["temp_schedule"] == 'geom':
        temperatures = torch.from_numpy(
            np.geomspace(config["temp_low"], config["temp_high"], config["total_n_temp"])
        ).float()
    elif config["temp_schedule"] == 'linear':
        temperatures = torch.linspace(
            config["temp_low"], config["temp_high"], config["total_n_temp"]
        ).float()
    
    # Set up directories
    base_plot_path = f"experiments/rama/plots/{config['name']}_{config['architecture']}"
    os.makedirs(base_plot_path, exist_ok=True)
    
    temp_plot_paths = {}
    for i, temp in enumerate(temperatures):
        temp_dir = f"{base_plot_path}/{temp.item():.2f}"
        os.makedirs(temp_dir, exist_ok=True)
        temp_plot_paths[i] = temp_dir
    
    # Set up target and initial coordinates
    pdb_path = f"datasets/pt_dipeptides/{config['name']}/ref.pdb"
    target = DipeptidePotentialCart(
        pdb_path=pdb_path,
        n_threads=1,
        device="cpu"  # Keep target on CPU for compatibility
    )
    
    # Get minimized initial coordinates
    state = target.context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    pos_array = np.array([[v.x, v.y, v.z] for v in minimized_positions])
    
    x_init = torch.tensor(pos_array, device="cpu").view(1, -1)
    x_init = x_init.unsqueeze(0).repeat(config["total_n_temp"], config["num_chains"], 1)
    
    # Load flows
    device = "cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu"
    flows = load_flows(
        architecture=config["architecture"],
        exp_config=exp_config,
        model_config=model_config["model"],
        target_name="dipeptide",
        pdb_path=pdb_path,
        temperatures=temperatures,
        device=device
    )
    
    print(f"Loaded {len(flows)} flows, running on {device}")
    
    # Create PT sampler with flows
    molecular_data_path = f"datasets/pt_dipeptides/{config['name']}"
    pt = FlowParallelTempering(
        flow_dict=flows,
        target=target,
        molecular_data_path=molecular_data_path,
        x=x_init,
        energy_func=lambda x: -target.log_prob(x),
        step_size=torch.tensor([config["step_size"]] * (config["total_n_temp"] * config["num_chains"])).unsqueeze(-1),
        swap_interval=config["swap_interval"],
        temperatures=temperatures,
        mh=True,
        device="cpu"  # Keep sampler on CPU
    )
    
    # Wrap with dynamic step size adjustment
    pt = DynSamplerWrapper(
        pt, 
        per_temp=True, 
        total_n_temp=config["total_n_temp"], 
        target_acceptance_rate=0.6, 
        alpha=0.25
    )
    
    # Evaluation function for Ramachandran plots
    def eval_fn(coords, plot_path):
        traj = md.Trajectory(
            coords.view(-1, target.n_atoms, 3).detach().cpu().numpy(),
            topology=md.Topology.from_openmm(target.topology)
        )
        phi = md.compute_phi(traj)[1].flatten()
        psi = md.compute_psi(traj)[1].flatten()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_Ramachandran(ax, phi, psi)
        ax.set_title(f"{config['name']} - {config['architecture']} flow")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        return fig
    
    # Main simulation loop
    progress_bar = tqdm(range(config["num_steps"]), desc="Flow-Enhanced PT")
    traj = []
    swap_rates = []
    
    for i in progress_bar:
        # Sample one step
        new_samples, acc, *_ = pt.sample()
        traj.append(new_samples.clone().detach().cpu().float())
        
        # Log swap rates
        if pt.sampler.swap_rates:
            swap_rates.append(pt.sampler.swap_rates)
        
        # Generate plots at checkpoints
        if i == 0 or (i + 1) % config["check_interval"] == 0:
            # Save trajectory checkpoint
            os.makedirs("experiments/rama/results", exist_ok=True)
            torch.save(
                torch.stack(traj, dim=2).detach().cpu().float(),
                f"experiments/rama/results/pt_{config['name']}_{config['architecture']}.pt"
            )
            
            # Generate Ramachandran plots for all temperatures
            recent_traj = torch.stack(traj[-config["check_interval"]:], dim=2)
            dim = new_samples.shape[-1]
            
            for temp_idx in range(config["total_n_temp"]):
                fig = eval_fn(
                    recent_traj[temp_idx].reshape(-1, dim),
                    f'{temp_plot_paths[temp_idx]}/{i + 1}.png'
                )
                plt.close(fig)
        
        # Update progress bar
        progress_bar.set_postfix_str(f"acc rate: {acc.mean().item():.3f}")
    
    # Save final trajectory
    traj = torch.stack(traj, dim=2)
    os.makedirs("experiments/rama/results", exist_ok=True)
    torch.save(
        traj.detach().cpu().float(),
        f"experiments/rama/results/pt_{config['name']}_{config['architecture']}_final.pt"
    )
    
    print(f"\nSimulation complete! Results saved in experiments/rama/")
    print(f"Plots: {base_plot_path}")
    print(f"Trajectory: experiments/rama/results/pt_{config['name']}_{config['architecture']}_final.pt")


if __name__ == '__main__':
    torch.manual_seed(42)
    
    # Configuration - edit these parameters
    config = {
        "name": "AA",                    # Peptide name (AA, AK, AS, etc.)
        "architecture": "transformer",   # Architecture: simple, graph, transformer
        "temp_schedule": "geom",         # Temperature schedule: geom or linear
        "temp_low": 1.0,
        "temp_high": 5.0,
        "total_n_temp": 5,
        "num_chains": 10,
        "num_steps": 1000000,              # PT simulation steps
        "step_size": 0.0001,             # MCMC step size
        "swap_interval": 100,            # Swap attempt frequency
        "check_interval": 100000,          # Plot generation frequency
        "use_gpu": True,                 # Use GPU for flows (CPU for target)
    }
    
    main(config)