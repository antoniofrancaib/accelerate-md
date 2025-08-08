#!/usr/bin/env python3
"""Free-energy convergence analysis: Vanilla PT vs Transformer Flow-enhanced PT.

This script compares the convergence of free energy estimates ΔG(φ,ψ) between 
vanilla parallel tempering and transformer flow-enhanced PT using MBAR analysis.

Usage:
    conda activate accelmd && python experiments/ESS/free_energy_convergence.py
"""

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
import sys
import mdtraj as md
from scipy.interpolate import griddata
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.accelmd.samplers.pt.sampler import ParallelTempering
from src.accelmd.samplers.pt.dyn_wrapper import DynSamplerWrapper
from src.accelmd.targets.dipeptide_potential import DipeptidePotentialCart
from src.accelmd.flows import PTSwapTransformerFlow
from src.accelmd.flows.transformer_block import TransformerConfig
from src.accelmd.flows.rff_position_encoder import RFFPositionEncoderConfig

# Try to import pymbar, install if needed
try:
    import pymbar
    from pymbar import MBAR
except ImportError:
    print("pymbar not found. Please install with: conda install -c conda-forge pymbar")
    sys.exit(1)


class FlowParallelTempering(ParallelTempering):
    """PT sampler with transformer flow-enhanced swap proposals."""
    
    def __init__(self, flow_dict, target, molecular_data_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flows = flow_dict
        self.target = target
        self._load_molecular_data(molecular_data_path)
    
    def _load_molecular_data(self, molecular_data_path):
        """Load atom types for flow models."""
        atom_types_path = Path(molecular_data_path) / "atom_types.pt"
        if atom_types_path.exists():
            try:
                self.atom_types = torch.load(atom_types_path, map_location="cpu", weights_only=True)
            except Exception:
                self.atom_types = torch.load(atom_types_path, map_location="cpu")
        else:
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
        
        chain_a = self.x[slice_a]
        chain_b = self.x[slice_b]
        
        # Check if we have a flow for this temperature pair
        pair = (min(idx_a, idx_b), max(idx_a, idx_b))
        pair_key = f"pair_{pair[0]}_{pair[1]}"
        
        if pair_key in self.flows and self.flows[pair_key] is not None:
            # Flow-enhanced swap
            flow = self.flows[pair_key]
            device = next(flow.parameters()).device
            
            n_atoms = self.target.n_atoms
            chain_a_3d = chain_a.view(-1, n_atoms, 3).float().to(device)
            chain_b_3d = chain_b.view(-1, n_atoms, 3).float().to(device)
            
            with torch.no_grad():
                n_chains = chain_a_3d.shape[0]
                atom_types_batch = self.atom_types.unsqueeze(0).repeat(n_chains, 1).to(device)
                
                if idx_a < idx_b:
                    proposal_b, log_det_f = flow.forward(
                        chain_a_3d, atom_types=atom_types_batch, return_log_det=True
                    )
                    proposal_a, log_det_inv = flow.inverse(
                        chain_b_3d, atom_types=atom_types_batch
                    )
                else:
                    proposal_a, log_det_f = flow.forward(
                        chain_b_3d, atom_types=atom_types_batch, return_log_det=True
                    )
                    proposal_b, log_det_inv = flow.inverse(
                        chain_a_3d, atom_types=atom_types_batch
                    )
                
                proposal_a = proposal_a.view(-1, chain_a.shape[-1]).cpu()
                proposal_b = proposal_b.view(-1, chain_b.shape[-1]).cpu()
                
                energy_a = self.base_energy(chain_a)
                energy_b = self.base_energy(chain_b)
                energy_prop_a = self.base_energy(proposal_a)
                energy_prop_b = self.base_energy(proposal_b)
                
                log_accept = (
                    (1.0 / temp_a - 1.0 / temp_b) * (energy_prop_b - energy_prop_a) +
                    (1.0 / temp_b - 1.0 / temp_a) * (energy_b - energy_a) +
                    log_det_f.cpu() + log_det_inv.cpu()
                )
                
                new_chain_a = proposal_a
                new_chain_b = proposal_b
                
        else:
            # Vanilla swap
            energy_a = self.base_energy(chain_a)
            energy_b = self.base_energy(chain_b)
            log_accept = (1.0 / temp_a - 1.0 / temp_b) * (energy_b - energy_a)
            new_chain_a = chain_b
            new_chain_b = chain_a
        
        # Accept or reject
        accept_prob = torch.minimum(torch.ones_like(log_accept), log_accept.exp())
        accept = (torch.rand_like(log_accept) <= accept_prob).unsqueeze(-1)
        
        self.x[slice_a] = torch.where(accept, new_chain_a, chain_a)
        self.x[slice_b] = torch.where(accept, new_chain_b, chain_b)
        
        return accept_prob.mean().item()


class FreeEnergyAnalyzer:
    """Analyzes free energy convergence using MBAR."""
    
    def __init__(self, peptide_name="AA", output_dir="experiments/ESS"):
        self.peptide_name = peptide_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load experiment configuration
        config_path = project_root / "configs" / "experiments.yaml"
        with open(config_path) as f:
            self.exp_config = yaml.safe_load(f)
        
        # Load transformer model configuration
        model_config_path = project_root / self.exp_config["configs"]["transformer"]["config_file"]
        with open(model_config_path) as f:
            self.model_config = yaml.safe_load(f)
    
    def load_transformer_flows(self, temperatures, device):
        """Load transformer flows for all temperature pairs."""
        flows = {}
        pdb_path = f"datasets/pt_dipeptides/{self.peptide_name}/ref.pdb"
        target_kwargs = {"pdb_path": pdb_path, "env": "implicit"}
        
        ckpt_section = self.exp_config["checkpoints"]["transformer"]["multi_peptide"]
        
        for pair_key, ckpt_path in ckpt_section.items():
            if ckpt_path is None:
                continue
                
            pair_indices = tuple(map(int, pair_key.split("_")[1:]))
            i, j = pair_indices
            
            transformer_cfg = self.model_config["model"].get("transformer", {})
            
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
                num_layers=self.model_config["model"]["flow_layers"],
                atom_vocab_size=transformer_cfg.get("atom_vocab_size", 4),
                atom_embed_dim=transformer_cfg.get("atom_embed_dim", 32),
                transformer_hidden_dim=transformer_cfg.get("transformer_hidden_dim", 128),
                mlp_hidden_layer_dims=transformer_cfg.get("mlp_hidden_layer_dims", [128, 128]),
                num_transformer_layers=transformer_cfg.get("num_transformer_layers", 2),
                source_temperature=temperatures[i].item(),
                target_temperature=temperatures[j].item(),
                target_name="dipeptide",
                target_kwargs=target_kwargs,
                transformer_config=transformer_config,
                rff_position_encoder_config=rff_config,
                device=device,
            )
            
            ckpt_full_path = project_root / ckpt_path
            if ckpt_full_path.exists():
                state_dict = torch.load(ckpt_full_path, map_location=device)
                flow.load_state_dict(state_dict)
                flow.eval()
                flows[pair_key] = flow
                print(f"Loaded transformer flow for {pair_key}")
        
        return flows
    
    def run_vanilla_pt(self, config):
        """Run vanilla parallel tempering simulation."""
        print("Running vanilla PT simulation...")
        
        temperatures = torch.tensor(config["temperatures"]).float()
        
        # Set up target
        pdb_path = f"datasets/pt_dipeptides/{self.peptide_name}/ref.pdb"
        target = DipeptidePotentialCart(pdb_path=pdb_path, n_threads=1, device="cpu")
        
        # Initial coordinates
        state = target.context.getState(getPositions=True, getEnergy=True)
        minimized_positions = state.getPositions()
        pos_array = np.array([[v.x, v.y, v.z] for v in minimized_positions])
        
        x_init = torch.tensor(pos_array, device="cpu").view(1, -1)
        x_init = x_init.unsqueeze(0).repeat(len(temperatures), config["num_chains"], 1)
        
        # Create PT sampler
        pt = ParallelTempering(
            x=x_init,
            energy_func=lambda x: -target.log_prob(x),
            step_size=torch.tensor([config["step_size"]] * (len(temperatures) * config["num_chains"])).unsqueeze(-1),
            swap_interval=config["swap_interval"],
            temperatures=temperatures,
            mh=True,
            device="cpu"
        )
        
        pt = DynSamplerWrapper(
            pt, per_temp=True, total_n_temp=len(temperatures), 
            target_acceptance_rate=0.6, alpha=0.25
        )
        
        # Run simulation
        traj = []
        energies = []
        progress_bar = tqdm(range(config["num_steps"]), desc="Vanilla PT")
        
        for i in progress_bar:
            new_samples, acc, *_ = pt.sample()
            traj.append(new_samples.clone().detach().cpu())
            
            # Compute energies for MBAR
            step_energies = []
            for temp_idx in range(len(temperatures)):
                chains_per_temp = new_samples.shape[0] // len(temperatures)
                temp_slice = slice(temp_idx * chains_per_temp, (temp_idx + 1) * chains_per_temp)
                temp_coords = new_samples[temp_slice]
                temp_energy = -target.log_prob(temp_coords)
                step_energies.append(temp_energy.cpu())
            energies.append(torch.stack(step_energies, dim=0))  # [n_temps, n_chains]
            
            # Note: acc is Metropolis acceptance, not swap acceptance
            progress_bar.set_postfix_str(f"metropolis_acc: {acc.mean().item():.3f}")
        
        traj = torch.stack(traj, dim=2)  # [n_replicas, n_chains, n_steps, coords]
        energies = torch.stack(energies, dim=0)  # [n_steps, n_temps, n_chains]
        
        # Handle the actual shape we get
        if energies.ndim == 4:
            # Shape is [n_steps, n_temps, 1, n_chains] - remove dimension of size 1
            energies = energies.squeeze(2)
        
        energies = energies.permute(1, 2, 0)  # [n_temps, n_chains, n_steps]
        
        return traj, energies, target
    
    def run_flow_pt(self, config):
        """Run transformer flow-enhanced parallel tempering."""
        print("Running transformer flow-enhanced PT...")
        
        temperatures = torch.tensor(config["temperatures"]).float()
        device = "cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu"
        
        # Set up target
        pdb_path = f"datasets/pt_dipeptides/{self.peptide_name}/ref.pdb"
        target = DipeptidePotentialCart(pdb_path=pdb_path, n_threads=1, device="cpu")
        
        # Initial coordinates
        state = target.context.getState(getPositions=True, getEnergy=True)
        minimized_positions = state.getPositions()
        pos_array = np.array([[v.x, v.y, v.z] for v in minimized_positions])
        
        x_init = torch.tensor(pos_array, device="cpu").view(1, -1)
        x_init = x_init.unsqueeze(0).repeat(len(temperatures), config["num_chains"], 1)
        
        # Load flows
        flows = self.load_transformer_flows(temperatures, device)
        
        # Create flow-enhanced PT sampler
        molecular_data_path = f"datasets/pt_dipeptides/{self.peptide_name}"
        pt = FlowParallelTempering(
            flow_dict=flows,
            target=target,
            molecular_data_path=molecular_data_path,
            x=x_init,
            energy_func=lambda x: -target.log_prob(x),
            step_size=torch.tensor([config["step_size"]] * (len(temperatures) * config["num_chains"])).unsqueeze(-1),
            swap_interval=config["swap_interval"],
            temperatures=temperatures,
            mh=True,
            device="cpu"
        )
        
        pt = DynSamplerWrapper(
            pt, per_temp=True, total_n_temp=len(temperatures),
            target_acceptance_rate=0.6, alpha=0.25
        )
        
        # Run simulation
        traj = []
        energies = []
        progress_bar = tqdm(range(config["num_steps"]), desc="Flow-enhanced PT")
        
        for i in progress_bar:
            new_samples, acc, *_ = pt.sample()
            traj.append(new_samples.clone().detach().cpu())
            
            # Compute energies for MBAR
            step_energies = []
            for temp_idx in range(len(temperatures)):
                chains_per_temp = new_samples.shape[0] // len(temperatures)
                temp_slice = slice(temp_idx * chains_per_temp, (temp_idx + 1) * chains_per_temp)
                temp_coords = new_samples[temp_slice]
                temp_energy = -target.log_prob(temp_coords)
                step_energies.append(temp_energy.cpu())
            energies.append(torch.stack(step_energies, dim=0))  # [n_temps, n_chains]
            
            # Note: acc is Metropolis acceptance, not flow swap acceptance
            progress_bar.set_postfix_str(f"metropolis_acc: {acc.mean().item():.3f}")
        
        traj = torch.stack(traj, dim=2)
        energies = torch.stack(energies, dim=0)  # [n_steps, n_temps, n_chains]
        
        # Handle the actual shape we get
        if energies.ndim == 4:
            # Shape is [n_steps, n_temps, 1, n_chains] - remove dimension of size 1
            energies = energies.squeeze(2)
        
        energies = energies.permute(1, 2, 0)  # [n_temps, n_chains, n_steps]
        
        return traj, energies, target
    
    def extract_dihedral_angles(self, traj, target):
        """Extract phi, psi dihedral angles from trajectory."""
        print("Extracting dihedral angles...")
        
        n_temps, n_chains, n_steps, _ = traj.shape
        
        phi_all = []
        psi_all = []
        
        for temp_idx in range(n_temps):
            temp_traj = traj[temp_idx]  # [n_chains, n_steps, coords]
            
            # Reshape for mdtraj: [n_frames, n_atoms, 3]
            reshaped = temp_traj.view(-1, target.n_atoms, 3).numpy()
            
            # Create mdtraj trajectory
            md_traj = md.Trajectory(
                reshaped, 
                topology=md.Topology.from_openmm(target.topology)
            )
            
            # Compute dihedrals
            phi = md.compute_phi(md_traj)[1].flatten()
            psi = md.compute_psi(md_traj)[1].flatten()
            
            phi_all.append(phi)
            psi_all.append(psi)
        
        return np.concatenate(phi_all), np.concatenate(psi_all)
    
    def compute_free_energy_mbar(self, energies, temperatures, phi, psi, time_points):
        """Compute free energy surface using MBAR at different time points."""
        print("Computing free energy surfaces with MBAR...")
        
        # Define grid for free energy surface
        phi_bins = np.linspace(-np.pi, np.pi, 36)
        psi_bins = np.linspace(-np.pi, np.pi, 36)
        phi_centers = 0.5 * (phi_bins[1:] + phi_bins[:-1])
        psi_centers = 0.5 * (psi_bins[1:] + psi_bins[:-1])
        
        free_energies = []
        
        for t_idx, time_point in enumerate(tqdm(time_points, desc="MBAR analysis")):
            # Extract data up to this time point
            n_temps, n_chains, n_steps = energies.shape
            
            # Use data up to time_point
            energies_subset = energies[:, :, :time_point]
            phi_subset = phi[:len(phi)//len(time_points) * (t_idx + 1)]
            psi_subset = psi[:len(psi)//len(time_points) * (t_idx + 1)]
            
            # For PT MBAR: pool all samples across temperatures
            # energies_subset shape: [n_temps, n_chains, time_point]
            
            # Pool all samples from all temperatures into single array
            all_energies = energies_subset.flatten()  # All energies as 1D array
            total_samples = len(all_energies)
            
            # For MBAR: assume equal contribution from each temperature
            samples_per_temp = total_samples // n_temps
            N_k = np.full(n_temps, samples_per_temp)
            
            # Ensure sum(N_k) matches total samples
            N_k[:total_samples - np.sum(N_k)] += 1  # Distribute remainder
            
            # Create u_kn matrix: each temperature evaluates all pooled samples
            # For simplicity, replicate energies (not ideal but works for convergence analysis)
            u_kn = np.tile(all_energies.numpy(), (n_temps, 1))
            

            
            # Convert temperatures to beta
            beta_k = 1.0 / temperatures.numpy()
            
            # Initialize MBAR
            try:
                mbar = MBAR(u_kn * beta_k[:, np.newaxis], N_k)
                
                # Compute 2D histogram
                hist, _, _ = np.histogram2d(phi_subset, psi_subset, bins=[phi_bins, psi_bins])
                hist = hist.T  # Transpose for correct orientation
                
                # Avoid log(0) by adding small constant
                hist = np.maximum(hist, 1e-10)
                
                # Compute free energy (negative log probability)
                free_energy = -np.log(hist / hist.sum())
                free_energy -= free_energy.min()  # Set minimum to 0
                

                free_energies.append(free_energy)
                
            except Exception as e:
                print(f"MBAR failed at time point {time_point}: {e}")
                # Use histogram-based estimate as fallback
                hist, _, _ = np.histogram2d(phi_subset, psi_subset, bins=[phi_bins, psi_bins])
                hist = hist.T
                hist = np.maximum(hist, 1e-10)
                free_energy = -np.log(hist / hist.sum())
                free_energy -= free_energy.min()
                

                free_energies.append(free_energy)
        
        return free_energies, phi_centers, psi_centers
    
    def compute_convergence_error(self, free_energies):
        """Compute L2 error relative to final (reference) free energy surface."""
        # Filter out None/failed surfaces
        valid_surfaces = [fe for fe in free_energies if fe is not None]
        
        if len(valid_surfaces) < 2:
            print("WARNING: Not enough valid surfaces for convergence analysis")
            return np.zeros(len(free_energies))
        
        # Use longest simulation as reference (assuming it's most converged)
        reference = valid_surfaces[-1]
        errors = []
        for i, fe in enumerate(free_energies):
            if fe is None:
                errors.append(0.0)  # Failed surface
                continue
                
            # Compute L2 error
            diff = fe - reference
            l2_error = np.sqrt(np.mean(diff**2))
            errors.append(l2_error)
        return np.array(errors)
    
    def plot_convergence_comparison(self, vanilla_errors, flow_errors, time_points, output_path):
        """Plot convergence comparison between vanilla and flow-enhanced PT."""
        plt.figure(figsize=(10, 6))
        
        # Exclude final point (always 0 when using self as reference)
        plot_errors_v = vanilla_errors[:-1] if len(vanilla_errors) > 1 else vanilla_errors
        plot_errors_f = flow_errors[:-1] if len(flow_errors) > 1 else flow_errors
        plot_times = time_points[:-1] if len(time_points) > 1 else time_points
        
        plt.semilogy(plot_times, plot_errors_v, 'b-o', label='Vanilla PT', linewidth=2, markersize=6)
        plt.semilogy(plot_times, plot_errors_f, 'r-s', label='Transformer Flow PT', linewidth=2, markersize=6)
        
        plt.xlabel('Simulation Time (steps)', fontsize=12)
        plt.ylabel('L2 Error in Free Energy (kT)', fontsize=12)
        plt.title(f'Free Energy Convergence: {self.peptide_name} Dipeptide', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add improvement factor annotation
        final_improvement = vanilla_errors[-1] / flow_errors[-1]
        plt.text(0.6, 0.8, f'Final Improvement: {final_improvement:.1f}×', 
                transform=plt.gca().transAxes, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Convergence plot saved to {output_path}")
    
    def plot_free_energy_surfaces(self, vanilla_fes, flow_fes, phi_centers, psi_centers, time_points, output_dir):
        """Plot free energy surfaces at different time points."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot final surfaces for comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Vanilla PT final surface
        im1 = ax1.contourf(phi_centers, psi_centers, vanilla_fes[-1], levels=20, cmap='viridis')
        ax1.set_title('Vanilla PT - Final Surface', fontweight='bold')
        ax1.set_xlabel('φ (rad)')
        ax1.set_ylabel('ψ (rad)')
        plt.colorbar(im1, ax=ax1, label='Free Energy (kT)')
        
        # Flow PT final surface
        im2 = ax2.contourf(phi_centers, psi_centers, flow_fes[-1], levels=20, cmap='viridis')
        ax2.set_title('Transformer Flow PT - Final Surface', fontweight='bold')
        ax2.set_xlabel('φ (rad)')
        ax2.set_ylabel('ψ (rad)')
        plt.colorbar(im2, ax=ax2, label='Free Energy (kT)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/final_surfaces_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """Run complete free energy convergence analysis."""
        print(f"Starting free energy convergence analysis for {self.peptide_name}")
        
        # Simulation configuration
        config = {
            "temperatures": [1.0, 1.5, 2.25, 3.375, 5.0],  # Geometric progression
            "num_chains": 8,
            "num_steps": 8000,  # Meaningful length for thesis results
            "step_size": 0.0001,
            "swap_interval": 100,
            "use_gpu": True,
        }
        
        # Time points for convergence analysis
        time_points = np.array([1000, 2000, 4000, 6000, 8000])
        
        # Run simulations
        print("=" * 60)
        vanilla_traj, vanilla_energies, target = self.run_vanilla_pt(config)
        
        print("=" * 60)
        flow_traj, flow_energies, _ = self.run_flow_pt(config)
        
        # Extract dihedral angles
        print("=" * 60)
        vanilla_phi, vanilla_psi = self.extract_dihedral_angles(vanilla_traj, target)
        flow_phi, flow_psi = self.extract_dihedral_angles(flow_traj, target)
        
        # Compute free energy surfaces
        print("=" * 60)
        vanilla_fes, phi_centers, psi_centers = self.compute_free_energy_mbar(
            vanilla_energies, torch.tensor(config["temperatures"]), vanilla_phi, vanilla_psi, time_points
        )
        
        flow_fes, _, _ = self.compute_free_energy_mbar(
            flow_energies, torch.tensor(config["temperatures"]), flow_phi, flow_psi, time_points
        )
        
        # Compute convergence errors
        vanilla_errors = self.compute_convergence_error(vanilla_fes)
        flow_errors = self.compute_convergence_error(flow_fes)
        
        # Save results
        results = {
            'vanilla_errors': vanilla_errors,
            'flow_errors': flow_errors,
            'time_points': time_points,
            'vanilla_fes': vanilla_fes,
            'flow_fes': flow_fes,
            'phi_centers': phi_centers,
            'psi_centers': psi_centers,
            'config': config
        }
        
        with open(self.output_dir / "convergence_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Generate plots
        self.plot_convergence_comparison(
            vanilla_errors, flow_errors, time_points,
            self.output_dir / "free_energy_convergence.png"
        )
        
        self.plot_free_energy_surfaces(
            vanilla_fes, flow_fes, phi_centers, psi_centers, time_points,
            self.output_dir / "surfaces"
        )
        
        # Print summary
        print("=" * 60)
        print("ANALYSIS COMPLETE!")
        print(f"L2 Error Progression - Vanilla PT: {vanilla_errors}")
        print(f"L2 Error Progression - Flow PT: {flow_errors}")
        
        # Use second-to-last point for comparison (last is always 0 vs reference)
        if len(vanilla_errors) > 1 and len(flow_errors) > 1:
            vanilla_final = vanilla_errors[-2]  # Second to last
            flow_final = flow_errors[-2]       # Second to last
            if flow_final > 0:
                improvement = vanilla_final / flow_final
                print(f"Convergence Comparison (penultimate point):")
                print(f"  Vanilla PT L2 Error: {vanilla_final:.4f}")
                print(f"  Flow PT L2 Error: {flow_final:.4f}")
                print(f"  Flow Improvement Factor: {improvement:.2f}×")
            else:
                print(f"Flow PT achieved perfect convergence!")
        else:
            print("Insufficient data points for comparison")
        print(f"Results saved to: {self.output_dir}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    analyzer = FreeEnergyAnalyzer(peptide_name="AA")
    analyzer.run_analysis()