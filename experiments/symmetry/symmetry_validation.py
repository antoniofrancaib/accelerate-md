#!/usr/bin/env python3
"""
Comprehensive symmetry validation for PT swap flows.

Tests both simple and transformer architectures on AA dipeptide data
to validate rotation and translation equivariance/invariance properties.

conda activate accelmd && python symmetry_validation.py --n-samples 50 --n-rotations 15 --output symmetry_validation
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
# Removed PdfPages import - now using individual PNG files
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Import AccelMD modules
from src.accelmd.utils.config import load_config, create_run_config, setup_device
from src.accelmd.data.pt_pair_dataset import PTTemperaturePairDataset
from src.accelmd.targets import build_target
from src.accelmd.evaluation.swap_acceptance import flow_acceptance
from torch.utils.data import DataLoader

# Import model building from main
import sys
sys.path.append('.')
from main import build_model

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def generate_random_rotation_matrix(device: str = "cpu") -> torch.Tensor:
    """Generate a random 3D rotation matrix using Rodrigues' formula."""
    # Random axis (unit vector)
    axis = torch.randn(3, device=device)
    axis = axis / torch.norm(axis)
    
    # Random angle between 0 and 2π
    angle = torch.rand(1, device=device) * 2 * np.pi
    
    # Rodrigues rotation formula: R = I + sin(θ)[K] + (1-cos(θ))[K]²
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Skew-symmetric matrix [K]
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=device, dtype=axis.dtype)
    
    # Rotation matrix
    I = torch.eye(3, device=device, dtype=axis.dtype)
    R = I + sin_angle * K + (1 - cos_angle) * torch.mm(K, K)
    
    return R, angle.item()


def apply_rigid_transform(coords: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rigid transformation: x' = R*x + t"""
    # coords: [B, N, 3]
    # R: [3, 3]
    # t: [3]
    return torch.matmul(coords, R.T) + t.unsqueeze(0).unsqueeze(0)


def compute_phi_psi_angles(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute phi and psi dihedral angles for alanine dipeptide.
    
    For AA dipeptide, the key atoms are:
    - phi: C(i-1) - N(i) - CA(i) - C(i)
    - psi: N(i) - CA(i) - C(i) - N(i+1)
    """
    # Simplified assumption for AA dipeptide backbone atoms
    # This is a rough approximation - real implementation would need proper atom mapping
    if coords.shape[1] < 10:  # Not enough atoms for proper dihedral calculation
        return torch.zeros(coords.shape[0]), torch.zeros(coords.shape[0])
    
    # Use approximate atom indices for backbone dihedrals
    # In practice, this should use proper topology information
    phi_indices = [0, 2, 4, 6]  # Approximate backbone atoms
    psi_indices = [2, 4, 6, 8]  # Approximate backbone atoms
    
    def dihedral_angle(coords, indices):
        """Compute dihedral angle for given atom indices."""
        p0, p1, p2, p3 = [coords[:, i] for i in indices]
        
        b1 = p1 - p0
        b2 = p2 - p1  
        b3 = p3 - p2
        
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        
        # Normalize
        n1 = n1 / (torch.norm(n1, dim=-1, keepdim=True) + 1e-8)
        n2 = n2 / (torch.norm(n2, dim=-1, keepdim=True) + 1e-8)
        
        # Compute angle
        cos_angle = torch.sum(n1 * n2, dim=-1)
        sin_angle = torch.sum(torch.cross(n1, n2, dim=-1) * b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8), dim=-1)
        
        angle = torch.atan2(sin_angle, cos_angle)
        return angle
    
    try:
        phi = dihedral_angle(coords, phi_indices)
        psi = dihedral_angle(coords, psi_indices)
    except:
        # Fallback if indices are invalid
        phi = torch.zeros(coords.shape[0])
        psi = torch.zeros(coords.shape[0])
    
    return phi, psi


class SymmetryValidator:
    """Main class for conducting symmetry validation experiments."""
    
    def __init__(self, config_paths: Dict[str, str], checkpoint_paths: Dict[str, Dict[str, str]], 
                 device: str = "cpu", n_test_samples: int = 100, n_rotations: int = 20):
        """
        Initialize the symmetry validator.
        
        Parameters
        ----------
        config_paths : Dict[str, str]
            Dictionary mapping architecture names to config file paths
        checkpoint_paths : Dict[str, Dict[str, str]]
            Nested dict: architecture -> pair -> checkpoint path
        device : str
            Device to run experiments on
        n_test_samples : int
            Number of test samples to use
        n_rotations : int
            Number of random rotations to test per sample
        """
        self.config_paths = config_paths
        self.checkpoint_paths = checkpoint_paths
        self.device = device
        self.n_test_samples = n_test_samples
        self.n_rotations = n_rotations
        
        # Load configs and create models
        self.configs = {}
        self.models = {}
        self.targets = {}
        
        for arch_name, config_path in config_paths.items():
            base_cfg = load_config(config_path)
            self.configs[arch_name] = base_cfg
            
            # Initialize models for each temperature pair
            self.models[arch_name] = {}
            self.targets[arch_name] = {}
            
            for pair_name, checkpoint_path in checkpoint_paths[arch_name].items():
                pair = self._parse_pair_name(pair_name)
                
                # Create run config
                run_cfg = create_run_config(base_cfg, pair, device)
                
                # Build model
                model = self._build_model(run_cfg, pair)
                
                # Load checkpoint
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                
                self.models[arch_name][pair_name] = model
                
                # Build target distributions
                target_kwargs = self._get_target_kwargs(base_cfg, pair)
                temps = run_cfg["temperatures"]["values"]
                
                target_low = build_target("dipeptide", temperature=temps[pair[0]], device="cpu", **target_kwargs)
                target_high = build_target("dipeptide", temperature=temps[pair[1]], device="cpu", **target_kwargs)
                
                self.targets[arch_name][pair_name] = (target_low, target_high)
        
        # Load test dataset (use pair 0_1 as representative)
        self.test_dataset = self._load_test_dataset()
        
        print(f"Initialized SymmetryValidator with {len(self.models)} architectures")
        print(f"Test dataset: {len(self.test_dataset)} samples")
    
    def _parse_pair_name(self, pair_name: str) -> Tuple[int, int]:
        """Parse pair name like 'pair_0_1' to tuple (0, 1)."""
        parts = pair_name.split('_')
        return (int(parts[1]), int(parts[2]))
    
    def _build_model(self, cfg: Dict, pair: Tuple[int, int]):
        """Build model using the same logic as main.py."""
        model_cfg = cfg["model"]
        temps = cfg["temperatures"]["values"]
        
        # Determine target configuration
        peptide_code = cfg.get("peptide_code", "AA").upper()
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
        
        # Add system-level energy parameters
        sys_cfg = cfg.get("system", {})
        energy_cut = sys_cfg.get("energy_cut")
        energy_max = sys_cfg.get("energy_max")
        
        target_kwargs_extra.update({
            "energy_cut": float(energy_cut) if energy_cut is not None else None,
            "energy_max": float(energy_max) if energy_max is not None else None,
        })
        
        # Get num_atoms from dataset
        num_atoms = self._get_num_atoms_from_dataset(cfg)
        
        return build_model(
            model_cfg=model_cfg,
            pair=pair,
            temps=temps,
            target_name=target_name,
            target_kwargs=target_kwargs_extra,
            device=self.device,
            num_atoms=num_atoms,
        )
    
    def _get_num_atoms_from_dataset(self, cfg: Dict) -> int:
        """Get number of atoms from the dataset."""
        peptide_code = cfg.get("peptide_code", "AA").upper()
        atom_types_path = f"datasets/pt_dipeptides/{peptide_code}/atom_types.pt"
        
        try:
            atom_types = torch.load(atom_types_path, map_location="cpu")
            return len(atom_types)
        except:
            # Fallback
            return 23  # AA dipeptide default
    
    def _get_target_kwargs(self, cfg: Dict, pair: Tuple[int, int]) -> Dict:
        """Get target kwargs for building Boltzmann distributions."""
        peptide_code = cfg.get("peptide_code", "AA").upper()
        pdb_path = f"datasets/pt_dipeptides/{peptide_code}/ref.pdb"
        
        sys_cfg = cfg.get("system", {})
        energy_cut = sys_cfg.get("energy_cut")
        energy_max = sys_cfg.get("energy_max")
        
        return {
            "pdb_path": pdb_path,
            "env": "implicit",
            "energy_cut": float(energy_cut) if energy_cut is not None else None,
            "energy_max": float(energy_max) if energy_max is not None else None,
        }
    
    def _load_test_dataset(self) -> PTTemperaturePairDataset:
        """Load test dataset for AA dipeptide."""
        # Use the simple config as base
        base_cfg = list(self.configs.values())[0]
        
        dataset = PTTemperaturePairDataset(
            pt_data_path="datasets/pt_dipeptides/AA/pt_AA.pt",
            molecular_data_path="datasets/pt_dipeptides/AA",
            temp_pair=(0, 1),  # Use first pair as representative
            subsample_rate=10000,  # Use even less frequent samples for testing
            device="cpu",
            filter_chirality=False,
            center_coordinates=False,  # Important: don't center for symmetry tests
        )
        
        # Subsample for testing
        indices = torch.randperm(len(dataset))[:self.n_test_samples]
        return torch.utils.data.Subset(dataset, indices)
    
    def run_symmetry_validation(self) -> Dict:
        """Run comprehensive symmetry validation experiments."""
        print("Running symmetry validation experiments...")
        
        results = {
            'equivariance_errors': {},
            'energy_drifts': {},
            'log_det_differences': {},
            'ramachandran_angles': {},
            'acceptance_changes': {},
            'rotation_angles': []
        }
        
        # Generate test rotations and translations
        rotations = []
        translations = []
        for _ in range(self.n_rotations):
            R, angle = generate_random_rotation_matrix(self.device)
            t = torch.randn(3, device=self.device) * 0.1  # Small random translation
            rotations.append((R, angle))
            translations.append(t)
            results['rotation_angles'].append(angle)
        
        # Test each architecture and pair
        for arch_name in self.models.keys():
            print(f"\nTesting architecture: {arch_name}")
            
            results['equivariance_errors'][arch_name] = {}
            results['energy_drifts'][arch_name] = {}
            results['log_det_differences'][arch_name] = {}
            results['ramachandran_angles'][arch_name] = {}
            results['acceptance_changes'][arch_name] = {}
            
            for pair_name in self.models[arch_name].keys():
                print(f"  Testing pair: {pair_name}")
                
                model = self.models[arch_name][pair_name]
                target_low, target_high = self.targets[arch_name][pair_name]
                
                # Get test batch
                test_batch = self._get_test_batch()
                source_coords = test_batch['source_coords'].to(self.device)
                target_coords = test_batch['target_coords'].to(self.device)
                atom_types = test_batch['atom_types'].to(self.device)
                adj_list = test_batch['adj_list'].to(self.device)
                edge_batch_idx = test_batch.get('edge_batch_idx', torch.tensor([], dtype=torch.long))
                
                # Store results for this pair
                pair_equivariance_errors = []
                pair_energy_drifts = []
                pair_log_det_diffs = []
                pair_acceptance_changes = []
                
                # Original forward pass
                original_output, original_log_det = self._forward_pass(
                    model, source_coords, atom_types, adj_list, edge_batch_idx
                )
                
                # Original Ramachandran angles
                orig_phi, orig_psi = compute_phi_psi_angles(original_output.cpu())
                
                # Test each rotation
                for (R, angle), t in zip(rotations, translations):
                    # Apply rigid transform to input
                    transformed_coords = apply_rigid_transform(source_coords, R, t)
                    
                    # Forward pass on transformed input
                    transformed_output, transformed_log_det = self._forward_pass(
                        model, transformed_coords, atom_types, adj_list, edge_batch_idx
                    )
                    
                    # Expected output (apply same transform to original output)
                    expected_output = apply_rigid_transform(original_output, R, t)
                    
                    # Compute equivariance error
                    equivariance_error = torch.norm(
                        transformed_output - expected_output, dim=(1, 2)
                    ).mean().item()
                    pair_equivariance_errors.append(equivariance_error)
                    
                    # Compute energy drift
                    original_energy = target_low.log_prob(original_output.cpu().view(-1, original_output.shape[1]*3))
                    transformed_energy = target_low.log_prob(transformed_output.cpu().view(-1, transformed_output.shape[1]*3))
                    expected_energy = target_low.log_prob(expected_output.cpu().view(-1, expected_output.shape[1]*3))
                    
                    energy_drift = torch.abs(transformed_energy - expected_energy).mean().item()
                    pair_energy_drifts.append(energy_drift)
                    
                    # Log-det difference
                    log_det_diff = torch.abs(transformed_log_det - original_log_det).mean().item()
                    pair_log_det_diffs.append(log_det_diff)
                    
                    # Acceptance change (simplified)
                    # This is a rough approximation - full calculation would need the acceptance formula
                    acc_change = torch.abs(original_energy - transformed_energy).mean().item()
                    pair_acceptance_changes.append(acc_change)
                
                # Store results
                results['equivariance_errors'][arch_name][pair_name] = pair_equivariance_errors
                results['energy_drifts'][arch_name][pair_name] = pair_energy_drifts
                results['log_det_differences'][arch_name][pair_name] = pair_log_det_diffs
                results['acceptance_changes'][arch_name][pair_name] = pair_acceptance_changes
                
                # Store Ramachandran angles
                trans_phi, trans_psi = compute_phi_psi_angles(transformed_output.cpu())
                results['ramachandran_angles'][arch_name][pair_name] = {
                    'original_phi': orig_phi.numpy(),
                    'original_psi': orig_psi.numpy(),
                    'transformed_phi': trans_phi.numpy(),
                    'transformed_psi': trans_psi.numpy(),
                }
        
        print("Symmetry validation completed!")
        return results
    
    def _get_test_batch(self) -> Dict[str, torch.Tensor]:
        """Get a test batch from the dataset."""
        loader = DataLoader(
            self.test_dataset, 
            batch_size=min(32, len(self.test_dataset)), 
            shuffle=False,
            collate_fn=PTTemperaturePairDataset.collate_fn
        )
        return next(iter(loader))
    
    def _forward_pass(self, model, coords, atom_types, adj_list, edge_batch_idx):
        """Perform forward pass through model with proper interface handling."""
        with torch.no_grad():
            if hasattr(model, 'forward'):
                # Check if this is a graph/transformer flow or simple flow
                if 'graph' in str(type(model)).lower() or 'transformer' in str(type(model)).lower():
                    # Graph or transformer flow
                    try:
                        output, log_det = model.forward(
                            coordinates=coords,
                            atom_types=atom_types,
                            adj_list=adj_list,
                            edge_batch_idx=edge_batch_idx,
                            reverse=False
                        )
                    except TypeError:
                        # Fallback: some models might not accept all arguments
                        output, log_det = model.forward(coords, atom_types, adj_list, reverse=False)
                else:
                    # Simple flow - just pass coordinates
                    output, log_det = model.forward(coords)
            else:
                # Fallback
                output, log_det = model(coords)
        
        return output, log_det
    
    def generate_plots(self, results: Dict, output_prefix: str = "symmetry_validation"):
        """Generate comprehensive plots and save as separate PNG files."""
        print(f"Generating plots with prefix: {output_prefix}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_prefix).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate individual plots
        plot_files = []
        
        # Plot 1: Equivariance Error vs Rotation Angle
        fig1_path = f"{output_prefix}_equivariance_scatter.png"
        self._plot_equivariance_scatter(results, fig1_path)
        plot_files.append(fig1_path)
        
        # Plot 2: Energy Drift Analysis
        fig2_path = f"{output_prefix}_energy_drift.png"
        self._plot_energy_drift(results, fig2_path)
        plot_files.append(fig2_path)
        
        # Plot 3: Ramachandran Angle Overlays
        fig3_path = f"{output_prefix}_ramachandran_overlays.png"
        self._plot_ramachandran_overlays(results, fig3_path)
        plot_files.append(fig3_path)
        
        # Plot 4: Log-Determinant Distributions
        fig4_path = f"{output_prefix}_log_det_distributions.png"
        self._plot_log_det_distributions(results, fig4_path)
        plot_files.append(fig4_path)
        
        # Plot 5: Summary Table
        fig5_path = f"{output_prefix}_summary_table.png"
        self._plot_summary_table(results, fig5_path)
        plot_files.append(fig5_path)
        
        print(f"Generated {len(plot_files)} plot files:")
        for file_path in plot_files:
            print(f"  {file_path}")
    
    def _plot_equivariance_scatter(self, results: Dict, output_path: str):
        """Plot equivariance error vs rotation angle."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left panel: Equivariance error vs rotation angle
        for arch_name in results['equivariance_errors'].keys():
            for pair_name in results['equivariance_errors'][arch_name].keys():
                errors = results['equivariance_errors'][arch_name][pair_name]
                angles = results['rotation_angles']
                
                ax1.scatter(angles, errors, alpha=0.6, 
                           label=f"{arch_name}-{pair_name}", s=30)
        
        ax1.set_xlabel('Rotation Angle (radians)')
        ax1.set_ylabel('Equivariance Error (Å)')
        ax1.set_title('Flow Equivariance: ||f(Rx+t) - (Rf(x)+t)||')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Right panel: Energy drift vs rotation angle
        for arch_name in results['energy_drifts'].keys():
            for pair_name in results['energy_drifts'][arch_name].keys():
                drifts = results['energy_drifts'][arch_name][pair_name]
                angles = results['rotation_angles']
                
                ax2.scatter(angles, drifts, alpha=0.6, 
                           label=f"{arch_name}-{pair_name}", s=30)
        
        ax2.set_xlabel('Rotation Angle (radians)')
        ax2.set_ylabel('Energy Drift |ΔU|')
        ax2.set_title('Energy Invariance: |U(f(Rx+t)) - U(Rf(x)+t)|')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_energy_drift(self, results: Dict, output_path: str):
        """Plot energy drift analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create violin plot of energy drifts
        data = []
        labels = []
        
        for arch_name in results['energy_drifts'].keys():
            for pair_name in results['energy_drifts'][arch_name].keys():
                drifts = results['energy_drifts'][arch_name][pair_name]
                data.extend(drifts)
                labels.extend([f"{arch_name}\n{pair_name}"] * len(drifts))
        
        df = pd.DataFrame({'Energy Drift': data, 'Model': labels})
        
        sns.violinplot(data=df, x='Model', y='Energy Drift', ax=ax)
        ax.set_title('Distribution of Energy Drifts Across Models and Pairs')
        ax.set_ylabel('|ΔU| (Energy Drift)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ramachandran_overlays(self, results: Dict, output_path: str):
        """Plot Ramachandran angle overlays."""
        n_pairs = len(list(results['ramachandran_angles'].values())[0])
        n_archs = len(results['ramachandran_angles'])
        
        fig, axes = plt.subplots(n_archs, n_pairs, figsize=(4*n_pairs, 4*n_archs))
        if n_archs == 1:
            axes = axes.reshape(1, -1)
        if n_pairs == 1:
            axes = axes.reshape(-1, 1)
        
        for i, arch_name in enumerate(results['ramachandran_angles'].keys()):
            for j, pair_name in enumerate(results['ramachandran_angles'][arch_name].keys()):
                ax = axes[i, j]
                
                angles = results['ramachandran_angles'][arch_name][pair_name]
                
                # Plot original and transformed angle distributions
                ax.scatter(angles['original_phi'], angles['original_psi'], 
                          alpha=0.6, s=20, label='Original', color='blue')
                ax.scatter(angles['transformed_phi'], angles['transformed_psi'], 
                          alpha=0.6, s=20, label='Rotated Input', color='red')
                
                ax.set_xlabel('φ (radians)')
                ax.set_ylabel('ψ (radians)')
                ax.set_title(f'{arch_name} - {pair_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Ramachandran Angle Invariance Test', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_log_det_distributions(self, results: Dict, output_path: str):
        """Plot log-determinant difference distributions."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create ridge plot data
        data = []
        labels = []
        
        for arch_name in results['log_det_differences'].keys():
            for pair_name in results['log_det_differences'][arch_name].keys():
                diffs = results['log_det_differences'][arch_name][pair_name]
                data.extend(diffs)
                labels.extend([f"{arch_name}-{pair_name}"] * len(diffs))
        
        df = pd.DataFrame({'Log-Det Difference': data, 'Model': labels})
        
        sns.violinplot(data=df, x='Model', y='Log-Det Difference', ax=ax)
        ax.set_title('Distribution of Log-Determinant Differences')
        ax.set_ylabel('|Δ log|det J||')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary_table(self, results: Dict, output_path: str):
        """Generate summary statistics table."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Compute summary statistics
        summary_data = []
        
        for arch_name in results['equivariance_errors'].keys():
            for pair_name in results['equivariance_errors'][arch_name].keys():
                eq_errors = results['equivariance_errors'][arch_name][pair_name]
                energy_drifts = results['energy_drifts'][arch_name][pair_name]
                log_det_diffs = results['log_det_differences'][arch_name][pair_name]
                acc_changes = results['acceptance_changes'][arch_name][pair_name]
                
                summary_data.append({
                    'Architecture': arch_name,
                    'Temperature Pair': pair_name,
                    'Mean Equivariance Error (Å)': f"{np.mean(eq_errors):.2e}",
                    'Std Equivariance Error (Å)': f"{np.std(eq_errors):.2e}",
                    'Mean Energy Drift': f"{np.mean(energy_drifts):.2e}",
                    'Std Energy Drift': f"{np.std(energy_drifts):.2e}",
                    'Mean Log-Det Diff': f"{np.mean(log_det_diffs):.2e}",
                    'Std Log-Det Diff': f"{np.std(log_det_diffs):.2e}",
                    'Mean Acceptance Change': f"{np.mean(acc_changes):.2e}",
                })
        
        # Create table
        df = pd.DataFrame(summary_data)
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Symmetry Validation Summary Statistics', pad=20, fontsize=16, weight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run symmetry validation."""
    parser = argparse.ArgumentParser(description="Symmetry validation for PT swap flows")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--n-rotations", type=int, default=15, help="Number of random rotations per sample")
    parser.add_argument("--output", default="symmetry_validation", help="Output prefix for PNG files")
    parser.add_argument("--architectures", nargs="+", default=["simple", "transformer", "graph"], 
                       choices=["simple", "transformer", "graph"],
                       help="Which architectures to test (default: all)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Define config and checkpoint paths
    all_config_paths = {
        "simple": "configs/AA_simple.yaml",
        "transformer": "configs/multi_transformer.yaml",
        "graph": "configs/multi_graph.yaml"
    }
    
    all_checkpoint_paths = {
        "simple": {
            "pair_0_1": "checkpoints/AA_simple/pair_0_1/models/best_model_epoch2787.pt",
            "pair_1_2": "checkpoints/AA_simple/pair_1_2/models/best_model_epoch1231.pt",
            "pair_2_3": "checkpoints/AA_simple/pair_2_3/models/best_model_epoch1301.pt",
            "pair_3_4": "checkpoints/AA_simple/pair_3_4/models/best_model_epoch773.pt",
        },
        "transformer": {
            "pair_0_1": "checkpoints/multi_transformer/pair_0_1/models/best_model_epoch331.pt",
            "pair_1_2": "checkpoints/multi_transformer/pair_1_2/models/best_model_epoch325.pt",
            "pair_2_3": "checkpoints/multi_transformer/pair_2_3/models/best_model_epoch257.pt",
            "pair_3_4": "checkpoints/multi_transformer/pair_3_4/models/best_model_epoch254.pt",
        },
        "graph": {
            "pair_0_1": "outputs/multi_graph/pair_0_1/models/best_model_epoch1475.pt",
        }
    }
    
    # Filter to only selected architectures
    config_paths = {arch: all_config_paths[arch] for arch in args.architectures if arch in all_config_paths}
    checkpoint_paths = {arch: all_checkpoint_paths[arch] for arch in args.architectures if arch in all_checkpoint_paths}
    
    # Verify files exist
    missing_files = []
    for arch_name, config_path in config_paths.items():
        if not Path(config_path).exists():
            missing_files.append(config_path)
    
    for arch_name, arch_checkpoints in checkpoint_paths.items():
        for pair_name, checkpoint_path in arch_checkpoints.items():
            if not Path(checkpoint_path).exists():
                missing_files.append(checkpoint_path)
    
    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  {f}")
        return
    
    # Initialize validator
    validator = SymmetryValidator(
        config_paths=config_paths,
        checkpoint_paths=checkpoint_paths,
        device=device,
        n_test_samples=args.n_samples,
        n_rotations=args.n_rotations
    )
    
    # Run validation
    results = validator.run_symmetry_validation()
    
    # Generate plots
    validator.generate_plots(results, args.output)
    
    # Save raw results
    results_path = f"{args.output}_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = convert_for_json(results)
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Raw results saved to {results_path}")
    print("Symmetry validation completed!")


if __name__ == "__main__":
    main()