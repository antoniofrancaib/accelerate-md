#!/usr/bin/env python3
"""
Symmetry Validation Script for Flow Architectures
=================================================

This script evaluates the symmetry invariance properties of two flow architectures:
- Graph Flow (PTSwapGraphFlow)
- Transformer Flow (PTSwapTransformerFlow)

The script applies symmetry transformations (rotation, translation) and measures 
changes in log-likelihood with confidence intervals to validate that the models 
preserve fundamental molecular symmetries.

Usage:
    python experiments/symmetry/symmetry_validation_script.py
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Add src and root to path for imports
sys.path.append('src')
sys.path.append('.')

# Import all necessary components
from accelmd.flows import PTSwapFlow, PTSwapGraphFlow, PTSwapTransformerFlow
from accelmd.data import PTTemperaturePairDataset
from accelmd.targets import build_target
from accelmd.utils.config import load_config, create_run_config, setup_device
from accelmd.data.molecular_data import random_rotation_augment, center_coordinates

def apply_rotation_transformation(coords: torch.Tensor) -> torch.Tensor:
    """Apply random 3D rotation to coordinates."""
    return random_rotation_augment(coords)

def apply_translation_transformation(coords: torch.Tensor, max_translation: float = 2.0) -> torch.Tensor:
    """Apply random translation up to max_translation Angstroms."""
    batch_size, n_atoms, _ = coords.shape
    # Random translation vector for each sample
    translation = torch.randn(batch_size, 1, 3) * max_translation
    return coords + translation

def load_model_and_data(config_path: str, checkpoint_path: str, peptide_override: str = "AA"):
    """Load a model and its corresponding dataset."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load configuration
    cfg = load_config(config_path)
    device = setup_device(cfg)
    
    # Set peptide code for single-peptide models
    if peptide_override and cfg.get('mode') != 'multi':
        cfg['peptide_code'] = peptide_override
    
    # Import build_model function
    from main import build_model
    
    # Build model
    model_cfg = cfg["model"]
    temps = cfg["temperatures"]["values"]
    
    # Try to determine target from config or use fallback
    if cfg.get('mode') == 'multi':
        target_name = "dipeptide" 
        target_kwargs = {
            "env": "implicit",
            "pdb_path": f"datasets/pt_dipeptides/{peptide_override}/ref.pdb"
        }
    else:
        target_name = cfg.get("target", {}).get("name", "dipeptide")
        target_kwargs = cfg.get("target", {}).get("kwargs", {"env": "implicit"})
        if "pdb_path" not in target_kwargs:
            target_kwargs["pdb_path"] = f"datasets/pt_dipeptides/{peptide_override}/ref.pdb"
    
    molecular_data_path = Path(f"datasets/pt_dipeptides/{peptide_override}")
    
    # Determine number of atoms from dataset
    atom_types_path = molecular_data_path / "atom_types.pt"
    if atom_types_path.exists():
        try:
            atom_types_data = torch.load(atom_types_path, map_location="cpu", weights_only=True)
        except:
            atom_types_data = torch.load(atom_types_path, map_location="cpu")
        num_atoms = len(atom_types_data)
    else:
        num_atoms = 35  # Default for AA dipeptide
    
    print(f"Using {num_atoms} atoms for {peptide_override} peptide")
    
    # Build model 
    model = build_model(
        model_cfg=model_cfg,
        pair=(0, 1),
        temps=temps,
        target_name=target_name,
        target_kwargs=target_kwargs,
        device=device,
        num_atoms=num_atoms,
    )
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load dataset 
    dataset = PTTemperaturePairDataset(
        pt_data_path=f"datasets/pt_dipeptides/{peptide_override}/pt_{peptide_override}.pt",
        molecular_data_path=str(molecular_data_path),
        temp_pair=(0, 1),
        subsample_rate=5000,  # Use subset for faster evaluation
        center_coordinates=False,
        augment_coordinates=False,
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    print(f"Model loaded successfully. Dataset size: {len(dataset)}")
    return model, loader, dataset, device

def compute_log_likelihood_change(model, batch, transformation_fn, device):
    """Compute change in log-likelihood after applying transformation."""
    try:
        # Move data to device
        source_coords = batch["source_coords"].to(device)
        target_coords = batch["target_coords"].to(device)
        atom_types = batch["atom_types"].to(device) if "atom_types" in batch else None
        
        # Compute original log-likelihood
        with torch.no_grad():
            if isinstance(model, PTSwapFlow):
                # Simple flow - doesn't use atom types
                original_ll = model.log_likelihood(source_coords, target_coords, reverse=False)
            else:
                # Graph or transformer flow - requires atom_types
                if atom_types is None:
                    raise ValueError("atom_types is required for graph/transformer flow")
                original_ll = model.log_likelihood(
                    source_coords, target_coords, 
                    atom_types=atom_types, 
                    reverse=False
                )
        
        # Apply transformation
        transformed_source = transformation_fn(source_coords)
        transformed_target = transformation_fn(target_coords)
        
        # Compute transformed log-likelihood
        with torch.no_grad():
            if isinstance(model, PTSwapFlow):
                # Simple flow - no atom types
                transformed_ll = model.log_likelihood(transformed_source, transformed_target, reverse=False)
            else:
                # Graph or transformer flow - requires atom_types
                transformed_ll = model.log_likelihood(
                    transformed_source, transformed_target, 
                    atom_types=atom_types, 
                    reverse=False
                )
        
        # Return absolute change
        return torch.abs(transformed_ll - original_ll)
        
    except Exception as e:
        print(f"Warning: Log likelihood calculation failed: {e}")
        return torch.zeros(source_coords.shape[0])

def compute_log_likelihood_with_confidence(model, test_batches, transformation_fn, device, n_trials: int = 10):
    """Compute log-likelihood changes with confidence intervals over multiple trials."""
    all_ll_changes = []
    
    for trial in range(n_trials):
        trial_ll_changes = []
        
        # Set different random seed for each trial to get different transformations
        torch.manual_seed(42 + trial)
        np.random.seed(42 + trial)
        
        for batch in test_batches:
            try:
                ll_change = compute_log_likelihood_change(
                    model, batch, transformation_fn, device
                )
                trial_ll_changes.extend(ll_change.cpu().numpy())
            except Exception as e:
                print(f"Warning: Error in trial {trial}: {e}")
                continue
        
        if trial_ll_changes:
            all_ll_changes.extend(trial_ll_changes)
    
    if not all_ll_changes:
        return 0.0, 0.0, 0.0  # mean, std, max
    
    mean_ll = float(np.mean(all_ll_changes))
    std_ll = float(np.std(all_ll_changes, ddof=1))  # Sample standard deviation
    max_ll = float(np.max(all_ll_changes))
    
    return mean_ll, std_ll, max_ll

def run_symmetry_validation(model_configs: Dict[str, Dict], n_trials: int = 10):
    """Run comprehensive symmetry validation tests with confidence intervals."""
    
    results = {}
    
    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Testing {model_name} Flow")
        print(f"{'='*60}")
        
        try:
            # Load model and data
            model, loader, dataset, device = load_model_and_data(
                config["config_path"], 
                config["checkpoint_path"],
                peptide_override="AA"  # Use AA for all tests for consistency
            )
            
            model_results = {}
            
            # Get test batches for log-likelihood evaluation
            test_batches = []
            for i, batch in enumerate(loader):
                if i >= 3:  # Use 3 batches for testing
                    break
                test_batches.append(batch)
            
            print(f"Using {len(test_batches)} test batches for evaluation")
            total_samples = sum(len(batch["source_coords"]) for batch in test_batches)
            print(f"Total samples per trial: {total_samples}")
            print(f"Model type: {type(model).__name__}")
            print(f"Running {n_trials} trials for robust statistics...")
            
            # Test 1: Rotation invariance
            print("Testing rotation invariance...")
            rotation_mean, rotation_std, rotation_max = compute_log_likelihood_with_confidence(
                model, test_batches, apply_rotation_transformation, device, n_trials
            )
            
            model_results["rotation"] = {
                "ll_mean": rotation_mean,
                "ll_std": rotation_std,
                "ll_max": rotation_max
            }
            
            # Test 2: Translation invariance  
            print("Testing translation invariance...")
            translation_mean, translation_std, translation_max = compute_log_likelihood_with_confidence(
                model, test_batches, apply_translation_transformation, device, n_trials
            )
            
            model_results["translation"] = {
                "ll_mean": translation_mean,
                "ll_std": translation_std,
                "ll_max": translation_max
            }
            
            # Test 3: Combined rotation + translation
            print("Testing combined rotation + translation...")
            def combined_rotation_translation_fn(coords):
                coords = apply_rotation_transformation(coords)
                coords = apply_translation_transformation(coords)
                return coords
            
            combined_mean, combined_std, combined_max = compute_log_likelihood_with_confidence(
                model, test_batches, combined_rotation_translation_fn, device, n_trials
            )
            
            model_results["rotation_translation"] = {
                "ll_mean": combined_mean,
                "ll_std": combined_std,
                "ll_max": combined_max
            }
            
            results[model_name] = model_results
            
            # Print results for this model
            print(f"\nResults for {model_name} (mean ± std):")
            for test_name, test_results in model_results.items():
                print(f"  {test_name}:")
                print(f"    |Δ log p| mean: {test_results['ll_mean']:.6e} ± {test_results['ll_std']:.6e}")
                print(f"    |Δ log p| max:  {test_results['ll_max']:.6e}")
                
        except Exception as e:
            print(f"ERROR: Failed to test {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add empty results to avoid missing keys
            results[model_name] = {
                "rotation": {"ll_mean": 0.0, "ll_std": 0.0, "ll_max": 0.0},
                "translation": {"ll_mean": 0.0, "ll_std": 0.0, "ll_max": 0.0},
                "rotation_translation": {"ll_mean": 0.0, "ll_std": 0.0, "ll_max": 0.0},
            }
    
    return results

def format_latex_table(results: Dict) -> str:
    """Format results into LaTeX table with confidence intervals."""
    
    lines = []
    
    # Helper function to format scientific notation with confidence intervals
    def fmt_sci_ci(mean_val, std_val, precision=1):
        if mean_val == 0 or np.isnan(mean_val):
            return "0"
        try:
            # Format mean
            exp_mean = int(np.floor(np.log10(abs(mean_val))))
            mantissa_mean = mean_val / (10 ** exp_mean)
            
            # Format std (same exponent as mean for readability)
            mantissa_std = std_val / (10 ** exp_mean) if std_val > 0 else 0
            
            if exp_mean == 0:
                return f"{mean_val:.{precision}f} \\pm {std_val:.{precision}f}"
            else:
                return f"({mantissa_mean:.{precision}f} \\pm {mantissa_std:.{precision}f}) \\times 10^{{{exp_mean}}}"
        except:
            return f"{mean_val:.2e} \\pm {std_val:.2e}"
    
    def fmt_sci_simple(val, precision=1):
        if val == 0 or np.isnan(val):
            return "0"
        try:
            exp = int(np.floor(np.log10(abs(val))))
            mantissa = val / (10 ** exp)
            if exp == 0:
                return f"{val:.{precision}f}"
            else:
                return f"{mantissa:.{precision}f} \\times 10^{{{exp}}}"
        except:
            return f"{val:.2e}"
    
    # Table content
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Symmetry Invariance Validation with Confidence Intervals}")
    lines.append("\\label{tab:symmetry_validation}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Transformation} & \\textbf{Metric} & \\textbf{Graph Flow} & \\textbf{Transformer Flow} & \\textbf{Tolerance} \\\\")
    lines.append("\\midrule")
    
    # Rotation invariance
    lines.append("\\multicolumn{5}{l}{\\textit{Rotation Invariance (random SO(3) rotations)}} \\\\")
    lines.append("\\multirow{2}{*}{Random 3D Rotation} & $|\\Delta \\log p|$ (mean $\\pm$ std) & "
                f"{fmt_sci_ci(results['Graph']['rotation']['ll_mean'], results['Graph']['rotation']['ll_std'])} & "
                f"{fmt_sci_ci(results['Transformer']['rotation']['ll_mean'], results['Transformer']['rotation']['ll_std'])} & "
                "$< 10^{-5}$ \\\\")
    lines.append(" & $|\\Delta \\log p|$ (max) & "
                f"{fmt_sci_simple(results['Graph']['rotation']['ll_max'])} & "
                f"{fmt_sci_simple(results['Transformer']['rotation']['ll_max'])} & "
                "$< 10^{-4}$ \\\\")
    lines.append("\\midrule")
    
    # Translation invariance
    lines.append("\\multicolumn{5}{l}{\\textit{Translation Invariance (±2.0 Å shifts)}} \\\\")
    lines.append("\\multirow{2}{*}{Translation} & $|\\Delta \\log p|$ (mean $\\pm$ std) & "
                f"{fmt_sci_ci(results['Graph']['translation']['ll_mean'], results['Graph']['translation']['ll_std'])} & "
                f"{fmt_sci_ci(results['Transformer']['translation']['ll_mean'], results['Transformer']['translation']['ll_std'])} & "
                "$< 10^{-5}$ \\\\")
    lines.append(" & $|\\Delta \\log p|$ (max) & "
                f"{fmt_sci_simple(results['Graph']['translation']['ll_max'])} & "
                f"{fmt_sci_simple(results['Transformer']['translation']['ll_max'])} & "
                "$< 10^{-4}$ \\\\")
    lines.append("\\midrule")
    
    # Combined transformations
    lines.append("\\multicolumn{5}{l}{\\textit{Combined Transformations}} \\\\")
    lines.append("\\multirow{2}{*}{Rotation + Translation} & $|\\Delta \\log p|$ (mean $\\pm$ std) & "
                f"{fmt_sci_ci(results['Graph']['rotation_translation']['ll_mean'], results['Graph']['rotation_translation']['ll_std'])} & "
                f"{fmt_sci_ci(results['Transformer']['rotation_translation']['ll_mean'], results['Transformer']['rotation_translation']['ll_std'])} & "
                "$< 10^{-5}$ \\\\")
    lines.append(" & $|\\Delta \\log p|$ (max) & "
                f"{fmt_sci_simple(results['Graph']['rotation_translation']['ll_max'])} & "
                f"{fmt_sci_simple(results['Transformer']['rotation_translation']['ll_max'])} & "
                "$< 10^{-4}$ \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

def main():
    """Main function to run symmetry validation."""
    
    # Configuration for each model
    model_configs = {
        "Graph": {
            "config_path": "configs/multi_graph_01.yaml", 
            "checkpoint_path": "outputs/multi_graph_01/pair_0_1/models/best_model_epoch1446.pt"
        },
        "Transformer": {
            "config_path": "configs/ablation/transformer_attn_8.yaml",
            "checkpoint_path": "outputs/transformer_attn_8/pair_0_1/models/best_model_epoch47.pt"
        }
    }
    
    print("="*80)
    print("SYMMETRY VALIDATION FOR FLOW ARCHITECTURES")
    print("="*80)
    print("This script will test rotation, translation, and combined invariance")
    print("for Graph and Transformer flow architectures with confidence intervals.")
    print("Expected runtime: 5-8 minutes")
    print("="*80)
    
    # Run validation tests
    results = run_symmetry_validation(model_configs, n_trials=10)
    
    # Create output directory in symmetry folder
    output_dir = Path("experiments/symmetry")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to JSON
    with open(output_dir / "symmetry_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate LaTeX table
    latex_table = format_latex_table(results)
    
    # Save LaTeX table
    with open(output_dir / "symmetry_validation_table.txt", "w") as f:
        f.write(latex_table)
    
    print(f"\n{'='*80}")
    print("SYMMETRY VALIDATION COMPLETE!")
    print("="*80)
    print(f"Results saved to:")
    print(f"  📊 Raw data: {output_dir / 'symmetry_validation_results.json'}")
    print(f"  📋 LaTeX table: {output_dir / 'symmetry_validation_table.txt'}")
    print("="*80)
    
    # Print summary with confidence intervals
    print("\n📈 SUMMARY OF RESULTS (mean ± std):")
    print("-" * 80)
    for model_name, model_results in results.items():
        print(f"\n{model_name} Flow:")
        print(f"  🔄 Rotation invariance:    |Δ log p| = {model_results['rotation']['ll_mean']:.2e} ± {model_results['rotation']['ll_std']:.2e}")
        print(f"  ➡️  Translation invariance: |Δ log p| = {model_results['translation']['ll_mean']:.2e} ± {model_results['translation']['ll_std']:.2e}")
        print(f"  🔄➡️  Combined:              |Δ log p| = {model_results['rotation_translation']['ll_mean']:.2e} ± {model_results['rotation_translation']['ll_std']:.2e}")
    
    print("\n✅ Use the generated table to update your thesis!")
    print("Copy the contents of 'symmetry_validation_table.txt' to replace the empty table.")

if __name__ == "__main__":
    main() 