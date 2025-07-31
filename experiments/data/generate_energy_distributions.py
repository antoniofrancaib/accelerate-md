#!/usr/bin/env python3
"""
Generate energy distribution plots across temperature ladder for thesis.

This script creates:
- Figure 4.X: Energy Distributions Across Temperature Ladder
- Violin plots or histograms showing energy distributions for each temperature
- Shows progressive broadening and higher means at higher temperatures
- Validates proper thermodynamic behavior

The output is saved as energy_distributions.pdf and energy_distributions.png
"""

import json
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import sys

# Add src to path
sys.path.append('src')

from accelmd.targets import build_target


# Working helper functions from generate_dataset_validation_tables.py
def _load_temps(meta_path: str | Path) -> List[float]:
    """Return list of temperatures (Kelvin) stored in *meta_path* JSON."""
    meta = json.loads(Path(meta_path).read_text())
    temps = meta.get("temperatures") or meta.get("temperature_values")
    if temps is None:
        raise KeyError("temperatures not found in meta JSON.")
    # Older ALDP meta files store scale factors (≤5).  Convert those to Kelvin.
    if max(temps) < 100:  # heuristic – everything below 100 interpreted as β-scale
        temps = [300.0 * float(t) for t in temps]
    return [float(t) for t in temps]


def _build_target(target_name: str, temperature: float, **kwargs):
    """Thin wrapper around src.accelmd.targets.build_target with minimal args."""
    if target_name == "aldp":
        return build_target("aldp", temperature=temperature)
    elif target_name == "dipeptide":
        return build_target("dipeptide", temperature=temperature, **kwargs)
    else:
        raise ValueError(f"Unsupported target '{target_name}'.")


def _energy(target, coords: torch.Tensor) -> torch.Tensor:  # kJ/mol
    """Potential energy helper matching evaluation.swap_acceptance logic."""
    if hasattr(target, "potential_energy"):
        return target.potential_energy(coords)
    beta = getattr(target, "beta", 1.0)
    return (-target.log_prob(coords)) / beta


def _sample_coords(
    traj: torch.Tensor,  # [T,R,F,dim]
    temp_idx: int,
    burn_in: int,
    n_samples: int,
) -> torch.Tensor:  # [n_samples, dim]
    """Randomly draw *n_samples* coordinates from replica *temp_idx*."""
    T, R, F, D = traj.shape
    coords = torch.empty((n_samples, D), dtype=traj.dtype)
    for i in range(n_samples):
        r = random.randrange(R)
        f = random.randrange(burn_in, F)
        coords[i] = traj[temp_idx, r, f]
    return coords


def calculate_energy_distributions(peptide_name: str, n_samples: int = 2000) -> Dict[float, np.ndarray]:
    """Calculate energy distributions for all temperatures of a peptide system.
    
    Returns:
        Dict mapping temperature to energy array
    """
    print(f"Calculating energy distributions for {peptide_name}...")
    
    peptide_dir = Path("datasets/pt_dipeptides") / peptide_name
    pt_file = peptide_dir / f"pt_{peptide_name}.pt"
    meta_file = peptide_dir / "meta.json"
    
    # Load temperatures
    temps = _load_temps(meta_file)
    
    # Load trajectory
    raw = torch.load(pt_file.as_posix(), map_location="cpu", mmap=True)
    traj = raw["trajectory"] if isinstance(raw, dict) else raw  # [T,R,F,dim]
    
    # Get burn-in
    with open(meta_file) as fh:
        burn_in = json.load(fh).get("burn_in", 0)
    burn_in = int(burn_in)
    
    # Determine target type and setup kwargs
    if peptide_name.upper() == "AX":
        target_name = "aldp"
        extra_kw = {}
    else:
        target_name = "dipeptide"
        pdb_path = peptide_dir / "ref.pdb"
        extra_kw = {"pdb_path": str(pdb_path), "env": "implicit"}
    
    energy_distributions = {}
    
    # Calculate energies for each temperature
    for temp_idx, temperature in enumerate(temps):
        print(f"  Temperature {temperature:.1f}K...")
        
        try:
            # Build target for this temperature
            target = _build_target(target_name, temperature, **extra_kw)
            
            # Sample coordinates for this temperature
            coords = _sample_coords(traj, temp_idx, burn_in, n_samples)
            
            # Calculate energies
            with torch.no_grad():
                energies = _energy(target, coords).cpu().numpy()
            
            energy_distributions[temperature] = energies
            
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            print(f"    Mean: {mean_energy:.1f} kJ/mol, Std: {std_energy:.1f} kJ/mol")
            
        except Exception as e:
            print(f"    Warning: Failed to calculate energies for T={temperature}K: {e}")
            continue
    
    return energy_distributions


def create_violin_plot(energy_data: Dict[str, Dict[float, np.ndarray]], output_path: str):
    """Create violin plots showing energy distributions across temperatures."""
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Determine number of peptides and create subplots
    n_peptides = len(energy_data)
    if n_peptides <= 3:
        ncols = n_peptides
        nrows = 1
        figsize = (6 * ncols, 5)
    else:
        ncols = 3
        nrows = (n_peptides + 2) // 3
        figsize = (18, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for idx, (peptide_name, distributions) in enumerate(energy_data.items()):
        ax = axes[idx]
        
        # Prepare data for violin plot
        temperatures = sorted(distributions.keys())
        energy_arrays = [distributions[temp] for temp in temperatures]
        temp_labels = [f"{temp:.0f}K" for temp in temperatures]
        
        # Create violin plot
        parts = ax.violinplot(energy_arrays, positions=range(len(temperatures)), 
                             showmeans=True, showmedians=True)
        
        # Customize violin plot appearance
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        # Color code by temperature (cooler = blue, hotter = red)
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
        
        # Customize axes
        ax.set_xticks(range(len(temperatures)))
        ax.set_xticklabels(temp_labels, rotation=45)
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Potential Energy (kJ/mol)')
        ax.set_title(f'{peptide_name} - Energy Distributions')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        means = [np.mean(energies) for energies in energy_arrays]
        stds = [np.std(energies) for energies in energy_arrays]
        
        # Show temperature trend
        energy_range = max(means) - min(means)
        std_range = max(stds) - min(stds)
        
        ax.text(0.02, 0.98, f'ΔE: {energy_range:.1f} kJ/mol\nΔσ: {std_range:.1f} kJ/mol', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(n_peptides, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    
    print(f"Violin plots saved as {output_path}.pdf and {output_path}.png")
    
    return fig


def create_histogram_plot(energy_data: Dict[str, Dict[float, np.ndarray]], output_path: str):
    """Create histogram plots showing energy distributions across temperatures."""
    
    plt.style.use('default')
    
    # Create separate figure for each peptide
    for peptide_name, distributions in energy_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        temperatures = sorted(distributions.keys())
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))
        
        # Create overlapping histograms with transparency
        for temp, color in zip(temperatures, colors):
            energies = distributions[temp]
            ax.hist(energies, bins=50, alpha=0.6, density=True, 
                   label=f'{temp:.0f}K (μ={np.mean(energies):.1f}, σ={np.std(energies):.1f})',
                   color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Potential Energy (kJ/mol)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{peptide_name} - Energy Distributions Across Temperature Ladder')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual histogram
        hist_output = f"{output_path}_{peptide_name}_histogram"
        plt.savefig(f"{hist_output}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{hist_output}.png", dpi=300, bbox_inches='tight')
        
        print(f"Histogram for {peptide_name} saved as {hist_output}.pdf and {hist_output}.png")
        
        plt.close()


def create_summary_statistics_table(energy_data: Dict[str, Dict[float, np.ndarray]]):
    """Create a summary table of energy statistics across temperatures."""
    
    print("\nSummary: Energy Statistics Across Temperature Ladder")
    print("=" * 80)
    
    for peptide_name, distributions in energy_data.items():
        print(f"\n{peptide_name} Energy Statistics:")
        print("-" * 40)
        print(f"{'Temperature (K)':<15} {'Mean (kJ/mol)':<15} {'Std (kJ/mol)':<15} {'Range (kJ/mol)':<20}")
        print("-" * 65)
        
        temperatures = sorted(distributions.keys())
        
        for temp in temperatures:
            energies = distributions[temp]
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            energy_range = np.max(energies) - np.min(energies)
            
            print(f"{temp:<15.1f} {mean_energy:<15.1f} {std_energy:<15.1f} {energy_range:<20.1f}")
        
        # Calculate trends
        means = [np.mean(distributions[temp]) for temp in temperatures]
        stds = [np.std(distributions[temp]) for temp in temperatures]
        
        mean_increase = means[-1] - means[0]
        std_increase = stds[-1] - stds[0]
        
        print(f"\nTrends:")
        print(f"  Mean energy increase: {mean_increase:.1f} kJ/mol")
        print(f"  Std dev increase: {std_increase:.1f} kJ/mol")
        print(f"  Expected trend: Higher T → Higher mean, Broader distribution")
        
        # Validate thermodynamic behavior
        mean_trend_correct = mean_increase > 0
        std_trend_correct = std_increase > 0
        
        print(f"  Thermodynamic validation:")
        print(f"    Mean increases with T: {'✓' if mean_trend_correct else '✗'}")
        print(f"    Std increases with T: {'✓' if std_trend_correct else '✗'}")


def main():
    """Main function to generate energy distribution plots."""
    print("Generating Energy Distribution Plots")
    print("=" * 50)
    
    # Select representative peptides for plotting
    # Use fewer peptides to keep plots readable
    representative_peptides = ['AA', 'AS', 'AK']  # Training set
    
    print(f"Processing peptides: {representative_peptides}")
    
    # Calculate energy distributions for each peptide
    all_energy_data = {}
    
    for peptide in representative_peptides:
        try:
            energy_distributions = calculate_energy_distributions(peptide, n_samples=1500)
            if energy_distributions:
                all_energy_data[peptide] = energy_distributions
                print(f"✓ Successfully processed {peptide}")
            else:
                print(f"✗ No data collected for {peptide}")
        except Exception as e:
            print(f"✗ Failed to process {peptide}: {e}")
            continue
    
    if not all_energy_data:
        print("No energy data collected. Exiting.")
        return
    
    print(f"\nSuccessfully collected data for {len(all_energy_data)} peptides")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Create violin plots
    violin_fig = create_violin_plot(all_energy_data, "energy_distributions_violin")
    
    # Create histogram plots
    create_histogram_plot(all_energy_data, "energy_distributions")
    
    # Generate summary statistics
    create_summary_statistics_table(all_energy_data)
    
    print("\n" + "=" * 50)
    print("Energy distribution analysis complete!")
    print("Files generated:")
    print("  - energy_distributions_violin.pdf/png (violin plots)")
    print("  - energy_distributions_*_histogram.pdf/png (individual histograms)")
    print("\nThese plots demonstrate:")
    print("  1. Progressive broadening of energy distributions at higher T")
    print("  2. Higher mean energies at higher temperatures") 
    print("  3. Proper thermodynamic behavior across the temperature ladder")


if __name__ == "__main__":
    main() 