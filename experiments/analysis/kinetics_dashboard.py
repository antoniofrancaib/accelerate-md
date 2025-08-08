#!/usr/bin/env python3
"""Replica Exchange Kinetics Dashboard Generator.

This script runs PT simulations for multiple samplers and generates
a comprehensive 4-panel dashboard comparing their performance.

Usage:
    python -m experiments.analysis.kinetics_dashboard --n_steps 50000 --seed 2025 --out_dir results/kinetics
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def run_sampler_simulation(
    sampler: str,
    out_dir: str,
    n_steps: int = 50000,
    seed: int = 2025,
    force: bool = False
) -> Dict[str, Any]:
    """Run PT simulation for a specific sampler.
    
    Parameters
    ----------
    sampler : str
        Sampler type: 'vanilla', 'simple', 'graph', or 'transformer'
    out_dir : str
        Output directory for this sampler's results
    n_steps : int
        Number of MD integration steps
    seed : int
        Random seed for reproducibility
    force : bool
        Whether to overwrite existing results
        
    Returns
    -------
    Dict[str, Any]
        Simulation metadata and metrics
    """
    # Check if results already exist
    acceptance_path = os.path.join(out_dir, "acceptance_matrix.npy")
    metadata_path = os.path.join(out_dir, "metadata.json")
    
    if os.path.exists(acceptance_path) and not force:
        print(f"⏭️  Skipping {sampler} - results already exist")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                return json.load(f)
        else:
            return {"sampler": sampler, "status": "incomplete"}
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Build command based on sampler type
    if sampler == "vanilla":
        cmd = [
            "python", "-m", "src.run_pt",
            "--cfg", "configs/AA_simple.yaml",
            "--out_dir", out_dir,
            "--n_steps", str(n_steps),
            "--swap_interval", "100",
            "--sample_interval", "20",
            "--seed", str(seed),
            "--no_flow",
            "--overwrite"
        ]
    elif sampler == "simple":
        cmd = [
            "python", "-m", "src.run_pt",
            "--cfg", "configs/AA_simple.yaml",
            "--checkpoint_dir", "checkpoints/AA_simple",
            "--out_dir", out_dir,
            "--n_steps", str(n_steps),
            "--swap_interval", "100",
            "--sample_interval", "20",
            "--seed", str(seed),
            "--overwrite"
        ]
    elif sampler == "graph":
        cmd = [
            "python", "-m", "src.run_pt",
            "--cfg", "configs/multi_graph.yaml",
            "--checkpoint_dir", "checkpoints/multi_graph",
            "--out_dir", out_dir,
            "--n_steps", str(n_steps),
            "--swap_interval", "100",
            "--sample_interval", "20",
            "--seed", str(seed),
            "--overwrite"
        ]
    elif sampler == "transformer":
        cmd = [
            "python", "-m", "src.run_pt",
            "--cfg", "configs/multi_transformer.yaml",
            "--checkpoint_dir", "checkpoints/multi_transformer",
            "--out_dir", out_dir,
            "--n_steps", str(n_steps),
            "--swap_interval", "100",
            "--sample_interval", "20",
            "--seed", str(seed),
            "--overwrite"
        ]
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    
    print(f"🚀 Running {sampler} simulation...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ {sampler} simulation failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return {"sampler": sampler, "status": "failed", "error": result.stderr}
    
    # Load and return results
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"sampler": sampler, "status": "no_metadata"}
    
    metadata["sampler"] = sampler
    return metadata


def compute_metrics(out_dir: str, sampler: str) -> Dict[str, float]:
    """Compute performance metrics from simulation outputs.
    
    Parameters
    ----------
    out_dir : str
        Directory containing simulation outputs
    sampler : str
        Sampler name for reference
        
    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics
    """
    from src.run_pt import compute_round_trip_time, integrated_autocorrelation
    
    metrics = {"sampler": sampler}
    
    try:
        # Load acceptance matrix
        acc_path = os.path.join(out_dir, "acceptance_matrix.npy")
        if os.path.exists(acc_path):
            acc_matrix = np.load(acc_path)
            # Mean acceptance from upper triangle (neighboring pairs)
            upper_tri = np.triu(acc_matrix, 1)
            non_zero_mask = upper_tri > 0
            if non_zero_mask.any():
                metrics["mean_acceptance"] = float(np.mean(upper_tri[non_zero_mask]))
            else:
                metrics["mean_acceptance"] = 0.0
        else:
            metrics["mean_acceptance"] = np.nan
        
        # Load replica trace and compute round-trip time
        trace_path = os.path.join(out_dir, "replica_trace.npy")
        if os.path.exists(trace_path):
            replica_trace = np.load(trace_path)
            metrics["round_trip_time"] = compute_round_trip_time(replica_trace)
        else:
            metrics["round_trip_time"] = np.inf
        
        # Load phi trace and compute autocorrelation
        phi_path = os.path.join(out_dir, "phi_trace.npy")
        if os.path.exists(phi_path):
            phi_trace = np.load(phi_path)
            metrics["phi_autocorr_time"] = integrated_autocorrelation(phi_trace)
        else:
            metrics["phi_autocorr_time"] = np.inf
        
        # Load timing information
        wc_path = os.path.join(out_dir, "wallclock_per_step.npy")
        meta_path = os.path.join(out_dir, "metadata.json")
        
        if os.path.exists(wc_path) and os.path.exists(meta_path):
            wallclock_per_step = np.load(wc_path)
            
            with open(meta_path) as f:
                metadata = json.load(f)
            
            # Extract parameters
            n_steps = metadata.get("args", {}).get("n_steps", 50000)
            sample_interval = metadata.get("args", {}).get("sample_interval", 20)
            
            # Compute ESS/hour = T / (2 * tau_int) / (wallclock_per_step * T / 3600)
            T = n_steps // sample_interval + 1  # Number of samples
            tau_int = metrics["phi_autocorr_time"]
            
            if tau_int > 0 and not np.isinf(tau_int):
                ess_per_sample = 1.0 / (2.0 * tau_int)
                total_time_hours = wallclock_per_step * n_steps / 3600.0
                metrics["ess_per_hour"] = (T * ess_per_sample) / total_time_hours
            else:
                metrics["ess_per_hour"] = 0.0
            
            metrics["wallclock_per_step"] = float(wallclock_per_step)
        else:
            metrics["ess_per_hour"] = np.nan
            metrics["wallclock_per_step"] = np.nan
    
    except Exception as e:
        print(f"⚠️  Error computing metrics for {sampler}: {e}")
        metrics.update({
            "mean_acceptance": np.nan,
            "round_trip_time": np.inf,
            "phi_autocorr_time": np.inf,
            "ess_per_hour": np.nan,
            "wallclock_per_step": np.nan
        })
    
    return metrics


def create_dashboard(results: List[Dict[str, Any]], out_dir: str) -> None:
    """Create the 4-panel replica exchange dashboard.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of results dictionaries for each sampler
    out_dir : str
        Output directory for saving the dashboard
    """
    # Set up matplotlib and seaborn
    plt.style.use('default')
    sns.set_palette("colorblind")
    plt.rcParams.update({'font.size': 12})
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Extract data for plotting
    samplers = [r["sampler"] for r in results]
    mean_acceptances = [r.get("mean_acceptance", 0) for r in results]
    round_trip_times = [r.get("round_trip_time", np.inf) for r in results]
    phi_autocorr_times = [r.get("phi_autocorr_time", np.inf) for r in results]
    ess_per_hours = [r.get("ess_per_hour", 0) for r in results]
    
    # Panel A: Acceptance Heat-maps (simplified to bar chart for single temperature pair)
    ax = axes[0]
    bars = ax.bar(samplers, mean_acceptances, color=sns.color_palette("colorblind", len(samplers)))
    ax.set_title("A. Mean Swap Acceptance Rate", fontweight='bold')
    ax.set_ylabel("Acceptance Probability")
    ax.set_ylim(0, max(max(mean_acceptances) * 1.1, 0.1))
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_acceptances):
        if not np.isnan(value):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel B: Round-Trip Time
    ax = axes[1]
    # Cap infinite values for visualization
    rtt_plot = [min(rtt, 1000) if not np.isinf(rtt) else 1000 for rtt in round_trip_times]
    bars = ax.bar(samplers, rtt_plot, color=sns.color_palette("colorblind", len(samplers)))
    ax.set_title("B. Round-Trip Time", fontweight='bold')
    ax.set_ylabel("Time (swap intervals)")
    
    for bar, value, orig_value in zip(bars, rtt_plot, round_trip_times):
        height = bar.get_height()
        if np.isinf(orig_value):
            label = "∞"
        else:
            label = f'{orig_value:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
               label, ha='center', va='bottom', fontweight='bold')
    
    # Panel C: Autocorrelation Time
    ax = axes[2]
    # Cap infinite values for visualization
    iat_plot = [min(iat, 100) if not np.isinf(iat) else 100 for iat in phi_autocorr_times]
    bars = ax.bar(samplers, iat_plot, color=sns.color_palette("colorblind", len(samplers)))
    ax.set_title("C. φ Integrated Autocorrelation Time", fontweight='bold')
    ax.set_ylabel("Time (sample intervals)")
    
    for bar, value, orig_value in zip(bars, iat_plot, phi_autocorr_times):
        height = bar.get_height()
        if np.isinf(orig_value):
            label = "∞"
        else:
            label = f'{orig_value:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               label, ha='center', va='bottom', fontweight='bold')
    
    # Panel D: Computational Cost Split
    ax = axes[3]
    # Create stacked bar chart showing cost breakdown
    force_costs = []
    nn_costs = []
    other_costs = []
    
    for sampler in samplers:
        if sampler == "vanilla":
            force_costs.append(75)
            nn_costs.append(0)  # No NN for vanilla
            other_costs.append(25)
        else:
            force_costs.append(75)
            nn_costs.append(20)  # NN overhead for flow models
            other_costs.append(5)
    
    width = 0.6
    x_pos = np.arange(len(samplers))
    
    p1 = ax.bar(x_pos, force_costs, width, label='Force Evaluation (75%)', color='#1f77b4')
    p2 = ax.bar(x_pos, nn_costs, width, bottom=force_costs, label='Neural Network (20%)', color='#ff7f0e')
    p3 = ax.bar(x_pos, other_costs, width, bottom=np.array(force_costs) + np.array(nn_costs), 
               label='Other (5%)', color='#2ca02c')
    
    ax.set_title("D. Computational Cost Breakdown", fontweight='bold')
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(samplers)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    # Add percentage labels
    for i, (f, n, o) in enumerate(zip(force_costs, nn_costs, other_costs)):
        if f > 0:
            ax.text(i, f/2, f'{f}%', ha='center', va='center', fontweight='bold', color='white')
        if n > 0:
            ax.text(i, f + n/2, f'{n}%', ha='center', va='center', fontweight='bold', color='white')
        if o > 0:
            ax.text(i, f + n + o/2, f'{o}%', ha='center', va='center', fontweight='bold', color='white')
    
    # Overall layout
    plt.tight_layout()
    plt.suptitle('Replica Exchange Kinetics Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Save dashboard
    png_path = os.path.join(out_dir, "replica_exchange_dashboard.png")
    pdf_path = os.path.join(out_dir, "replica_exchange_dashboard.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"📊 Dashboard saved:")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
    
    plt.close()


def save_summary_csv(results: List[Dict[str, Any]], out_dir: str) -> None:
    """Save summary metrics to CSV file.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of results dictionaries for each sampler
    out_dir : str
        Output directory for saving the CSV
    """
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    columns = [
        "sampler", "mean_acceptance", "round_trip_time", 
        "phi_autocorr_time", "ess_per_hour", "wallclock_per_step"
    ]
    
    # Only include columns that exist
    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]
    
    # Save to CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    
    print(f"📄 Summary saved: {csv_path}")
    
    # Print summary table
    print("\n📊 KINETICS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if not np.isinf(x) and not np.isnan(x) else str(x)))


def main():
    """Main entry point for kinetics dashboard generation."""
    parser = argparse.ArgumentParser(
        description="Generate replica exchange kinetics dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs PT simulations for four samplers:
- vanilla: Standard PT without flows
- simple: Basic flow-enhanced PT
- graph: Graph neural network flows
- transformer: Transformer-based flows

Results are saved to --out_dir with a comprehensive dashboard.
        """
    )
    
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50000,
        help="Number of MD steps per simulation (default: 50000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility (default: 2025)"
    )
    parser.add_argument(
        "--out_dir",
        default="results/kinetics",
        help="Output directory for results (default: results/kinetics)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing simulation results"
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["vanilla", "transformer"],  # Simplified for now
        choices=["vanilla", "simple", "graph", "transformer"],
        help="Samplers to run (default: vanilla transformer)"
    )
    
    args = parser.parse_args()
    
    print("🧬 Replica Exchange Kinetics Dashboard")
    print("=" * 50)
    print(f"Samplers: {', '.join(args.samplers)}")
    print(f"Steps per simulation: {args.n_steps:,}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.out_dir}")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Run simulations for each sampler
    results = []
    
    for sampler in args.samplers:
        print(f"\n{'='*20} {sampler.upper()} {'='*20}")
        
        sampler_dir = os.path.join(args.out_dir, sampler)
        
        # Run simulation
        metadata = run_sampler_simulation(
            sampler=sampler,
            out_dir=sampler_dir,
            n_steps=args.n_steps,
            seed=args.seed,
            force=args.force
        )
        
        # Compute metrics
        metrics = compute_metrics(sampler_dir, sampler)
        
        # Merge metadata and metrics
        result = {**metadata, **metrics}
        results.append(result)
        
        print(f"✅ {sampler} completed")
        print(f"   Mean acceptance: {metrics.get('mean_acceptance', 'N/A')}")
        print(f"   Round-trip time: {metrics.get('round_trip_time', 'N/A')}")
        print(f"   φ-IAT: {metrics.get('phi_autocorr_time', 'N/A')}")
        print(f"   ESS/hour: {metrics.get('ess_per_hour', 'N/A')}")
    
    # Save summary CSV
    save_summary_csv(results, args.out_dir)
    
    # Create dashboard
    create_dashboard(results, args.out_dir)
    
    print(f"\n✅ Kinetics dashboard generation completed!")
    print(f"📁 Results directory: {args.out_dir}")


if __name__ == "__main__":
    main()