#!/usr/bin/env python3
"""
Extended analysis of free energy convergence results.
Creates additional plots to visualize the sampling acceleration.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results():
    """Load convergence results."""
    with open('experiments/ESS/convergence_results.pkl', 'rb') as f:
        return pickle.load(f)

def plot_improvement_factors(data, output_dir):
    """Plot improvement factors over time."""
    time_points = data['time_points'][:-1]  # Exclude final point
    vanilla_errors = data['vanilla_errors'][:-1]
    flow_errors = data['flow_errors'][:-1]
    
    improvement_factors = vanilla_errors / flow_errors
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Improvement factors
    ax1.plot(time_points, improvement_factors, 'g-o', linewidth=3, markersize=8)
    ax1.set_xlabel('Simulation Time (steps)', fontsize=12)
    ax1.set_ylabel('Speedup Factor (Vanilla/Flow)', fontsize=12)
    ax1.set_title('Convergence Speedup Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1.0, max(improvement_factors) * 1.1)
    
    # Add annotations
    for i, (t, f) in enumerate(zip(time_points, improvement_factors)):
        ax1.annotate(f'{f:.2f}×', (t, f), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    # Right: Cumulative sampling efficiency
    efficiency_vanilla = 1 / vanilla_errors  # Higher is better
    efficiency_flow = 1 / flow_errors
    
    ax2.semilogy(time_points, efficiency_vanilla, 'b-o', label='Vanilla PT', linewidth=2)
    ax2.semilogy(time_points, efficiency_flow, 'r-s', label='Transformer Flow PT', linewidth=2)
    ax2.set_xlabel('Simulation Time (steps)', fontsize=12)
    ax2.set_ylabel('Sampling Efficiency (1/L2 Error)', fontsize=12)
    ax2.set_title('Sampling Efficiency Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_surface_evolution(data, output_dir):
    """Plot evolution of free energy surfaces."""
    vanilla_fes = data['vanilla_fes']
    flow_fes = data['flow_fes']
    time_points = data['time_points']
    phi_centers = data['phi_centers']
    psi_centers = data['psi_centers']
    
    # Create 2x3 subplot grid showing evolution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Show first 3 time points for each method
    for i in range(3):
        if i < len(vanilla_fes) - 1:  # Exclude final (reference)
            # Vanilla PT surfaces
            im1 = axes[0, i].contourf(phi_centers, psi_centers, vanilla_fes[i], 
                                    levels=20, cmap='viridis')
            axes[0, i].set_title(f'Vanilla PT - {time_points[i]} steps', fontweight='bold')
            axes[0, i].set_xlabel('φ (radians)')
            axes[0, i].set_ylabel('ψ (radians)')
            plt.colorbar(im1, ax=axes[0, i], label='Free Energy (kT)')
            
            # Flow PT surfaces  
            im2 = axes[1, i].contourf(phi_centers, psi_centers, flow_fes[i], 
                                    levels=20, cmap='viridis')
            axes[1, i].set_title(f'Transformer Flow PT - {time_points[i]} steps', fontweight='bold')
            axes[1, i].set_xlabel('φ (radians)')
            axes[1, i].set_ylabel('ψ (radians)')
            plt.colorbar(im2, ax=axes[1, i], label='Free Energy (kT)')
    
    plt.suptitle('Free Energy Surface Evolution: AA Dipeptide', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'surface_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_decomposition(data, output_dir):
    """Analyze error components and convergence rates."""
    time_points = data['time_points'][:-1]
    vanilla_errors = data['vanilla_errors'][:-1]
    flow_errors = data['flow_errors'][:-1]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Error vs time (linear scale)
    ax1.plot(time_points, vanilla_errors, 'b-o', label='Vanilla PT', linewidth=2)
    ax1.plot(time_points, flow_errors, 'r-s', label='Transformer Flow PT', linewidth=2)
    ax1.set_xlabel('Simulation Time (steps)')
    ax1.set_ylabel('L2 Error (kT)')
    ax1.set_title('Convergence: Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence rate analysis
    if len(time_points) > 2:
        # Fit exponential decay: error = A * exp(-λ * t)
        log_vanilla = np.log(vanilla_errors[vanilla_errors > 0])
        log_flow = np.log(flow_errors[flow_errors > 0])
        t_valid_v = time_points[:len(log_vanilla)]
        t_valid_f = time_points[:len(log_flow)]
        
        if len(log_vanilla) > 1:
            slope_v = np.polyfit(t_valid_v, log_vanilla, 1)[0]
            tau_v = -1/slope_v if slope_v < 0 else float('inf')
        else:
            tau_v = float('inf')
            
        if len(log_flow) > 1:
            slope_f = np.polyfit(t_valid_f, log_flow, 1)[0]
            tau_f = -1/slope_f if slope_f < 0 else float('inf')
        else:
            tau_f = float('inf')
        
        ax2.semilogy(t_valid_v, np.exp(log_vanilla), 'b-o', label=f'Vanilla (τ={tau_v:.0f})')
        ax2.semilogy(t_valid_f, np.exp(log_flow), 'r-s', label=f'Flow (τ={tau_f:.0f})')
        ax2.set_xlabel('Simulation Time (steps)')
        ax2.set_ylabel('L2 Error (kT)')
        ax2.set_title('Exponential Decay Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Relative improvement over time
    relative_improvement = (vanilla_errors - flow_errors) / vanilla_errors * 100
    ax3.plot(time_points, relative_improvement, 'g-^', linewidth=2, markersize=8)
    ax3.set_xlabel('Simulation Time (steps)')
    ax3.set_ylabel('Relative Improvement (%)')
    ax3.set_title('Flow PT Improvement Over Vanilla')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error difference (absolute gain)
    error_difference = vanilla_errors - flow_errors
    ax4.bar(time_points, error_difference, alpha=0.7, color='purple', width=200)
    ax4.set_xlabel('Simulation Time (steps)')
    ax4.set_ylabel('Error Reduction (kT)')
    ax4.set_title('Absolute Error Reduction')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(data, output_dir):
    """Create a summary table of results."""
    time_points = data['time_points'][:-1]
    vanilla_errors = data['vanilla_errors'][:-1]
    flow_errors = data['flow_errors'][:-1]
    
    # Calculate metrics
    improvements = vanilla_errors / flow_errors
    relative_gains = (vanilla_errors - flow_errors) / vanilla_errors * 100
    
    print("\n" + "="*80)
    print("FREE ENERGY CONVERGENCE ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Time (steps)':<12} {'Vanilla L2':<12} {'Flow L2':<12} {'Speedup':<10} {'Rel. Gain':<12}")
    print("-" * 80)
    
    for i, t in enumerate(time_points):
        print(f"{t:<12} {vanilla_errors[i]:<12.3f} {flow_errors[i]:<12.3f} "
              f"{improvements[i]:<10.2f}× {relative_gains[i]:<12.1f}%")
    
    print("-" * 80)
    print(f"AVERAGE SPEEDUP: {np.mean(improvements):.2f}×")
    print(f"FINAL SPEEDUP: {improvements[-1]:.2f}×")
    print(f"MAX SPEEDUP: {np.max(improvements):.2f}×")
    print("="*80)

def main():
    """Run extended analysis."""
    print("Loading convergence results...")
    data = load_results()
    
    output_dir = Path('experiments/ESS')
    
    print("Creating improvement factor plots...")
    plot_improvement_factors(data, output_dir)
    
    print("Creating surface evolution plots...")
    plot_surface_evolution(data, output_dir)
    
    print("Creating error decomposition analysis...")
    plot_error_decomposition(data, output_dir)
    
    print("Generating summary table...")
    create_summary_table(data, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Files created:")
    print("  - improvement_analysis.png")
    print("  - surface_evolution.png") 
    print("  - error_decomposition.png")

if __name__ == "__main__":
    main()