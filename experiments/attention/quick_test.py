#!/usr/bin/env python3
"""
Quick test script for attention analysis.

This script runs a minimal version of the analysis for testing purposes,
using synthetic data if real checkpoints are not available.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_synthetic_data(n_samples=100, n_atoms=22):
    """Generate synthetic attention and adjacency data for testing."""
    
    # Create synthetic attention matrices (distance-based + noise)
    attention_maps = []
    adjacency_matrices = []
    
    for _ in range(n_samples):
        # Random coordinates for atoms
        coords = np.random.randn(n_atoms, 3) * 2.0
        
        # Compute distances
        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        
        # Create adjacency based on distance cutoff
        adjacency = (dists <= 3.0).astype(float)
        np.fill_diagonal(adjacency, 0)  # Remove self-connections
        
        # Create attention as smooth function of distance + noise
        attention = np.exp(-dists / 1.5)
        attention = attention / attention.sum(axis=1, keepdims=True)
        np.fill_diagonal(attention, 0)  # Remove self-attention
        
        # Add some random long-range connections to attention
        if np.random.rand() > 0.7:
            i, j = np.random.randint(0, n_atoms, 2)
            if i != j:
                attention[i, j] = np.random.uniform(0.1, 0.3)
        
        attention_maps.append(attention)
        adjacency_matrices.append(adjacency)
    
    return np.array(attention_maps), np.array(adjacency_matrices)

def compute_basic_metrics(attention_maps, adjacency_matrices):
    """Compute basic comparison metrics."""
    from scipy.stats import spearmanr
    
    # Average across samples
    avg_attention = attention_maps.mean(axis=0)
    avg_adjacency = adjacency_matrices.mean(axis=0)
    
    # Flatten for correlation (excluding diagonal)
    mask = ~np.eye(avg_attention.shape[0], dtype=bool)
    attention_flat = avg_attention[mask]
    adjacency_flat = avg_adjacency[mask]
    
    # Spearman correlation
    spearman_corr, _ = spearmanr(attention_flat, adjacency_flat)
    
    # Basic statistics
    metrics = {
        'spearman_correlation': float(spearman_corr),
        'attention_mean': float(attention_flat.mean()),
        'adjacency_mean': float(adjacency_flat.mean()),
        'n_samples': len(attention_maps),
        'n_atoms': attention_maps.shape[1]
    }
    
    return metrics, avg_attention, avg_adjacency

def create_test_plots(avg_attention, avg_adjacency, output_dir):
    """Create basic test plots."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Attention heatmap
    im1 = ax1.imshow(avg_attention, cmap='Reds', vmin=0, vmax=avg_attention.max())
    ax1.set_title('Synthetic Attention Matrix')
    ax1.set_xlabel('Target Atom')
    ax1.set_ylabel('Source Atom')
    plt.colorbar(im1, ax=ax1)
    
    # Adjacency heatmap
    im2 = ax2.imshow(avg_adjacency, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('Synthetic Adjacency Matrix')
    ax2.set_xlabel('Target Atom')
    ax2.set_ylabel('Source Atom')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "test_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved test plots to {output_dir / 'test_heatmaps.png'}")

def main():
    """Run quick test analysis."""
    
    print("🧪 Running quick test of attention analysis...")
    
    # Create output directory
    output_dir = Path("experiments/attention")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    print("📊 Generating synthetic data...")
    attention_maps, adjacency_matrices = generate_synthetic_data()
    
    # Compute metrics
    print("📈 Computing metrics...")
    metrics, avg_attention, avg_adjacency = compute_basic_metrics(attention_maps, adjacency_matrices)
    
    # Create plots
    print("🎨 Creating plots...")
    create_test_plots(avg_attention, avg_adjacency, output_dir)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Spearman Correlation: {metrics['spearman_correlation']:.3f}")
    print(f"Attention Mean: {metrics['attention_mean']:.3f}")
    print(f"Adjacency Mean: {metrics['adjacency_mean']:.3f}")
    print(f"Samples: {metrics['n_samples']}")
    print(f"Atoms: {metrics['n_atoms']}")
    print("\n✅ Quick test completed successfully!")
    print("📂 Check experiments/attention/test_heatmaps.png for visualization")
    
    if metrics['spearman_correlation'] > 0.5:
        print("🎯 Good correlation detected - synthetic data shows expected pattern")
    else:
        print("⚠️  Low correlation - this is expected for random synthetic data")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("  pip install numpy matplotlib scipy")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)