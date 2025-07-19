#!/usr/bin/env python3
"""
Architecture Equivalence Testing Script

This script systematically tests all three PT swap flow architectures
(simple, graph, transformer) to compare their expressivity and performance.

Usage:
    python test_architecture_equivalence.py
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import time


def run_training(config_path: str, temp_pair: tuple = (0, 1)) -> Dict:
    """Run training for a specific configuration and return results."""
    print(f"\n{'='*50}")
    print(f"Training with config: {config_path}")
    print(f"Temperature pair: {temp_pair}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # Activate conda environment and run training
    cmd = [
        "conda", "run", "-n", "accelmd", "python", "main.py",
        "--config", config_path,
        "--temp-pair", str(temp_pair[0]), str(temp_pair[1])
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        if result.returncode != 0:
            print(f"Error running {config_path}:")
            print(result.stderr)
            return {"error": result.stderr}
        
        # Parse output for key metrics
        output_lines = result.stdout.split('\n')
        
        # Extract final training metrics
        final_loss = None
        naive_acceptance = None
        flow_acceptance = None
        training_time = time.time() - start_time
        
        for line in output_lines:
            if "best_metric" in line:
                try:
                    # Try to parse JSON output from training summary
                    summary = json.loads(line)
                    final_loss = summary.get("best_metric")
                except:
                    pass
            elif "naïve swap" in line:
                try:
                    naive_acceptance = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "flow swap" in line:
                try:
                    flow_acceptance = float(line.split(":")[-1].strip())
                except:
                    pass
        
        return {
            "config": config_path,
            "final_loss": final_loss,
            "naive_acceptance": naive_acceptance,
            "flow_acceptance": flow_acceptance,
            "training_time_hours": training_time / 3600,
            "improvement": (flow_acceptance - naive_acceptance) / naive_acceptance * 100 if naive_acceptance and flow_acceptance else None,
            "success": True
        }
        
    except subprocess.TimeoutExpired:
        return {"error": "Training timed out", "config": config_path}
    except Exception as e:
        return {"error": str(e), "config": config_path}


def run_ablation_study() -> List[Dict]:
    """Run systematic ablation study across architectures."""
    
    # Configuration files to test
    configs = [
        "configs/AA_simple.yaml",
        "configs/AA.yaml",  # Enhanced graph
        "configs/AA_transformer.yaml",
    ]
    
    results = []
    
    # Test each configuration
    for config in configs:
        if not Path(config).exists():
            print(f"Warning: Config {config} not found, skipping...")
            continue
            
        result = run_training(config)
        result["architecture"] = config.split("_")[-1].replace(".yaml", "")
        if config == "configs/AA.yaml":
            result["architecture"] = "graph_enhanced"
        results.append(result)
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv("architecture_comparison_results.csv", index=False)
        print(f"Intermediate results saved to architecture_comparison_results.csv")
    
    return results


def analyze_results(results: List[Dict]) -> None:
    """Analyze and visualize the results."""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Filter out failed runs
    successful_runs = df[df['success'] == True].copy()
    
    if len(successful_runs) == 0:
        print("No successful runs to analyze!")
        return
    
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("="*60)
    
    # Summary table
    summary_cols = ['architecture', 'final_loss', 'naive_acceptance', 'flow_acceptance', 'improvement', 'training_time_hours']
    print("\nPerformance Summary:")
    print(successful_runs[summary_cols].to_string(index=False, float_format='%.4f'))
    
    # Rankings
    print("\n" + "-"*40)
    print("RANKINGS")
    print("-"*40)
    
    if 'improvement' in successful_runs.columns:
        improvement_ranking = successful_runs.sort_values('improvement', ascending=False)
        print("\nBy Improvement over Naive Swap:")
        for i, (_, row) in enumerate(improvement_ranking.iterrows(), 1):
            print(f"{i}. {row['architecture']}: {row['improvement']:.2f}% improvement")
    
    if 'flow_acceptance' in successful_runs.columns:
        acceptance_ranking = successful_runs.sort_values('flow_acceptance', ascending=False)
        print("\nBy Absolute Flow Acceptance:")
        for i, (_, row) in enumerate(acceptance_ranking.iterrows(), 1):
            print(f"{i}. {row['architecture']}: {row['flow_acceptance']:.4f}")
    
    if 'training_time_hours' in successful_runs.columns:
        time_ranking = successful_runs.sort_values('training_time_hours')
        print("\nBy Training Time:")
        for i, (_, row) in enumerate(time_ranking.iterrows(), 1):
            print(f"{i}. {row['architecture']}: {row['training_time_hours']:.2f} hours")
    
    # Statistical analysis
    print("\n" + "-"*40)
    print("STATISTICAL INSIGHTS")
    print("-"*40)
    
    if len(successful_runs) >= 2:
        best_arch = successful_runs.loc[successful_runs['improvement'].idxmax()]
        worst_arch = successful_runs.loc[successful_runs['improvement'].idxmin()]
        
        print(f"\nBest performing architecture: {best_arch['architecture']}")
        print(f"  - Improvement: {best_arch['improvement']:.2f}%")
        print(f"  - Flow acceptance: {best_arch['flow_acceptance']:.4f}")
        
        print(f"\nWorst performing architecture: {worst_arch['architecture']}")
        print(f"  - Improvement: {worst_arch['improvement']:.2f}%")
        print(f"  - Flow acceptance: {worst_arch['flow_acceptance']:.4f}")
        
        improvement_range = best_arch['improvement'] - worst_arch['improvement']
        print(f"\nPerformance gap: {improvement_range:.2f} percentage points")
    
    # Efficiency analysis
    if 'training_time_hours' in successful_runs.columns and 'improvement' in successful_runs.columns:
        successful_runs['efficiency'] = successful_runs['improvement'] / successful_runs['training_time_hours']
        efficiency_ranking = successful_runs.sort_values('efficiency', ascending=False)
        
        print("\nEfficiency Ranking (Improvement / Training Time):")
        for i, (_, row) in enumerate(efficiency_ranking.iterrows(), 1):
            print(f"{i}. {row['architecture']}: {row['efficiency']:.2f} %/hour")


def generate_recommendations(results: List[Dict]) -> None:
    """Generate recommendations based on results."""
    
    successful_runs = [r for r in results if r.get('success', False)]
    
    if not successful_runs:
        print("\nNo successful runs - cannot generate recommendations")
        return
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Find best performer
    best_result = max(successful_runs, key=lambda x: x.get('improvement', -100))
    
    print(f"\n🏆 BEST ARCHITECTURE: {best_result['architecture'].upper()}")
    print(f"   Improvement: {best_result.get('improvement', 0):.2f}%")
    print(f"   Flow acceptance: {best_result.get('flow_acceptance', 0):.4f}")
    
    # Analyze why it might be best
    if best_result['architecture'] == 'graph_enhanced':
        print("\n📈 WHY GRAPH ENHANCED WORKS:")
        print("   • Temperature conditioning helps model understand T-dependence")
        print("   • Chemical coupling masks respect molecular structure")
        print("   • Richer edge features (angles, dihedrals) capture geometry")
        print("   • Deeper networks provide more expressivity")
        print("   • Larger scale range (±50%) allows meaningful transformations")
    
    elif best_result['architecture'] == 'transformer':
        print("\n📈 WHY TRANSFORMER WORKS:")
        print("   • Attention mechanism captures long-range molecular interactions")
        print("   • RFF position encoding provides spatial awareness")
        print("   • No structural assumptions - learns patterns from data")
        print("   • Alternating position/velocity coupling is expressive")
    
    elif best_result['architecture'] == 'simple':
        print("\n📈 WHY SIMPLE WORKS:")
        print("   • Sometimes simpler is better - fewer parameters to tune")
        print("   • Direct coordinate transformation may be sufficient")
        print("   • Less prone to overfitting with limited data")
    
    # Practical recommendations
    print("\n🔧 PRACTICAL RECOMMENDATIONS:")
    
    best_improvement = best_result.get('improvement', 0)
    if best_improvement < 5:
        print("   ⚠️  All architectures show low improvement (<5%)")
        print("   → Try larger temperature gaps (e.g., pair [0,2] instead of [0,1])")
        print("   → Increase training data (lower subsample_rate)")
        print("   → Add acceptance loss component")
        print("   → Check if energy landscapes are too similar")
    elif best_improvement < 10:
        print("   ✅ Moderate improvement - continue optimizing best architecture")
        print(f"   → Focus on {best_result['architecture']} hyperparameter tuning")
        print("   → Try different temperature pairs")
    else:
        print("   🎉 Good improvement - architecture is working!")
        print(f"   → Scale up {best_result['architecture']} for production")
        print("   → Test on other peptides for generalization")
    
    print("\n🧪 FURTHER EXPERIMENTS:")
    print("   1. Test on different temperature pairs: [0,2], [1,3], [2,4]")
    print("   2. Vary training data amount: subsample_rate = [1, 5, 20, 100]") 
    print("   3. Add acceptance loss with different weight schedules")
    print("   4. Test generalization: train on AA, evaluate on AK/AS")
    print("   5. Ablation study on graph enhancements individually")


def main():
    """Main execution function."""
    print("🧬 PT Swap Flow Architecture Equivalence Testing")
    print("=" * 60)
    print("\nThis script will systematically test three architectures:")
    print("1. Simple (baseline coordinate-to-coordinate flow)")
    print("2. Graph Enhanced (with temperature conditioning & chemical masks)")
    print("3. Transformer (attention-based with RFF position encoding)")
    print("\nExpected runtime: 4-8 hours total")
    
    # Confirm before starting
    response = input("\nProceed with testing? (y/N): ").strip().lower()
    if response != 'y':
        print("Testing cancelled.")
        return
    
    # Run ablation study
    results = run_ablation_study()
    
    # Save final results
    df = pd.DataFrame(results)
    df.to_csv("final_architecture_comparison.csv", index=False)
    df.to_json("final_architecture_comparison.json", indent=2)
    
    # Analyze results
    analyze_results(results)
    
    # Generate recommendations
    generate_recommendations(results)
    
    print(f"\n✅ Testing complete! Results saved to:")
    print(f"   - final_architecture_comparison.csv")
    print(f"   - final_architecture_comparison.json")


if __name__ == "__main__":
    main() 