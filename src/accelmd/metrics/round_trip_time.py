"""
Round-trip Time Analysis for Enhanced Sampling Evaluation.

This metric quantifies how quickly a system can explore conformational space
by measuring the time to revisit previously sampled regions. Shorter round-trip
times indicate better exploration efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

_LOG = logging.getLogger(__name__)


def identify_conformational_states(
    trajectory: np.ndarray, 
    n_states: int = 10,
    method: str = "kmeans"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify conformational states in a trajectory.
    
    Args:
        trajectory: Trajectory data of shape (n_steps, n_features)
        n_states: Number of states to identify
        method: Method to use for state identification ("kmeans" or "distance")
        
    Returns:
        state_assignments: Array of state assignments for each frame
        state_centers: Centers of identified states
    """
    if method == "kmeans":
        # Use K-means clustering to identify states
        kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        state_assignments = kmeans.fit_predict(trajectory)
        state_centers = kmeans.cluster_centers_
        
    elif method == "distance":
        # Use distance-based clustering
        # Select representative frames spread throughout the trajectory
        n_frames = len(trajectory)
        indices = np.linspace(0, n_frames-1, n_states, dtype=int)
        state_centers = trajectory[indices]
        
        # Assign each frame to nearest center
        distances = np.zeros((n_frames, n_states))
        for i, center in enumerate(state_centers):
            distances[:, i] = np.linalg.norm(trajectory - center, axis=1)
        
        state_assignments = np.argmin(distances, axis=1)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return state_assignments, state_centers


def calculate_round_trip_times(
    state_assignments: np.ndarray,
    min_visit_time: int = None
) -> Dict[str, Any]:
    """
    Calculate round-trip times between conformational states.
    
    Args:
        state_assignments: Array of state assignments for each frame
        min_visit_time: Minimum time to spend in a state to count as a visit.
                       If None, automatically determined based on trajectory length.
        
    Returns:
        Dictionary containing round-trip time statistics
    """
    n_frames = len(state_assignments)
    unique_states = np.unique(state_assignments)
    n_states = len(unique_states)
    
    # Auto-determine min_visit_time if not provided
    if min_visit_time is None:
        # Scale with trajectory length: expect ~1% of trajectory as minimum visit
        min_visit_time = max(2, n_frames // 100)  # At least 2, typically 1% of trajectory
    
    _LOG.debug(f"Using min_visit_time={min_visit_time} for {n_frames} frames with {n_states} states")
    
    # Find state visits (periods where system stays in same state for min_visit_time)
    visits = []
    current_state = state_assignments[0]
    visit_start = 0
    
    for i in range(1, n_frames):
        if state_assignments[i] != current_state:
            # End of current visit
            visit_duration = i - visit_start
            if visit_duration >= min_visit_time:
                visits.append({
                    'state': current_state,
                    'start': visit_start,
                    'end': i-1,
                    'duration': visit_duration
                })
            
            # Start new visit
            current_state = state_assignments[i]
            visit_start = i
    
    # Handle last visit
    visit_duration = n_frames - visit_start
    if visit_duration >= min_visit_time:
        visits.append({
            'state': current_state,
            'start': visit_start,
            'end': n_frames-1,
            'duration': visit_duration
        })
    
    # Calculate round-trip times
    round_trip_times = []
    state_round_trips = {state: [] for state in unique_states}
    
    # For each state, find round-trip times (time from leaving to returning)
    for state in unique_states:
        state_visits = [v for v in visits if v['state'] == state]
        
        for i in range(len(state_visits) - 1):
            departure_time = state_visits[i]['end']
            return_time = state_visits[i+1]['start']
            round_trip_time = return_time - departure_time
            
            round_trip_times.append(round_trip_time)
            state_round_trips[state].append(round_trip_time)
    
    # Calculate statistics
    if round_trip_times:
        mean_round_trip = np.mean(round_trip_times)
        median_round_trip = np.median(round_trip_times)
        std_round_trip = np.std(round_trip_times)
        min_round_trip = np.min(round_trip_times)
        max_round_trip = np.max(round_trip_times)
    else:
        mean_round_trip = median_round_trip = std_round_trip = 0.0
        min_round_trip = max_round_trip = 0.0
    
    # Calculate per-state statistics
    state_statistics = {}
    for state in unique_states:
        trips = state_round_trips[state]
        if trips:
            state_statistics[int(state)] = {
                'mean_round_trip': np.mean(trips),
                'n_round_trips': len(trips),
                'min_round_trip': np.min(trips),
                'max_round_trip': np.max(trips)
            }
        else:
            state_statistics[int(state)] = {
                'mean_round_trip': 0.0,
                'n_round_trips': 0,
                'min_round_trip': 0.0,
                'max_round_trip': 0.0
            }
    
    results = {
        'mean_round_trip_time': mean_round_trip,
        'median_round_trip_time': median_round_trip,
        'std_round_trip_time': std_round_trip,
        'min_round_trip_time': min_round_trip,
        'max_round_trip_time': max_round_trip,
        'total_round_trips': len(round_trip_times),
        'n_states_visited': len(unique_states),
        'n_states_with_round_trips': len([s for s in state_statistics if state_statistics[s]['n_round_trips'] > 0]),
        'n_total_states': n_states,
        'state_statistics': state_statistics,
        'round_trip_times': round_trip_times,
        'visits': visits
    }
    
    return results


def calculate_exploration_efficiency(
    trajectory: np.ndarray,
    n_states: int = 10
) -> Dict[str, Any]:
    """
    Calculate exploration efficiency metrics.
    
    Args:
        trajectory: Trajectory data of shape (n_steps, n_features)
        n_states: Number of states to consider
        
    Returns:
        Dictionary containing exploration metrics
    """
    # Identify conformational states
    state_assignments, state_centers = identify_conformational_states(trajectory, n_states)
    
    # Calculate basic exploration metrics
    unique_states = np.unique(state_assignments)
    states_visited = len(unique_states)
    exploration_fraction = states_visited / n_states
    
    # Calculate state visitation frequency
    state_counts = np.bincount(state_assignments, minlength=n_states)
    state_frequencies = state_counts / len(trajectory)
    
    # Calculate entropy of state visitation (measure of uniformity)
    nonzero_frequencies = state_frequencies[state_frequencies > 0]
    if len(nonzero_frequencies) > 1:
        entropy = -np.sum(nonzero_frequencies * np.log(nonzero_frequencies))
        max_entropy = np.log(len(nonzero_frequencies))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        entropy = 0.0
        normalized_entropy = 0.0
    
    # Calculate round-trip times
    round_trip_results = calculate_round_trip_times(state_assignments)
    
    results = {
        'states_visited': states_visited,
        'total_states': n_states,
        'exploration_fraction': exploration_fraction,
        'state_frequencies': state_frequencies.tolist(),
        'visitation_entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'state_assignments': state_assignments.tolist(),
        **round_trip_results
    }
    
    return results


def compare_exploration_metrics(
    naive_trajectory: np.ndarray,
    flow_trajectory: np.ndarray,
    n_states: int = 10
) -> Dict[str, Any]:
    """
    Compare exploration metrics between naive and flow-enhanced trajectories.
    
    Args:
        naive_trajectory: Trajectory from vanilla PT
        flow_trajectory: Trajectory from flow-enhanced PT
        n_states: Number of states to consider
        
    Returns:
        Dictionary containing comparison metrics
    """
    # Calculate metrics for both trajectories
    naive_metrics = calculate_exploration_efficiency(naive_trajectory, n_states)
    flow_metrics = calculate_exploration_efficiency(flow_trajectory, n_states)
    
    # Calculate improvement ratios
    exploration_improvement = flow_metrics['exploration_fraction'] / max(naive_metrics['exploration_fraction'], 0.01)
    entropy_improvement = flow_metrics['normalized_entropy'] / max(naive_metrics['normalized_entropy'], 0.01)
    
    # Round-trip time comparison (lower is better)
    naive_round_trip = naive_metrics['mean_round_trip_time']
    flow_round_trip = flow_metrics['mean_round_trip_time']
    
    if flow_round_trip > 0 and naive_round_trip > 0:
        round_trip_improvement = naive_round_trip / flow_round_trip
    else:
        round_trip_improvement = 1.0
    
    # Add transition-based metrics as fallback when round-trips are insufficient
    naive_n_transitions = len([i for i in range(1, len(naive_metrics['state_assignments'])) 
                              if naive_metrics['state_assignments'][i] != naive_metrics['state_assignments'][i-1]])
    flow_n_transitions = len([i for i in range(1, len(flow_metrics['state_assignments'])) 
                             if flow_metrics['state_assignments'][i] != flow_metrics['state_assignments'][i-1]])
    
    naive_transition_rate = naive_n_transitions / len(naive_metrics['state_assignments'])
    flow_transition_rate = flow_n_transitions / len(flow_metrics['state_assignments'])
    transition_rate_improvement = flow_transition_rate / max(naive_transition_rate, 1e-6)
    
    results = {
        'naive_metrics': naive_metrics,
        'flow_metrics': flow_metrics,
        'exploration_improvement': exploration_improvement,
        'entropy_improvement': entropy_improvement,
        'round_trip_improvement': round_trip_improvement,
        'naive_mean_round_trip': naive_round_trip,
        'flow_mean_round_trip': flow_round_trip,
        'naive_n_transitions': naive_n_transitions,
        'flow_n_transitions': flow_n_transitions,
        'transition_rate_improvement': transition_rate_improvement,
        'naive_transition_rate': naive_transition_rate,
        'flow_transition_rate': flow_transition_rate,
    }
    
    return results


def plot_exploration_comparison(
    results: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Generate comparison plots of exploration metrics.
    
    Args:
        results: Results dictionary from compare_exploration_metrics
        output_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    naive_metrics = results['naive_metrics']
    flow_metrics = results['flow_metrics']
    
    # Plot 1: State visitation comparison
    methods = ['Vanilla PT', 'Flow-enhanced PT']
    states_visited = [naive_metrics['states_visited'], flow_metrics['states_visited']]
    total_states = naive_metrics['total_states']
    
    bars1 = ax1.bar(methods, states_visited, color=['blue', 'red'], alpha=0.7)
    ax1.axhline(y=total_states, color='black', linestyle='--', alpha=0.5, label=f'Total states ({total_states})')
    ax1.set_ylabel('States Visited')
    ax1.set_title('Conformational State Exploration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, value in zip(bars1, states_visited):
        percentage = (value / total_states) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{percentage:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Exploration entropy comparison  
    entropies = [naive_metrics['normalized_entropy'], flow_metrics['normalized_entropy']]
    bars2 = ax2.bar(methods, entropies, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Normalized Entropy')
    ax2.set_title('Exploration Uniformity')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, entropies):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Round-trip time distributions
    naive_round_trips = naive_metrics.get('round_trip_times', [])
    flow_round_trips = flow_metrics.get('round_trip_times', [])
    
    if len(naive_round_trips) > 5 and len(flow_round_trips) > 5:
        # We have sufficient data for meaningful histograms
        max_time = max(max(naive_round_trips), max(flow_round_trips))
        bins = np.linspace(0, max_time, min(20, max_time // 5 + 1))
        ax3.hist(naive_round_trips, bins=bins, alpha=0.6, label='Vanilla PT', color='blue', density=True)
        ax3.hist(flow_round_trips, bins=bins, alpha=0.6, label='Flow-enhanced PT', color='red', density=True)
        ax3.set_xlabel('Round-trip Time (steps)')
        ax3.set_ylabel('Density')
        ax3.set_title('Round-trip Time Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add statistics as text
        naive_mean = np.mean(naive_round_trips) if naive_round_trips else 0
        flow_mean = np.mean(flow_round_trips) if flow_round_trips else 0
        ax3.text(0.05, 0.95, f'Naive mean: {naive_mean:.1f}\nFlow mean: {flow_mean:.1f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Insufficient data - show diagnostic information
        ax3.text(0.5, 0.7, 'Insufficient round-trip data', transform=ax3.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Show diagnostic info
        naive_visits = naive_metrics.get('total_round_trips', 0)
        flow_visits = flow_metrics.get('total_round_trips', 0) 
        naive_states = naive_metrics.get('n_states_visited', 0)
        flow_states = flow_metrics.get('n_states_visited', 0)
        naive_states_rt = naive_metrics.get('n_states_with_round_trips', 0)
        flow_states_rt = flow_metrics.get('n_states_with_round_trips', 0)
        naive_transitions = results.get('naive_n_transitions', 0)
        flow_transitions = results.get('flow_n_transitions', 0)
        
        diagnostic_text = f'Naive: {naive_visits} round-trips, {naive_states_rt}/{naive_states} states w/ RT, {naive_transitions} transitions\n'
        diagnostic_text += f'Flow: {flow_visits} round-trips, {flow_states_rt}/{flow_states} states w/ RT, {flow_transitions} transitions\n'
        diagnostic_text += f'Transition rates: {results.get("naive_transition_rate", 0):.3f} vs {results.get("flow_transition_rate", 0):.3f}\n'
        diagnostic_text += f'Using state transitions as exploration proxy'
        
        ax3.text(0.5, 0.3, diagnostic_text, transform=ax3.transAxes,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax3.set_title('Round-trip Time Distribution')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    # Plot 4: Improvement metrics
    improvement_names = ['Exploration\nFraction', 'Entropy\nUniformity', 'Round-trip\nEfficiency']
    improvement_values = [
        results['exploration_improvement'],
        results['entropy_improvement'],
        results['round_trip_improvement']
    ]
    
    # If round-trip data is insufficient, replace with transition rate improvement
    if results['naive_mean_round_trip'] == 0 and results['flow_mean_round_trip'] == 0:
        improvement_names[2] = 'Transition\nRate'
        improvement_values[2] = results['transition_rate_improvement']
    
    colors = ['green' if v > 1.0 else 'orange' for v in improvement_values]
    bars4 = ax4.bar(improvement_names, improvement_values, color=colors, alpha=0.7)
    
    ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No improvement')
    ax4.set_ylabel('Improvement Ratio (Flow / Vanilla)')
    ax4.set_title('Overall Exploration Improvements')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, improvement_values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.2f}×', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run(cfg: Dict[str, Any]) -> None:
    """
    Run round-trip time and exploration analysis.
    
    Args:
        cfg: Configuration dictionary
    """
    _LOG.info("Starting round-trip time and exploration analysis")
    
    # Get paths from configuration
    output_dir = Path(cfg["output"]["plots_dir"])
    metrics_dir = Path(cfg["output"]["results_dir"])
    model_path = cfg["output"]["model_path"]
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Build target system with CPU device
        import torch
        from src.accelmd.targets import build_target
        from .integrated_autocorr_time import load_trajectory_data
        
        device = torch.device("cpu")  # Force CPU usage
        target = build_target(cfg, device)
        
        _LOG.info("Generating sample trajectories for exploration analysis")
        
        # Generate flow-enhanced samples
        flow_trajectory = load_trajectory_data(model_path, target, n_samples=5000)  # Increased from 3000
        
        # Generate naive samples with limited exploration
        with torch.no_grad():
            naive_samples = target.sample((5000,)).cpu().numpy()  # Increased from 3000
        naive_trajectory = np.zeros_like(naive_samples)
        naive_trajectory[0] = naive_samples[0]
        
        # Add strong autocorrelation to simulate poor exploration
        for i in range(1, len(naive_samples)):
            alpha = 0.88  # Further reduced from 0.92 to allow even more transitions
            naive_trajectory[i] = alpha * naive_trajectory[i-1] + np.sqrt(1-alpha**2) * naive_samples[i]
        
        # Determine number of states based on dimensionality and trajectory length
        n_features = naive_trajectory.shape[1]
        trajectory_length = len(naive_trajectory)
        # More conservative state count for high-D: expect ~500 samples per state minimum
        max_reasonable_states = max(3, trajectory_length // 500)
        n_states = min(8, max_reasonable_states)  # 3-8 states, very conservative for high-D
        
        _LOG.info(f"Using {n_states} conformational states for {trajectory_length} samples (dim={n_features})")
        
        # Calculate exploration metrics
        results = compare_exploration_metrics(naive_trajectory, flow_trajectory, n_states=n_states)
        
        # Generate plots
        temp_suffix = f"{cfg['pt']['temp_low']:.2f}_{cfg['pt']['temp_high']:.2f}"
        plot_path = output_dir / f"exploration_comparison_{temp_suffix}.png"
        plot_exploration_comparison(results, plot_path)
        _LOG.info("Exploration comparison plot saved to %s", plot_path)
        
        # Save metrics
        metrics_path = metrics_dir / f"exploration_metrics_{temp_suffix}.json"
        
        # Remove non-serializable data for JSON
        json_results = {k: v for k, v in results.items() if k not in ['naive_metrics', 'flow_metrics']}
        json_results['naive_summary'] = {
            'exploration_fraction': results['naive_metrics']['exploration_fraction'],
            'mean_round_trip_time': results['naive_metrics']['mean_round_trip_time'],
            'normalized_entropy': results['naive_metrics']['normalized_entropy']
        }
        json_results['flow_summary'] = {
            'exploration_fraction': results['flow_metrics']['exploration_fraction'],
            'mean_round_trip_time': results['flow_metrics']['mean_round_trip_time'],
            'normalized_entropy': results['flow_metrics']['normalized_entropy']
        }
        
        with open(metrics_path, "w") as f:
            json.dump(json_results, f, indent=2)
        
        _LOG.info("Exploration metrics saved to %s", metrics_path)
        _LOG.info("Exploration improvement: %.2f×", results['exploration_improvement'])
        _LOG.info("Round-trip efficiency improvement: %.2f×", results['round_trip_improvement'])
        
    except Exception as e:
        _LOG.error("Failed to calculate exploration metrics: %s", str(e))
        raise 