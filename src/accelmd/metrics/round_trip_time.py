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

# +++ Added: Helper for ALDP predefined states +++
def get_predefined_aldp_state_centers(
    n_states_requested: int, 
    dim: int, 
    custom_centers_file_path: str = None
) -> np.ndarray:
    """
    Returns predefined state centers for ALDP.
    Attempts to load from `custom_centers_file_path` if provided.
    Otherwise, uses placeholder data with a warning.

    Args:
        n_states_requested: The number of states the calling function expects.
        dim: The dimensionality of the trajectory data (e.g., 66 for ALDP Cartesian).
        custom_centers_file_path: Path to a .npy file containing custom state centers.
                                   The .npy file should store an array of shape (N, dim),
                                   where N is the number of custom states.

    Returns:
        np.ndarray: Array of state centers, or an empty array if loading fails and
                    no placeholder is applicable.
    """
    if custom_centers_file_path:
        try:
            centers_path = Path(custom_centers_file_path)
            if centers_path.is_file():
                _LOG.info(f"Attempting to load custom ALDP state centers from: {centers_path}")
                centers = np.load(centers_path)
                if centers.ndim != 2 or centers.shape[1] != dim:
                    _LOG.error(
                        f"Loaded custom ALDP centers from {centers_path} have incorrect shape. "
                        f"Expected (N, {dim}), got {centers.shape}. Will use placeholders."
                    )
                    # Fall through to placeholder logic
                elif centers.shape[0] < n_states_requested:
                    _LOG.warning(
                        f"Loaded {centers.shape[0]} custom ALDP states from {centers_path}, "
                        f"which is less than the requested {n_states_requested} states. Using all loaded states."
                    )
                    return centers 
                elif centers.shape[0] > n_states_requested:
                    _LOG.info(
                        f"Loaded {centers.shape[0]} custom ALDP states from {centers_path}. "
                        f"Using the first {n_states_requested} as requested."
                    )
                    return centers[:n_states_requested, :]
                else: # centers.shape[0] == n_states_requested
                    _LOG.info(f"Successfully loaded {centers.shape[0]} custom ALDP state centers from {centers_path}.")
                    return centers
            else:
                _LOG.warning(
                    f"Custom ALDP state centers file not found: {centers_path}. "
                    "Will use placeholder random centers. Please create this file if you intend to use custom states."
                )
                # Fall through to placeholder logic
        except Exception as e:
            _LOG.error(
                f"Error loading custom ALDP state centers from {custom_centers_file_path}: {e}. "
                "Will use placeholder random centers."
            )
            # Fall through to placeholder logic
    else:
        _LOG.warning(
            "No custom_centers_file_path provided for ALDP. Using PLACEHOLDER random state centers. "
            "This is NOT recommended for production analysis. Please define and provide a path to your ALDP state centers .npy file."
        )

    # Fallback to placeholder random centers if file not loaded or not provided
    _LOG.warning(
        f"Using PLACEHOLDER random state centers for ALDP with {n_states_requested} states and dim={dim}. "
        "These MUST be replaced with actual, meaningful ALDP state definitions in Cartesian coordinates, "
        "or provide a valid 'custom_centers_file_path' in the configuration."
    )
    if n_states_requested <= 0 or dim <= 0:
        return np.array([])
    
    # Example: Centering around origin with some spread.
    placeholder_centers = np.random.randn(n_states_requested, dim) * 0.1 # Small spread around origin
    
    _LOG.info(
        "To define meaningful ALDP states (e.g., C7eq, C7ax, alphaR, alphaL):\n"
        "1. Identify target dihedral angle regions (phi, psi).\n"
        "2. Obtain representative 66D Cartesian coordinate structures for these states.\n"
        "   (e.g., from simulations, or by modifying a base structure and minimizing).\n"
        "3. Save these N structures as a NumPy array of shape (N, 66) to a '.npy' file.\n"
        "4. Configure 'evaluation.exploration_aldp_custom_centers_file' in your YAML config to point to this .npy file."
    )
    
    # The original placeholder logic for n_states_requested vs actual available might not apply here
    # if we are generating them on the fly. We just generate what was requested.
    return placeholder_centers
# +++ End Added +++

def identify_conformational_states(
    trajectory: np.ndarray, 
    n_states: int = 10,
    method: str = "kmeans",
    # +++ Added: Parameters for custom ALDP states +++
    custom_centers_source: str = None, 
    target_type: str = None # e.g., "aldp" or "gmm"
    # +++ End Added +++
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify conformational states in a trajectory.
    
    Args:
        trajectory: Trajectory data of shape (n_steps, n_features)
        n_states: Number of states to identify
        method: Method to use for state identification ("kmeans", "distance", "custom_aldp")
        custom_centers_source: Identifier for custom centers if method='custom_aldp'
        target_type: Type of the target system (e.g., 'aldp') to guide custom logic
        
    Returns:
        state_assignments: Array of state assignments for each frame
        state_centers: Centers of identified states
    """
    n_frames, n_features = trajectory.shape

    if method == "kmeans":
        _LOG.debug(f"Identifying {n_states} states using K-Means.")
        # Use K-means clustering to identify states
        kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        state_assignments = kmeans.fit_predict(trajectory)
        state_centers = kmeans.cluster_centers_
        
    elif method == "distance":
        _LOG.debug(f"Identifying {n_states} states using distance-based representative frames.")
        # Use distance-based clustering
        # Select representative frames spread throughout the trajectory
        indices = np.linspace(0, n_frames-1, n_states, dtype=int)
        state_centers = trajectory[indices]
        
        # Assign each frame to nearest center
        distances = np.zeros((n_frames, n_states))
        for i, center in enumerate(state_centers):
            distances[:, i] = np.linalg.norm(trajectory - center, axis=1)
        
        state_assignments = np.argmin(distances, axis=1)
    # +++ Added: Custom ALDP state definition +++
    elif method == "custom_aldp":
        if target_type and "aldp" in target_type.lower():
            _LOG.info(f"Identifying {n_states} states using predefined ALDP centers ('{custom_centers_source}').")
            # Pass the file path from cfg (via custom_centers_id which now holds the path)
            # The custom_centers_source variable will now hold the path to the .npy file
            # if configured, or None otherwise.
            aldp_centers_filepath = custom_centers_source # custom_centers_source now used as filepath or main ID
            
            state_centers = get_predefined_aldp_state_centers(n_states, n_features, aldp_centers_filepath)
            
            if state_centers.size == 0 or state_centers.shape[0] == 0:
                _LOG.error("Failed to get predefined ALDP state centers or no states defined. Falling back to K-Means.")
                # Fallback to K-Means if custom states are not properly defined/loaded
                return identify_conformational_states(trajectory, n_states, method="kmeans")

            if state_centers.shape[1] != n_features:
                raise ValueError(
                    f"Dimension mismatch for predefined ALDP state centers. Trajectory has {n_features} features, "
                    f"but centers have {state_centers.shape[1]} features."
                )
            
            # Ensure we use the actual number of states provided by the custom function
            actual_n_states = state_centers.shape[0]
            if actual_n_states != n_states:
                _LOG.warning(f"Requested {n_states} but got {actual_n_states} from custom ALDP definition. Using {actual_n_states}.")
                n_states = actual_n_states

            distances = np.zeros((n_frames, n_states))
            for i, center in enumerate(state_centers):
                distances[:, i] = np.linalg.norm(trajectory - center, axis=1)
            state_assignments = np.argmin(distances, axis=1)
        else:
            _LOG.warning(f"Method 'custom_aldp' selected, but target_type is not ALDP (got '{target_type}'). Defaulting to K-Means.")
            return identify_conformational_states(trajectory, n_states, method="kmeans")
    # +++ End Added +++
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
    unique_states, counts = np.unique(state_assignments, return_counts=True)
    n_defined_states = len(unique_states) # Number of states actually present in assignments
    
    # Auto-determine min_visit_time if not provided
    if min_visit_time is None:
        # Scale with trajectory length: expect ~1% of trajectory as minimum visit
        min_visit_time = max(2, n_frames // 100)  # At least 2, typically 1% of trajectory
    
    _LOG.debug(f"Using min_visit_time={min_visit_time} for {n_frames} frames with {n_defined_states} unique states found in assignments.")
    
    # Find state visits (periods where system stays in same state for min_visit_time)
    visits = []
    if n_frames == 0: # Handle empty trajectory
        current_state = -1 # No valid state
        visit_start = 0
    else:
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
    
    # Handle last visit if trajectory was not empty
    if n_frames > 0:
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
    state_round_trips = {state: [] for state in unique_states} # Only for states that appear
    
    # For each state, find round-trip times (time from leaving to returning)
    for state_val in unique_states: # Iterate over actual state values present
        state_visits = [v for v in visits if v['state'] == state_val]
        
        for i in range(len(state_visits) - 1):
            departure_time = state_visits[i]['end']
            return_time = state_visits[i+1]['start']
            round_trip_time = return_time - departure_time
            
            round_trip_times.append(round_trip_time)
            state_round_trips[state_val].append(round_trip_time) # Use state_val as key
    
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
    for state_val in unique_states: # Iterate over actual state values present
        trips = state_round_trips[state_val]
        if trips:
            state_statistics[int(state_val)] = { # Use int(state_val)
                'mean_round_trip': np.mean(trips),
                'n_round_trips': len(trips),
                'min_round_trip': np.min(trips),
                'max_round_trip': np.max(trips)
            }
        else:
            state_statistics[int(state_val)] = { # Use int(state_val)
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
        'n_states_visited_in_assignments': n_defined_states, # How many states were actually assigned
        'n_states_with_round_trips': len([s for s_val, s in state_statistics.items() if s['n_round_trips'] > 0]),
        # 'n_total_states_expected': n_states, # This was the input n_states, might differ from n_defined_states
        'state_statistics': state_statistics,
        'round_trip_times': round_trip_times,
        'visits': visits
    }
    
    return results


def calculate_exploration_efficiency(
    trajectory: np.ndarray,
    n_states: int = 10,
    # +++ Added: Parameters for state definition +++
    state_definition_method: str = "kmeans", 
    custom_centers_id: str = None,
    target_type: str = None
    # +++ End Added +++
) -> Dict[str, Any]:
    """
    Calculate exploration efficiency metrics.
    
    Args:
        trajectory: Trajectory data of shape (n_steps, n_features)
        n_states: Number of states to consider/request
        state_definition_method: Method for identify_conformational_states
        custom_centers_id: Identifier for custom centers
        target_type: Type of target system
        
    Returns:
        Dictionary containing exploration metrics
    """
    # Identify conformational states
    state_assignments, state_centers = identify_conformational_states(
        trajectory, 
        n_states, 
        method=state_definition_method,
        custom_centers_source=custom_centers_id,
        target_type=target_type
    )
    
    # n_states might have been adjusted by identify_conformational_states if custom_aldp provided fewer centers
    actual_n_states_defined_by_centers = state_centers.shape[0] if state_centers.ndim == 2 and state_centers.shape[0] > 0 else 0
    
    # Calculate basic exploration metrics
    unique_states_in_traj = np.unique(state_assignments)
    n_states_visited_in_traj = len(unique_states_in_traj)
    
    # Exploration fraction should be relative to the number of states we *attempted* to define or that *were* defined
    # If using K-Means, actual_n_states_defined_by_centers will be n_states (input)
    # If custom, it's the number of centers returned by get_predefined_aldp_state_centers
    denominator_for_fraction = actual_n_states_defined_by_centers if actual_n_states_defined_by_centers > 0 else n_states

    exploration_fraction = n_states_visited_in_traj / denominator_for_fraction if denominator_for_fraction > 0 else 0.0
    
    # Calculate state visitation frequency (bincount needs max_label + 1, or minlength)
    # If state_assignments can be empty or have max value less than actual_n_states_defined_by_centers
    min_len_bincount = actual_n_states_defined_by_centers if actual_n_states_defined_by_centers > 0 else (np.max(state_assignments) + 1 if state_assignments.size > 0 else 1)

    state_counts = np.bincount(state_assignments, minlength=min_len_bincount) if state_assignments.size > 0 else np.array([0]*min_len_bincount)
    state_frequencies = state_counts / len(trajectory) if len(trajectory) > 0 else np.zeros_like(state_counts, dtype=float)
    
    # Calculate entropy of state visitation (measure of uniformity)
    nonzero_frequencies = state_frequencies[state_frequencies > 0]
    if len(nonzero_frequencies) > 1: # Entropy requires at least two distinct outcomes
        entropy = -np.sum(nonzero_frequencies * np.log(nonzero_frequencies))
        # Max entropy for the number of *visited* states, not *defined* states.
        # Or, should it be max_entropy for number of *defined* states if we want to penalize not visiting some?
        # Let's use number of *visited* states for how uniform the visitation of *those* states is.
        max_entropy_for_visited = np.log(len(nonzero_frequencies))
        normalized_entropy = entropy / max_entropy_for_visited if max_entropy_for_visited > 0 else 0.0
    else: # If only one state visited (or no states), entropy is 0
        entropy = 0.0
        normalized_entropy = 0.0 # No variation, so 0 normalized entropy
    
    # Calculate round-trip times
    round_trip_results = calculate_round_trip_times(state_assignments)
    
    results = {
        'states_visited_in_traj': n_states_visited_in_traj, # Number of unique states found in trajectory assignments
        'n_actual_states_defined_by_centers': actual_n_states_defined_by_centers, # Number of centers used for assignment
        'exploration_fraction': exploration_fraction, # Visited / DefinedByCenters
        'state_frequencies': state_frequencies.tolist(),
        'visitation_entropy': entropy,
        'normalized_entropy': normalized_entropy, # Entropy relative to max for states *actually visited*
        'state_assignments_preview': state_assignments[:min(100, len(state_assignments))].tolist(), # Preview, not full list
        **round_trip_results 
    }
    # Add the original n_states requested as input for clarity
    results['n_states_requested_for_identification'] = n_states
    
    return results


def compare_exploration_metrics(
    naive_trajectory: np.ndarray,
    flow_trajectory: np.ndarray,
    n_states: int = 10,
    # +++ Added: Parameters for state definition +++
    state_definition_method: str = "kmeans", 
    custom_centers_id: str = None,
    target_type: str = None
    # +++ End Added +++
) -> Dict[str, Any]:
    """
    Compare exploration metrics between naive and flow-enhanced trajectories.
    
    Args:
        naive_trajectory: Trajectory from vanilla PT
        flow_trajectory: Trajectory from flow-enhanced PT
        n_states: Number of states to consider/request
        state_definition_method: Method for identify_conformational_states
        custom_centers_id: Identifier for custom centers
        target_type: Type of target system

    Returns:
        Dictionary containing comparison metrics
    """
    # Calculate metrics for both trajectories using the same state definition parameters
    # The n_states here is the *requested* number of states. 
    # The actual number of defined states might be different if using custom_aldp and fewer centers are provided.
    # calculate_exploration_efficiency will return 'n_actual_states_defined_by_centers'
    
    _LOG.info(f"Comparing exploration: Naive vs Flow with requested n_states={n_states}, method='{state_definition_method}'")
    
    naive_metrics = calculate_exploration_efficiency(
        naive_trajectory, n_states, state_definition_method, custom_centers_id, target_type
    )
    flow_metrics = calculate_exploration_efficiency(
        flow_trajectory, n_states, state_definition_method, custom_centers_id, target_type
    )
    
    # For a fair comparison of exploration_fraction and entropy, ensure they are normalized against the same denominator.
    # This should be handled by calculate_exploration_efficiency returning 'n_actual_states_defined_by_centers'
    # and using that for its own exploration_fraction.
    
    # Calculate improvement ratios
    # Ensure naive_metrics denominator is not zero to avoid division by zero.
    # exploration_fraction is already scaled by the number of defined states in calculate_exploration_efficiency
    exploration_improvement = flow_metrics['exploration_fraction'] / max(naive_metrics['exploration_fraction'], 1e-6) # Use a small epsilon
    
    # normalized_entropy is already normalized to [0,1] based on visited states.
    # Improvement means flow's normalized entropy is higher.
    entropy_improvement = flow_metrics['normalized_entropy'] / max(naive_metrics['normalized_entropy'], 1e-6) # Use a small epsilon
    
    # Round-trip time comparison (lower is better for mean_round_trip_time)
    naive_mean_round_trip = naive_metrics['mean_round_trip_time']
    flow_mean_round_trip = flow_metrics['mean_round_trip_time']
    
    round_trip_improvement = 1.0 # Default to no improvement
    if flow_mean_round_trip > 0 and naive_mean_round_trip > 0: # Both must have valid RTs
        round_trip_improvement = naive_mean_round_trip / flow_mean_round_trip
    elif naive_mean_round_trip == 0 and flow_mean_round_trip == 0: # Neither had round trips
        round_trip_improvement = 1.0 # No change
    elif naive_mean_round_trip > 0 and flow_mean_round_trip == 0: # Flow eliminated RT (inf improvement, cap it)
        round_trip_improvement = 100.0 # Cap improvement if flow has 0 RT and naive has >0
    # If naive is 0 and flow is >0, this is worse, improvement will be < 1.

    # Transition-based metrics: These are always calculated on state_assignments
    # For state_assignments_preview, we don't need the full list in JSON.
    len_naive_assignments = len(naive_metrics['state_assignments_preview']) if 'state_assignments_preview' in naive_metrics else len(naive_trajectory)
    len_flow_assignments = len(flow_metrics['state_assignments_preview']) if 'state_assignments_preview' in flow_metrics else len(flow_trajectory)

    # Re-calculate based on full trajectories if preview was used.
    # This requires having the full state_assignments. calculate_exploration_efficiency should return the full one.
    # For now, assume state_assignments_preview is NOT the full list, and we need original length.
    # Correct approach: calculate_exploration_efficiency must pass the full assignments or calculate transitions there.
    # Let's assume it does, or we re-calculate here from trajectory and identified centers.
    # For simplicity, let's assume the state_assignments were stored in full if needed by a helper.
    # The current implementation of compare_exploration_metrics recalculates transitions from full trajectories:
    
    naive_state_assignments_full, _ = identify_conformational_states(
        naive_trajectory, n_states, method=state_definition_method, custom_centers_source=custom_centers_id, target_type=target_type
    )
    flow_state_assignments_full, _ = identify_conformational_states(
        flow_trajectory, n_states, method=state_definition_method, custom_centers_source=custom_centers_id, target_type=target_type
    )

    naive_n_transitions = len([i for i in range(1, len(naive_state_assignments_full)) 
                              if naive_state_assignments_full[i] != naive_state_assignments_full[i-1]])
    flow_n_transitions = len([i for i in range(1, len(flow_state_assignments_full)) 
                             if flow_state_assignments_full[i] != flow_state_assignments_full[i-1]])
    
    naive_transition_rate = naive_n_transitions / max(1, len(naive_state_assignments_full)) # Avoid div by zero
    flow_transition_rate = flow_n_transitions / max(1, len(flow_state_assignments_full))   # Avoid div by zero
    transition_rate_improvement = flow_transition_rate / max(naive_transition_rate, 1e-6) # Use a small epsilon
    
    results = {
        # Store only summaries of naive_metrics and flow_metrics to keep JSON smaller
        'naive_summary': {
            'exploration_fraction': naive_metrics['exploration_fraction'],
            'normalized_entropy': naive_metrics['normalized_entropy'],
            'mean_round_trip_time': naive_metrics['mean_round_trip_time'],
            'n_actual_states_defined': naive_metrics['n_actual_states_defined_by_centers'],
            'n_states_visited_in_traj': naive_metrics['states_visited_in_traj'],
        },
        'flow_summary': {
            'exploration_fraction': flow_metrics['exploration_fraction'],
            'normalized_entropy': flow_metrics['normalized_entropy'],
            'mean_round_trip_time': flow_metrics['mean_round_trip_time'],
            'n_actual_states_defined': flow_metrics['n_actual_states_defined_by_centers'],
            'n_states_visited_in_traj': flow_metrics['states_visited_in_traj'],
        },
        'exploration_improvement': exploration_improvement,
        'entropy_improvement': entropy_improvement,
        'round_trip_improvement': round_trip_improvement, # Based on mean round trip time
        'naive_mean_round_trip': naive_mean_round_trip, # For direct comparison
        'flow_mean_round_trip': flow_mean_round_trip,   # For direct comparison
        'naive_n_transitions': naive_n_transitions,
        'flow_n_transitions': flow_n_transitions,
        'transition_rate_improvement': transition_rate_improvement,
        'naive_transition_rate': naive_transition_rate,
        'flow_transition_rate': flow_transition_rate,
        # Include full metrics for detailed inspection if needed, but they can be large
        # 'detailed_naive_metrics': naive_metrics, 
        # 'detailed_flow_metrics': flow_metrics,
    }
    
    # For plotting, we need the full round_trip_times list from detailed metrics
    results_for_plot = {**results} 
    results_for_plot['naive_metrics_for_plot'] = naive_metrics # Contains 'round_trip_times'
    results_for_plot['flow_metrics_for_plot'] = flow_metrics   # Contains 'round_trip_times'
    
    return results, results_for_plot


def plot_exploration_comparison(
    # results: Dict[str, Any], # This should be results_for_plot
    plot_data: Dict[str, Any], # Renamed to avoid confusion
    output_path: Path
) -> None:
    """
    Generate comparison plots of exploration metrics.
    
    Args:
        plot_data: Results dictionary from compare_exploration_metrics (the second item, results_for_plot)
        output_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12)) # Increased size
    fig.suptitle("Exploration Metrics Comparison: Flow-enhanced vs Vanilla PT", fontsize=16)
    
    naive_metrics_plot = plot_data['naive_metrics_for_plot'] # Use the detailed metrics for plotting
    flow_metrics_plot = plot_data['flow_metrics_for_plot']
    
    # Plot 1: State visitation comparison
    methods = ['Vanilla PT', 'Flow-enhanced PT']
    
    # Use n_actual_states_defined_by_centers for total_states if available and consistent
    # Fallback to requested n_states if not.
    total_states_naive = naive_metrics_plot.get('n_actual_states_defined_by_centers', naive_metrics_plot.get('n_states_requested_for_identification',0))
    total_states_flow = flow_metrics_plot.get('n_actual_states_defined_by_centers', flow_metrics_plot.get('n_states_requested_for_identification',0))
    
    # If using custom states, total_states_naive and total_states_flow should be the same.
    # We plot against the number of states that were *attempted* to be defined for a fair comparison bar.
    # If K-Means was used, this is n_states from input. If custom, it's number of custom centers.
    # Let's assume they are consistent if the same method was used.
    plot_total_states = max(total_states_naive, total_states_flow, 1) # Max of defined, at least 1 for plotting

    states_visited = [naive_metrics_plot['states_visited_in_traj'], flow_metrics_plot['states_visited_in_traj']]
    
    bars1 = ax1.bar(methods, states_visited, color=['#1f77b4', '#ff7f0e'], alpha=0.7) # Standard blue/orange
    ax1.axhline(y=plot_total_states, color='black', linestyle='--', alpha=0.5, label=f'Defined States ({plot_total_states})')
    ax1.set_ylabel('Unique States Visited in Trajectory')
    ax1.set_title('State Space Coverage')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(0, plot_total_states * 1.15 if plot_total_states > 0 else 1.15) # Adjust y-limit
    
    # Add percentage labels
    for bar, value in zip(bars1, states_visited):
        percentage = (value / plot_total_states) * 100 if plot_total_states > 0 else 0
        ax1.text(bar.get_x() + bar.get_width()/2., value + 0.02 * plot_total_states, # Position above bar
                f'{value} ({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Exploration entropy comparison  
    entropies = [naive_metrics_plot['normalized_entropy'], flow_metrics_plot['normalized_entropy']]
    bars2 = ax2.bar(methods, entropies, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    ax2.set_ylabel('Normalized State Visitation Entropy')
    ax2.set_title('Uniformity of Exploration')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, value in zip(bars2, entropies):
        ax2.text(bar.get_x() + bar.get_width()/2., value + 0.02, # Position above bar
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Round-trip time distributions or Transition Counts
    naive_round_trips = naive_metrics_plot.get('round_trip_times', [])
    flow_round_trips = flow_metrics_plot.get('round_trip_times', [])
    
    # Check if there are enough round trips for a meaningful histogram
    # A round trip requires at least two visits to the same state with an excursion.
    sufficient_rt_data_naive = len(naive_round_trips) >= 5 
    sufficient_rt_data_flow = len(flow_round_trips) >= 5

    if sufficient_rt_data_naive or sufficient_rt_data_flow: # Plot if at least one has enough data
        all_rt_times = naive_round_trips + flow_round_trips
        if not all_rt_times: all_rt_times = [0] # Ensure not empty for max
        
        max_time_overall = max(all_rt_times) if all_rt_times else 1
        bins = np.linspace(0, max_time_overall, min(20, int(max_time_overall / max(1, (max_time_overall // 20))) + 1 if max_time_overall > 0 else 2))
        
        if sufficient_rt_data_naive:
            ax3.hist(naive_round_trips, bins=bins, alpha=0.6, label='Vanilla PT', color='#1f77b4', density=True)
        if sufficient_rt_data_flow:
            ax3.hist(flow_round_trips, bins=bins, alpha=0.6, label='Flow-enhanced PT', color='#ff7f0e', density=True)
        
        ax3.set_xlabel('Round-trip Time (steps)')
        ax3.set_ylabel('Density')
        ax3.set_title('Round-trip Time Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        naive_mean_rt = plot_data.get('naive_mean_round_trip', 0)
        flow_mean_rt = plot_data.get('flow_mean_round_trip', 0)
        ax3.text(0.95, 0.95, f'Mean RT:\nNaive: {naive_mean_rt:.1f}\nFlow: {flow_mean_rt:.1f}', 
                transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    else:
        # Insufficient round-trip data - show transition counts as a proxy for dynamics
        ax3.text(0.5, 0.7, 'Insufficient round-trip data.\nPlotting state transition counts.', 
                 transform=ax3.transAxes, ha='center', va='center', fontsize=10, fontweight='bold')
        
        naive_transitions = plot_data.get('naive_n_transitions', 0)
        flow_transitions = plot_data.get('flow_n_transitions', 0)
        
        bar_labels_trans = ['Vanilla PT', 'Flow PT']
        transition_counts = [naive_transitions, flow_transitions]
        
        bars_trans = ax3.bar(bar_labels_trans, transition_counts, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax3.set_ylabel('Total State Transitions')
        ax3.set_title('State Transition Dynamics')
        ax3.grid(True, axis='y', alpha=0.3)
        for bar_t, val_t in zip(bars_trans, transition_counts):
            ax3.text(bar_t.get_x() + bar_t.get_width()/2., val_t + 0.02 * max(transition_counts if transition_counts else [1]),
                     f'{val_t}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Improvement metrics (using main results dict)
    main_results = plot_data # The top-level dictionary for improvement ratios
    improvement_names = ['Exploration\nFraction', 'Entropy\nUniformity', 'Transition\nRate']
    improvement_values = [
        main_results['exploration_improvement'],
        main_results['entropy_improvement'],
        main_results['transition_rate_improvement'] # Use transition rate as a robust measure
    ]
    
    # If mean round trip times were valid for both, offer that as an alternative or additional metric
    if main_results['naive_mean_round_trip'] > 0 and main_results['flow_mean_round_trip'] > 0:
        improvement_names.append('Round-trip\nEfficiency') # Lower is better for RT, so improvement is Naive/Flow
        improvement_values.append(main_results['round_trip_improvement'])

    colors = ['#2ca02c' if v > 1.0 else '#d62728' if v < 1.0 else '#7f7f7f' for v in improvement_values] # Green/Red/Grey
    bars4 = ax4.bar(improvement_names, improvement_values, color=colors, alpha=0.7)
    
    ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5) # No improvement line
    ax4.set_ylabel('Improvement Ratio (Flow / Vanilla)')
    ax4.set_title('Overall Exploration Improvements')
    # ax4.legend() # No legend needed if using color indication
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.spines['bottom'].set_position('zero') # y=0 line for x-axis
    
    # Add value labels
    for bar, value in zip(bars4, improvement_values):
        ax4.text(bar.get_x() + bar.get_width()/2., value + (0.05 if value >=0 else -0.15) , # Adjust based on value
                f'{value:.2f}×', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


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
    
    target_type_from_cfg = cfg.get("target", {}).get("type", "unknown").lower()
    
    try:
        # Build target system with CPU device
        import torch
        from src.accelmd.targets import build_target
        # Assuming integrated_autocorr_time.load_trajectory_data is appropriate
        # It loads the *flow* model to generate samples, which is one part.
        # We also need a "naive" trajectory.
        from .integrated_autocorr_time import load_trajectory_data 
        
        device = torch.device("cpu")  # Force CPU usage
        target = build_target(cfg, device) # Target at base temperature T=1.0
        
        _LOG.info("Generating sample trajectories for exploration analysis (this may take a moment)...")
        
        # Generate flow-enhanced samples: These are typically samples from the *flow model itself*,
        # not from a PT simulation run with the flow.
        # For a true comparison of "vanilla PT" vs "flow-enhanced PT", we'd ideally load
        # actual trajectories from such simulations. This script seems to generate synthetic ones.
        n_metric_samples = cfg.get("evaluation", {}).get("num_samples_for_exploration_metric", 5000)

        # Flow samples are typically from the model sampling its base distribution and transforming.
        # load_trajectory_data (from IACT script) loads the flow and calls model.sample()
        _LOG.info(f"Generating {n_metric_samples} samples using the trained flow model...")
        flow_trajectory = load_trajectory_data(model_path, target, n_samples=n_metric_samples) 
        
        # Generate naive samples: The original script created a highly autocorrelated trajectory.
        # For a more direct comparison of exploration from a similar starting point,
        # one might sample from the target distribution (e.g., at T_low from PT config).
        # Let's stick to the original script's intention of a "poorly exploring" naive trajectory.
        _LOG.info(f"Generating {n_metric_samples} 'naive' (highly autocorrelated) samples...")
        with torch.no_grad():
            # Start from a single sample from the target.
            initial_naive_sample = target.sample((1,)).cpu().numpy().flatten() 
            
        naive_trajectory_raw_samples = target.sample((n_metric_samples,)).cpu().numpy()

        naive_trajectory = np.zeros_like(naive_trajectory_raw_samples)
        if naive_trajectory.shape[0] > 0:
            naive_trajectory[0] = initial_naive_sample if initial_naive_sample.shape == naive_trajectory[0].shape else naive_trajectory_raw_samples[0]
            # Add strong autocorrelation to simulate poor exploration of a simple MCMC chain
            # The alpha here controls how "sticky" the naive sampler is.
            # Closer to 1 means more correlated / slower exploration.
            alpha_corr = cfg.get("evaluation",{}).get("naive_exploration_autocorr_factor", 0.95) 
            _LOG.info(f"Applying autocorrelation factor alpha={alpha_corr} to naive samples.")
            for i in range(1, len(naive_trajectory_raw_samples)):
                naive_trajectory[i] = alpha_corr * naive_trajectory[i-1] + np.sqrt(1-alpha_corr**2) * naive_trajectory_raw_samples[i]
        
        # Determine number of states based on dimensionality and trajectory length
        n_features = naive_trajectory.shape[1] if naive_trajectory.ndim > 1 else 1
        trajectory_length = len(naive_trajectory)
        
        # --- State Definition Logic ---
        state_definition_method = "kmeans" # Default
        custom_centers_id = None
        
        # Default n_states for K-Means, can be overridden by ALDP specific logic
        max_reasonable_states_kmeans = max(3, trajectory_length // 500) # ~500 samples per state
        n_states_kmeans = min(cfg.get("evaluation",{}).get("exploration_kmeans_n_states_max", 8), max_reasonable_states_kmeans)

        n_states_to_use = n_states_kmeans # Default for non-ALDP or if ALDP custom fails

        if "aldp" in target_type_from_cfg:
            # Try to use custom ALDP states first
            n_states_aldp_custom = cfg.get("evaluation",{}).get("exploration_aldp_custom_n_states", 16)
            # You might have different sets of custom states, e.g., "phi_psi_common", "phi_psi_rare"
            custom_centers_id_aldp = cfg.get("evaluation",{}).get("exploration_aldp_custom_centers_id", "default_aldp_states")
            # +++ Path to custom centers file +++
            aldp_custom_centers_file = cfg.get("evaluation", {}).get("exploration_aldp_custom_centers_file", None)
            
            # Check if custom states are intended to be used
            use_custom_aldp_states = cfg.get("evaluation",{}).get("exploration_use_custom_aldp_states", True) # Default to True if using ALDP

            if use_custom_aldp_states and aldp_custom_centers_file: # Must have file path to try custom
                _LOG.info(f"ALDP target: Attempting to use custom state definition from file '{aldp_custom_centers_file}' with up to {n_states_aldp_custom} states.")
                try:
                    # get_predefined_aldp_state_centers will handle loading and n_states adjustment
                    # We pass n_states_aldp_custom as the *desired* number from config.
                    # The function will return what it could load/generate.
                    test_centers = get_predefined_aldp_state_centers(n_states_aldp_custom, n_features, aldp_custom_centers_file)
                    if test_centers.size > 0 and test_centers.shape[0] > 0:
                        state_definition_method = "custom_aldp"
                        # The 'custom_centers_id' passed to compare_exploration_metrics should be the filepath
                        # so that identify_conformational_states can pass it to get_predefined_aldp_state_centers.
                        custom_centers_id = aldp_custom_centers_file 
                        n_states_to_use = test_centers.shape[0] # Use actual number of centers provided/loaded
                        _LOG.info(f"Successfully obtained {n_states_to_use} custom ALDP state centers from '{aldp_custom_centers_file}'.")
                    else:
                        _LOG.warning(f"Custom ALDP states from '{aldp_custom_centers_file}' not found or empty. Falling back to K-Means with {n_states_kmeans} states.")
                        state_definition_method = "kmeans" # Explicitly set back
                        custom_centers_id = None
                        n_states_to_use = n_states_kmeans
                except Exception as e_custom:
                    _LOG.warning(f"Error processing custom ALDP states from '{aldp_custom_centers_file}': {e_custom}. Falling back to K-Means with {n_states_kmeans} states.")
                    state_definition_method = "kmeans" # Explicitly set back
                    custom_centers_id = None
                    n_states_to_use = n_states_kmeans
            elif use_custom_aldp_states and not aldp_custom_centers_file:
                _LOG.warning(f"ALDP target: 'exploration_use_custom_aldp_states' is True, but "
                             f"'exploration_aldp_custom_centers_file' is not provided in config. "
                             f"Falling back to K-Means with {n_states_kmeans} states. "
                             f"Or, if you want to use the placeholder random centers, ensure a file path is given that does not exist.")
                state_definition_method = "kmeans"
                custom_centers_id = None
                n_states_to_use = n_states_kmeans
        else:
            _LOG.info(f"Non-ALDP target: Using K-Means for state definition with {n_states_kmeans} states.")
            state_definition_method = "kmeans"
            n_states_to_use = n_states_kmeans
        
        _LOG.info(f"Final state definition: Using method='{state_definition_method}', n_states={n_states_to_use} (features={n_features}, traj_len={trajectory_length})")
        
        # Calculate exploration metrics
        results, plot_data_dict = compare_exploration_metrics(
            naive_trajectory, 
            flow_trajectory, 
            n_states=n_states_to_use,
            state_definition_method=state_definition_method,
            custom_centers_id=custom_centers_id,
            target_type=target_type_from_cfg
        )
        
        # Generate plots
        temp_suffix = f"{cfg['pt']['temp_low']:.2f}_{cfg['pt']['temp_high']:.2f}"
        plot_path = output_dir / f"exploration_comparison_{temp_suffix}.png"
        plot_exploration_comparison(plot_data_dict, plot_path) # Pass the dict with detailed metrics for plotting
        _LOG.info("Exploration comparison plot saved to %s", plot_path)
        
        # Save metrics (using the summarized 'results' dict)
        metrics_path = metrics_dir / f"exploration_metrics_{temp_suffix}.json"
        
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2) # Save the summarized results
        
        _LOG.info("Exploration metrics saved to %s", metrics_path)
        _LOG.info("Exploration improvement (fraction of states): %.2f×", results['exploration_improvement'])
        _LOG.info("Round-trip efficiency improvement (based on mean RT or transition rate): %.2f×", results.get('round_trip_improvement', results.get('transition_rate_improvement', 1.0))) # Use transition rate if RT not available
        
    except Exception as e:
        _LOG.error("Failed to calculate exploration metrics: %s", str(e), exc_info=True)
        # Optionally re-raise, or allow pipeline to continue if this metric is non-critical
        # For now, let's re-raise to make sure issues are visible
        raise 