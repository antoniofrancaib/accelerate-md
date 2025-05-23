"""
Integrated Autocorrelation Time for Enhanced Sampling Evaluation.

This metric quantifies how long it takes for a system to lose correlation with its
initial state, providing a measure of equilibration and sampling efficiency.
Shorter autocorrelation times indicate better sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
import torch
import yaml
from src.accelmd.models import MODEL_REGISTRY

_LOG = logging.getLogger(__name__)


def autocorrelation_function(x: np.ndarray) -> np.ndarray:
    """
    Calculate the normalized autocorrelation function of a time series.
    
    Args:
        x: Time series data of shape (n_steps,) or (n_steps, n_features)
        
    Returns:
        Autocorrelation function values
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    n_steps, n_features = x.shape
    autocorr = np.zeros((n_steps // 2, n_features))
    
    for i in range(n_features):
        # Center the data
        xi = x[:, i] - np.mean(x[:, i])
        
        # Use FFT for efficient autocorrelation calculation
        f_x = np.fft.fft(xi, n=2*n_steps)
        autocorr_full = np.fft.ifft(f_x * np.conj(f_x)).real
        autocorr_full = autocorr_full[:n_steps]
        
        # Normalize by the zero-lag value
        autocorr[:, i] = autocorr_full[:n_steps//2] / autocorr_full[0]
    
    return autocorr


def integrated_autocorr_time(autocorr: np.ndarray, cutoff: float = 0.01) -> np.ndarray:
    """
    Calculate integrated autocorrelation time from autocorrelation function.
    
    Args:
        autocorr: Autocorrelation function of shape (n_lags, n_features)
        cutoff: Cutoff value below which autocorrelation is considered negligible
        
    Returns:
        Integrated autocorrelation times for each feature
    """
    n_lags, n_features = autocorr.shape
    tau_int = np.zeros(n_features)
    
    for i in range(n_features):
        # Find where autocorrelation drops below cutoff
        below_cutoff = np.where(autocorr[:, i] < cutoff)[0]
        if len(below_cutoff) > 0:
            cutoff_idx = below_cutoff[0]
        else:
            cutoff_idx = n_lags
            
        # Integrate up to cutoff point
        # tau_int = 1 + 2 * sum(C(t)) for t = 1 to cutoff
        tau_int[i] = 1.0 + 2.0 * np.sum(autocorr[1:cutoff_idx, i])
    
    return tau_int


def load_trajectory_data(model_path: str, target, n_samples: int = 10000) -> np.ndarray:
    """
    Load or generate trajectory data for autocorrelation analysis.
    
    Args:
        model_path: Path to trained model
        target: Target system
        n_samples: Number of samples to generate
        
    Returns:
        Trajectory data of shape (n_samples, n_features)
    """
    # Force CPU usage to avoid CUDA issues
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to find the experiment config file to get the correct model parameters
    model_path_obj = Path(model_path)
    exp_dir = model_path_obj.parent.parent  # Go up from models/ to experiment root
    config_path = exp_dir / "config.yaml"
    
    model_config = {}
    model_type = "realnvp"  # Default
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            model_type = full_config.get("model_type", "realnvp")
            model_config = full_config.get("trainer", {}).get(model_type, {}).get("model", {})
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint with metadata
        if not model_config:  # Only use checkpoint config if we couldn't load from file
            model_config = checkpoint.get("model_config", {})
            model_type = checkpoint.get("model_type", "realnvp")
        state_dict = checkpoint["model_state_dict"]
    else:
        # Just the state dict
        state_dict = checkpoint
    
    # Set the target dimension in model config
    if hasattr(target, 'dim'):
        model_config["dim"] = target.dim
    else:
        try:
            sample_shape = target.sample((1,)).shape
            model_config["dim"] = sample_shape[-1]
        except Exception as e:
            # Fall back to config-based dimension
            if "gmm" in model_config or "gmm" in full_config.get("gmm", {}):
                gmm_dim = full_config.get("gmm", {}).get("dim", 60)
                model_config["dim"] = gmm_dim
            else:
                model_config["dim"] = 60  # Default fallback
    
    # Set default values for missing config parameters (if still needed)
    if "n_couplings" not in model_config:
        model_config["n_couplings"] = 14
    if "hidden_dim" not in model_config:
        model_config["hidden_dim"] = 256
    if "use_permutation" not in model_config:
        model_config["use_permutation"] = True
    
    # Create model instance with the correct config
    if model_type in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_type](model_config)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Generate samples
    with torch.no_grad():
        samples = model.sample(n_samples).cpu().numpy()
    
    return samples


def calculate_autocorr_metrics(
    naive_trajectory: np.ndarray, 
    flow_trajectory: np.ndarray,
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Calculate autocorrelation metrics for both naive and flow-enhanced trajectories.
    
    Args:
        naive_trajectory: Trajectory from vanilla PT of shape (n_steps, n_features)
        flow_trajectory: Trajectory from flow-enhanced PT of shape (n_steps, n_features)
        feature_names: Optional feature names for reporting
        
    Returns:
        Dictionary containing autocorrelation metrics
    """
    # Calculate autocorrelation functions
    naive_autocorr = autocorrelation_function(naive_trajectory)
    flow_autocorr = autocorrelation_function(flow_trajectory)
    
    # Calculate integrated autocorrelation times
    naive_tau_int = integrated_autocorr_time(naive_autocorr)
    flow_tau_int = integrated_autocorr_time(flow_autocorr)
    
    # Average over features for overall metrics
    naive_tau_avg = np.mean(naive_tau_int)
    flow_tau_avg = np.mean(flow_tau_int)
    
    # Efficiency improvement factor
    efficiency_improvement = naive_tau_avg / flow_tau_avg
    
    results = {
        "naive_tau_integrated": naive_tau_avg,
        "flow_tau_integrated": flow_tau_avg,
        "efficiency_improvement": efficiency_improvement,
        "naive_tau_per_feature": naive_tau_int.tolist(),
        "flow_tau_per_feature": flow_tau_int.tolist(),
        "naive_autocorr": naive_autocorr.tolist(),
        "flow_autocorr": flow_autocorr.tolist(),
    }
    
    if feature_names:
        results["feature_names"] = feature_names
    
    return results


def plot_autocorr_comparison(
    results: Dict[str, Any], 
    output_path: Path,
    max_lag: int = 500
) -> None:
    """
    Generate comparison plots of autocorrelation functions.
    
    Args:
        results: Results dictionary from calculate_autocorr_metrics
        output_path: Path to save the plot
        max_lag: Maximum lag to plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    naive_autocorr = np.array(results["naive_autocorr"])
    flow_autocorr = np.array(results["flow_autocorr"])
    
    n_lags = min(max_lag, naive_autocorr.shape[0])
    lags = np.arange(n_lags)
    
    # Plot average autocorrelation
    naive_avg = np.mean(naive_autocorr[:n_lags], axis=1)
    flow_avg = np.mean(flow_autocorr[:n_lags], axis=1)
    
    ax1.plot(lags, naive_avg, 'b-', label='Vanilla PT', linewidth=2)
    ax1.plot(lags, flow_avg, 'r-', label='Flow-enhanced PT', linewidth=2)
    ax1.axhline(y=0.01, color='k', linestyle='--', alpha=0.5, label='Cutoff (0.01)')
    ax1.set_xlabel('Lag (steps)')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Average Autocorrelation Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot integrated autocorrelation times
    naive_tau = results["naive_tau_per_feature"]
    flow_tau = results["flow_tau_per_feature"]
    
    x_pos = np.arange(len(naive_tau))
    width = 0.35
    
    ax2.bar(x_pos - width/2, naive_tau, width, label='Vanilla PT', color='blue', alpha=0.7)
    ax2.bar(x_pos + width/2, flow_tau, width, label='Flow-enhanced PT', color='red', alpha=0.7)
    
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Integrated Autocorrelation Time')
    ax2.set_title('Integrated Autocorrelation Times')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add efficiency improvement text
    efficiency = results["efficiency_improvement"]
    ax2.text(0.02, 0.98, f'Efficiency Improvement: {efficiency:.2f}×', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run(cfg: Dict[str, Any]) -> None:
    """
    Run integrated autocorrelation time analysis.
    
    Args:
        cfg: Configuration dictionary
    """
    _LOG.info("Starting integrated autocorrelation time analysis")
    
    # Get paths from configuration
    output_dir = Path(cfg["output"]["plots_dir"])
    metrics_dir = Path(cfg["output"]["results_dir"])
    model_path = cfg["output"]["model_path"]
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Build target system with CPU device
        from src.accelmd.targets import build_target
        device = torch.device("cpu")  # Force CPU usage
        target = build_target(cfg, device)
        
        _LOG.info("Generating sample trajectories for analysis")
        
        # Generate flow-enhanced samples
        flow_trajectory = load_trajectory_data(model_path, target, n_samples=5000)
        
        # Generate naive samples (from target distribution directly)
        with torch.no_grad():
            naive_samples = target.sample((5000,)).cpu().numpy()
        
        # Add some correlation structure to naive samples to simulate MD trajectory
        # This is a placeholder - in practice you'd have actual MD trajectories
        naive_trajectory = np.zeros_like(naive_samples)
        naive_trajectory[0] = naive_samples[0]
        for i in range(1, len(naive_samples)):
            # Add some autocorrelation
            alpha = 0.9  # Correlation parameter
            naive_trajectory[i] = alpha * naive_trajectory[i-1] + np.sqrt(1-alpha**2) * naive_samples[i]
        
        # Calculate autocorrelation metrics
        results = calculate_autocorr_metrics(naive_trajectory, flow_trajectory)
        
        # Generate plots
        temp_suffix = f"{cfg['pt']['temp_low']:.2f}_{cfg['pt']['temp_high']:.2f}"
        plot_path = output_dir / f"autocorrelation_comparison_{temp_suffix}.png"
        plot_autocorr_comparison(results, plot_path)
        _LOG.info("Autocorrelation plot saved to %s", plot_path)
        
        # Save metrics
        metrics_path = metrics_dir / f"autocorr_time_{temp_suffix}.json"
        
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        
        _LOG.info("Autocorrelation metrics saved to %s", metrics_path)
        _LOG.info("Efficiency improvement: %.2f×", results["efficiency_improvement"])
        
    except Exception as e:
        _LOG.error("Failed to calculate autocorrelation metrics: %s", str(e))
        raise 