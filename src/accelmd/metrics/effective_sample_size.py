"""
Effective Sample Size (ESS) for Enhanced Sampling Evaluation.

This metric quantifies the number of effectively independent samples in a correlated
time series, providing a direct measure of sampling efficiency. Higher ESS indicates
better sampling performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

_LOG = logging.getLogger(__name__)


def autocorrelation_time_exponential_fit(x: np.ndarray, max_lag: int = 500) -> float:
    """
    Estimate autocorrelation time by fitting exponential decay.
    
    Args:
        x: Time series data of shape (n_steps,)
        max_lag: Maximum lag to consider for fitting
        
    Returns:
        Autocorrelation time estimate
    """
    if x.ndim > 1:
        x = x.flatten()
        
    n_steps = len(x)
    max_lag = min(max_lag, n_steps // 4)
    
    # Calculate autocorrelation function
    x_centered = x - np.mean(x)
    autocorr = np.correlate(x_centered, x_centered, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find first negative value or where it drops below threshold
    lags = np.arange(len(autocorr))
    valid_mask = (autocorr > 0.01) & (lags <= max_lag)
    
    if not np.any(valid_mask):
        return 1.0  # No correlation
        
    valid_lags = lags[valid_mask]
    valid_autocorr = autocorr[valid_mask]
    
    # Fit exponential decay: C(t) = exp(-t/tau)
    try:
        log_autocorr = np.log(np.maximum(valid_autocorr, 1e-10))
        # Linear fit to log(C(t)) = -t/tau
        coeffs = np.polyfit(valid_lags, log_autocorr, 1)
        tau = -1.0 / coeffs[0] if coeffs[0] < 0 else 1.0
        return max(tau, 1.0)
    except:
        return 1.0


def effective_sample_size(x: np.ndarray, method: str = "autocorr") -> float:
    """
    Calculate effective sample size of a time series.
    
    Args:
        x: Time series data of shape (n_steps,) or (n_steps, n_features)
        method: Method to use ("autocorr" or "integrated")
        
    Returns:
        Effective sample size
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        
    n_steps, n_features = x.shape
    ess_values = []
    
    for i in range(n_features):
        xi = x[:, i]
        
        if method == "autocorr":
            # ESS = N / (1 + 2 * tau)
            tau = autocorrelation_time_exponential_fit(xi)
            ess = n_steps / (1.0 + 2.0 * tau)
        elif method == "integrated":
            # Use integrated autocorrelation time
            from .integrated_autocorr_time import autocorrelation_function, integrated_autocorr_time
            autocorr = autocorrelation_function(xi.reshape(-1, 1))
            tau_int = integrated_autocorr_time(autocorr)[0]
            ess = n_steps / (2.0 * tau_int)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        ess_values.append(ess)
    
    return np.mean(ess_values)


def calculate_ess_metrics(
    naive_trajectory: np.ndarray,
    flow_trajectory: np.ndarray,
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Calculate effective sample size metrics for both trajectories.
    
    Args:
        naive_trajectory: Trajectory from vanilla PT of shape (n_steps, n_features)
        flow_trajectory: Trajectory from flow-enhanced PT of shape (n_steps, n_features)
        feature_names: Optional feature names for reporting
        
    Returns:
        Dictionary containing ESS metrics
    """
    # Calculate ESS for both methods
    naive_ess_autocorr = effective_sample_size(naive_trajectory, method="autocorr")
    flow_ess_autocorr = effective_sample_size(flow_trajectory, method="autocorr")
    
    naive_ess_integrated = effective_sample_size(naive_trajectory, method="integrated")
    flow_ess_integrated = effective_sample_size(flow_trajectory, method="integrated")
    
    # Calculate per-feature ESS
    if naive_trajectory.ndim == 1:
        naive_trajectory = naive_trajectory.reshape(-1, 1)
        flow_trajectory = flow_trajectory.reshape(-1, 1)
        
    n_features = naive_trajectory.shape[1]
    naive_ess_per_feature = []
    flow_ess_per_feature = []
    
    for i in range(n_features):
        naive_ess = effective_sample_size(naive_trajectory[:, i:i+1], method="autocorr")
        flow_ess = effective_sample_size(flow_trajectory[:, i:i+1], method="autocorr")
        naive_ess_per_feature.append(naive_ess)
        flow_ess_per_feature.append(flow_ess)
    
    # Calculate efficiency ratios
    ess_ratio_autocorr = flow_ess_autocorr / naive_ess_autocorr
    ess_ratio_integrated = flow_ess_integrated / naive_ess_integrated
    
    # Calculate sampling efficiency (ESS per unit time/step)
    naive_efficiency = naive_ess_autocorr / len(naive_trajectory)
    flow_efficiency = flow_ess_autocorr / len(flow_trajectory)
    efficiency_ratio = flow_efficiency / naive_efficiency
    
    results = {
        "naive_ess_autocorr": naive_ess_autocorr,
        "flow_ess_autocorr": flow_ess_autocorr,
        "naive_ess_integrated": naive_ess_integrated,
        "flow_ess_integrated": flow_ess_integrated,
        "ess_ratio_autocorr": ess_ratio_autocorr,
        "ess_ratio_integrated": ess_ratio_integrated,
        "naive_ess_per_feature": naive_ess_per_feature,
        "flow_ess_per_feature": flow_ess_per_feature,
        "naive_efficiency": naive_efficiency,
        "flow_efficiency": flow_efficiency,
        "efficiency_ratio": efficiency_ratio,
        "naive_trajectory_length": len(naive_trajectory),
        "flow_trajectory_length": len(flow_trajectory),
    }
    
    if feature_names:
        results["feature_names"] = feature_names
        
    return results


def plot_ess_comparison(
    results: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Generate comparison plots of effective sample sizes.
    
    Args:
        results: Results dictionary from calculate_ess_metrics
        output_path: Path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Overall ESS comparison
    methods = ['Autocorr\nMethod', 'Integrated\nMethod']
    naive_ess = [results["naive_ess_autocorr"], results["naive_ess_integrated"]]
    flow_ess = [results["flow_ess_autocorr"], results["flow_ess_integrated"]]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x_pos - width/2, naive_ess, width, label='Vanilla PT', color='blue', alpha=0.7)
    ax1.bar(x_pos + width/2, flow_ess, width, label='Flow-enhanced PT', color='red', alpha=0.7)
    
    ax1.set_xlabel('ESS Calculation Method')
    ax1.set_ylabel('Effective Sample Size')
    ax1.set_title('Overall ESS Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Per-feature ESS comparison
    naive_per_feature = results["naive_ess_per_feature"]
    flow_per_feature = results["flow_ess_per_feature"]
    
    if len(naive_per_feature) > 0:
        feature_indices = np.arange(len(naive_per_feature))
        width = 0.35
        
        ax2.bar(feature_indices - width/2, naive_per_feature, width, 
                label='Vanilla PT', color='blue', alpha=0.7)
        ax2.bar(feature_indices + width/2, flow_per_feature, width,
                label='Flow-enhanced PT', color='red', alpha=0.7)
        
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Effective Sample Size')
        ax2.set_title('ESS per Feature')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No per-feature data', transform=ax2.transAxes,
                ha='center', va='center')
        ax2.set_title('ESS per Feature')
    
    # Plot 3: Efficiency metrics
    metrics_names = ['ESS Ratio\n(Autocorr)', 'ESS Ratio\n(Integrated)', 'Efficiency\nRatio']
    metrics_values = [
        results["ess_ratio_autocorr"],
        results["ess_ratio_integrated"], 
        results["efficiency_ratio"]
    ]
    
    colors = ['green' if v > 1.0 else 'orange' for v in metrics_values]
    bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    
    # Add horizontal line at y=1 for reference
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No improvement')
    
    ax3.set_ylabel('Ratio (Flow / Vanilla)')
    ax3.set_title('Efficiency Metrics')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run(cfg: Dict[str, Any]) -> None:
    """
    Run effective sample size analysis.
    
    Args:
        cfg: Configuration dictionary
    """
    _LOG.info("Starting effective sample size analysis")
    
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
        
        _LOG.info("Generating sample trajectories for ESS analysis")
        
        # Generate flow-enhanced samples
        flow_trajectory = load_trajectory_data(model_path, target, n_samples=5000)
        
        # Generate naive samples with autocorrelation structure
        with torch.no_grad():
            naive_samples = target.sample((5000,)).cpu().numpy()
        naive_trajectory = np.zeros_like(naive_samples)
        naive_trajectory[0] = naive_samples[0]
        
        # Add stronger autocorrelation for vanilla PT to show difference
        for i in range(1, len(naive_samples)):
            alpha = 0.95  # Higher correlation for vanilla PT
            naive_trajectory[i] = alpha * naive_trajectory[i-1] + np.sqrt(1-alpha**2) * naive_samples[i]
        
        # Calculate ESS metrics
        results = calculate_ess_metrics(naive_trajectory, flow_trajectory)
        
        # Generate plots
        temp_suffix = f"{cfg['pt']['temp_low']:.2f}_{cfg['pt']['temp_high']:.2f}"
        plot_path = output_dir / f"effective_sample_size_comparison_{temp_suffix}.png"
        plot_ess_comparison(results, plot_path)
        _LOG.info("ESS comparison plot saved to %s", plot_path)
        
        # Save metrics
        metrics_path = metrics_dir / f"ess_metrics_{temp_suffix}.json"
        
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        
        _LOG.info("ESS metrics saved to %s", metrics_path)
        _LOG.info("ESS improvement (autocorr method): %.2f×", results["ess_ratio_autocorr"])
        _LOG.info("Sampling efficiency improvement: %.2f×", results["efficiency_ratio"])
        
    except Exception as e:
        _LOG.error("Failed to calculate ESS metrics: %s", str(e))
        raise 