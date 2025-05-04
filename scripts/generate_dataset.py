import torch
import yaml
from typing import Tuple, Dict, List, Any, Callable, Union
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
import logging

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.targets.gmm import GMM
from main.sampler.sampler import ParallelTempering
from main.sampler.dyn_mcmc_warp import DynSamplerWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Get the appropriate device based on configuration and availability.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device: The device to use for computations
    """
    # Check if CUDA is requested and available
    if config["device"].lower() == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, switching to CPU.")
            return torch.device("cpu")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that the configuration has all required keys.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        KeyError: If a required key is missing
    """
    # Check top-level required keys
    top_level_required_keys = ["device", "gmm", "pt", "io"]
    for key in top_level_required_keys:
        if key not in config:
            raise KeyError(f"Required top-level configuration key '{key}' is missing")
    
    # Check GMM section keys
    gmm_required_keys = ["dim", "n_mixes", "loc_scaling"]
    for key in gmm_required_keys:
        if key not in config["gmm"]:
            raise KeyError(f"Required GMM configuration key '{key}' is missing")
    
    # Check PT section keys
    pt_required_keys = [
        "temp_low", "temp_high", "total_n_temp", "temp_schedule",
        "num_chains", "swap_interval", "step_size", "num_steps", "burn_in"
    ]
    for key in pt_required_keys:
        if key not in config["pt"]:
            raise KeyError(f"Required PT configuration key '{key}' is missing")
    
    # Check IO section keys
    io_required_keys = ["save_fold"]
    for key in io_required_keys:
        if key not in config["io"]:
            raise KeyError(f"Required IO configuration key '{key}' is missing")


def initialize_pt_coordinates(config: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, GMM]:
    """
    Initialize coordinates and temperature ladder for Parallel Tempering.
    
    Args:
        config: Configuration dictionary
        device: Device to use for computations
        
    Returns:
        Tuple containing:
            - x0: Initial samples of shape [total_n_temp, num_chains, dim]
            - temps: Temperature ladder of shape [total_n_temp]
            - target: GMM target distribution
    """
    # Validate configuration
    validate_config(config)
    
    # Set random seed for reproducibility if provided
    if "seed" in config:
        torch.manual_seed(config["seed"])
        logger.info(f"Using random seed: {config['seed']}")
    
    # Get configuration sections
    gmm_cfg = config["gmm"]
    pt_cfg = config["pt"]
    
    # Generate temperature ladder based on specified schedule
    if pt_cfg["temp_schedule"].lower() == "geom":
        temps = torch.logspace(
            start=torch.log10(torch.tensor(pt_cfg["temp_low"])),
            end=torch.log10(torch.tensor(pt_cfg["temp_high"])),
            steps=pt_cfg["total_n_temp"],
            device=device
        )
    elif pt_cfg["temp_schedule"].lower() == "linear":
        temps = torch.linspace(
            start=pt_cfg["temp_low"],
            end=pt_cfg["temp_high"],
            steps=pt_cfg["total_n_temp"],
            device=device
        )
    else:
        raise ValueError(f"Unknown temperature schedule: {pt_cfg['temp_schedule']}. "
                         f"Supported values are 'geom' and 'linear'.")
    
    # Get GMM parameters from config
    dim = gmm_cfg["dim"]
    n_mixes = gmm_cfg["n_mixes"]
    loc_scaling = gmm_cfg["loc_scaling"]
    
    # Initialize GMM target distribution and sample initial points
    target = GMM(dim=dim, n_mixes=n_mixes, loc_scaling=loc_scaling, device=device)
    x0 = target.sample((pt_cfg["total_n_temp"], pt_cfg["num_chains"])).to(device)
    
    # Log initialization details
    logger.info(f"Initialized PT coordinates with:")
    logger.info(f"- Initial samples shape: {x0.shape}")
    logger.info(f"- Temperature ladder: {temps}")
    logger.info(f"- Temperature schedule: {temps[1]/temps[0]:.3f} ratio between first temps")
    
    return x0, temps, target


def collect_pt_trajectories(
    x0: torch.Tensor, 
    temps: torch.Tensor, 
    config: Dict[str, Any],
    target: GMM,
    device: torch.device
) -> np.ndarray:
    """
    Run Parallel Tempering to collect trajectories.
    
    Args:
        x0: Initial samples of shape [total_n_temp, num_chains, dim]
        temps: Temperature ladder of shape [total_n_temp]
        config: Configuration dictionary
        target: GMM target distribution (preinitialized)
        device: Device to use for computations
        
    Returns:
        np.ndarray: A NumPy array of shape [num_temps, num_chains, n_samples, dim] containing the trajectories
    """
    # Get configuration sections
    pt_cfg = config["pt"]
    
    # Set up the step size with the correct broadcasting
    step_size = torch.tensor(pt_cfg["step_size"], device=device)
    total_chains = pt_cfg["total_n_temp"] * pt_cfg["num_chains"]
    step_size = step_size.repeat(total_chains).unsqueeze(-1)
    
    # Initialize the Parallel Tempering sampler
    pt = ParallelTempering(
        x=x0,
        energy_func=lambda x: -target.log_prob(x),
        step_size=step_size,
        swap_interval=pt_cfg["swap_interval"],
        temperatures=temps,
        device=device
    )
    
    # Wrap with DynSamplerWrapper
    pt = DynSamplerWrapper(
        pt, 
        per_temp=True, 
        total_n_temp=pt_cfg["total_n_temp"]
    )
    
    # Get parameters for PT run
    num_steps = pt_cfg["num_steps"]
    burn_in = pt_cfg["burn_in"]
    check_interval = pt_cfg.get("check_interval", 1000)
    
    # Ensure burn_in is less than num_steps to collect at least one sample
    if burn_in >= num_steps:
        logger.warning(f"burn_in ({burn_in}) >= num_steps ({num_steps}). Setting burn_in to num_steps-1 to collect at least one sample.")
        burn_in = max(0, num_steps - 1)
    
    # Calculate number of samples to collect after burn-in
    n_samples = num_steps - burn_in
    
    # Initialize array to store the collected samples
    samples_list = []
    
    # Run PT and collect trajectories
    logger.info(f"Running PT for {num_steps} steps (burn-in: {burn_in})...")
    for i in tqdm(range(num_steps)):
        x, acc, _ = pt.sample()
        
        # After burn-in period, collect samples
        if i >= burn_in:
            # Store the current state for all temperatures and chains
            samples_list.append(x.detach().cpu().numpy())
        
        # Log progress periodically
        if i % check_interval == 0:
            logger.info(f"[{i}] acc = {acc}, swap_rate = {pt.sampler.swap_rate:.3f}")
    
    # Safety check: ensure we have at least one sample
    if not samples_list:
        logger.warning("No samples were collected after burn-in. Adding final state as a sample.")
        samples_list.append(x.detach().cpu().numpy())
    
    # Convert list of samples to a well-shaped numpy array
    # [num_temps, num_chains, n_samples, dim]
    traj_array = np.stack(samples_list, axis=2)
    
    logger.info(f"Collected trajectory array of shape: {traj_array.shape}")
    return traj_array


def setup_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for output files.
    
    Args:
        config: Configuration dictionary
    """
    os.makedirs(config["io"]["save_fold"], exist_ok=True)


def save_trajectories(traj_array: np.ndarray, temps: torch.Tensor, config: Dict[str, Any]) -> None:
    """
    Save trajectories and temperatures to a .npz file.
    
    Args:
        traj_array: Array of trajectories
        temps: Temperature ladder
        config: Configuration dictionary
    """
    save_path = f"{config['io']['save_fold']}/gmm_PT_trajectories.npz"
    np.savez(save_path,
             traj=traj_array,
             temps=temps.cpu().numpy())
    logger.info(f"Saved trajectories to {save_path}")


def main(config_path: str = "configs/pt/gmm.yaml") -> None:
    """
    Main entry point for the script.
    
    Args:
        config_path: Path to the configuration file
    """
    # Load configuration once at the beginning
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up the device to use consistently throughout the code
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Create necessary directories
    setup_directories(config)
    
    # Initialize coordinates and get target
    x0, temps, target = initialize_pt_coordinates(config, device)
    
    # Collect PT trajectories
    traj_array = collect_pt_trajectories(x0, temps, config, target, device)
    
    # Save trajectories
    save_trajectories(traj_array, temps, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Parallel Tempering coordinates")
    parser.add_argument("--config", type=str, default="configs/pt/gmm.yaml",
                        help="Path to the configuration file")
    args = parser.parse_args()
    
    main(args.config) 