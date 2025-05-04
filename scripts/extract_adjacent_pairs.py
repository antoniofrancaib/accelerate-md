#!/usr/bin/env python
import os
import sys
import numpy as np
import argparse
import logging
from typing import List, Tuple

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trajectories(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the PT trajectories from a .npz file.
    
    Args:
        file_path: Path to the .npz file containing the trajectories
        
    Returns:
        Tuple containing:
            - traj: Array of trajectories of shape [num_temps, num_chains, n_samples, dim]
            - temps: Array of temperatures
    """
    try:
        data = np.load(file_path)
        traj = data['traj']
        temps = data['temps']
        
        logger.info(f"Loaded trajectories with shape: {traj.shape}")
        logger.info(f"Loaded temperatures with shape: {temps.shape}")
        
        return traj, temps
    except Exception as e:
        logger.error(f"Error loading trajectories from {file_path}: {e}")
        raise


def extract_adjacent_pairs(traj: np.ndarray, temps: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract pairs of adjacent temperature trajectories.
    
    Args:
        traj: Array of trajectories of shape [num_temps, num_chains, n_samples, dim]
        temps: Array of temperatures
        
    Returns:
        List of tuples containing pairs of adjacent temperature trajectories,
        where each element is flattened to shape [num_chains * n_samples, dim]
    """
    num_temps = traj.shape[0]
    pairs = []
    
    logger.info(f"Extracting adjacent temperature pairs from {num_temps} temperatures...")
    
    for k in range(num_temps - 1):
        # Extract adjacent temperature trajectories
        traj_k = traj[k]    # Shape: [num_chains, n_samples, dim]
        traj_k1 = traj[k+1] # Shape: [num_chains, n_samples, dim]
        
        # Flatten chains and steps
        # Original shape: [num_chains, n_samples, dim]
        # Reshape to: [num_chains * n_samples, dim]
        flat_traj_k = traj_k.reshape(-1, traj_k.shape[-1])
        flat_traj_k1 = traj_k1.reshape(-1, traj_k1.shape[-1])
        
        # Add to pairs list
        pairs.append((flat_traj_k, flat_traj_k1))
        
        logger.info(f"Pair {k}: T={temps[k]:.2f} -> T={temps[k+1]:.2f}, "
                   f"Shape: {flat_traj_k.shape}")
    
    return pairs


def save_pairs(pairs: List[Tuple[np.ndarray, np.ndarray]], temps: np.ndarray, output_path: str) -> None:
    """
    Save the extracted pairs to a .npz file.
    
    Args:
        pairs: List of tuples containing pairs of adjacent temperature trajectories
        temps: Array of temperatures
        output_path: Path to save the .npz file
    """
    try:
        # Convert pairs list to a format suitable for saving
        # Each pair is a tuple of two arrays, we need to convert to a single array
        pairs_array = np.array(pairs, dtype=object)
        
        np.savez(output_path, pairs=pairs_array, temps=temps)
        logger.info(f"Saved {len(pairs)} adjacent temperature pairs to {output_path}")
    except Exception as e:
        logger.error(f"Error saving pairs to {output_path}: {e}")
        raise


def main(input_path: str, output_path: str) -> None:
    """
    Main function to extract adjacent temperature pairs from PT trajectories.
    
    Args:
        input_path: Path to the input .npz file containing the trajectories
        output_path: Path to save the extracted pairs
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load trajectories
    traj, temps = load_trajectories(input_path)
    
    # Extract adjacent pairs
    pairs = extract_adjacent_pairs(traj, temps)
    
    # Save pairs
    save_pairs(pairs, temps, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract adjacent temperature pairs from PT trajectories")
    parser.add_argument("--input", type=str, default="data/pt/gmm_PT_trajectories.npz",
                        help="Path to the input .npz file containing the trajectories")
    parser.add_argument("--output", type=str, default="data/pt/gmm_pairs.npz",
                        help="Path to save the extracted pairs")
    args = parser.parse_args()
    
    main(args.input, args.output) 