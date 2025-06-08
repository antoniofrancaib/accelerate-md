"""
Dataset class for loading temperature pairs from parallel tempering simulation data.

This replaces the problematic TemperedPairDataset that generates fresh samples,
instead using equilibrated samples from PT simulations.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from typing import Union, Tuple, Optional
import tempfile
import numpy as np

# Import coordinate transform dependencies
import boltzgen as bg
from openmm import unit
import openmm as mm
from openmmtools import testsystems
import openmm.app as app
from src.accelmd.utils.se3_utils import remove_mean

logger = logging.getLogger(__name__)


def create_coordinate_transform(data_path: str = None, transform_type: str = "cartesian"):
    """Create the same CoordinateTransform used by AldpPotentialCart."""
    
    # Set up the same parameters as in AldpBoltzmann.__init__
    temperature = 300
    env = 'implicit'
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    shift_dih = False
    shift_dih_params = {'hist_bins': 100}
    default_std = {'bond': 0.005, 'angle': 0.15, 'dih': 0.2}
    
    # Define z_matrix (same as in boltzmann.py)
    z_matrix = [
        (0, [1, 4, 6]),
        (1, [4, 6, 8]),
        (2, [1, 4, 0]),
        (3, [1, 4, 0]),
        (4, [6, 8, 14]),
        (5, [4, 6, 8]),
        (7, [6, 8, 4]),
        (9, [8, 6, 4]),
        (10, [8, 6, 4]),
        (11, [10, 8, 6]),
        (12, [10, 8, 11]),
        (13, [10, 8, 11]),
        (15, [14, 8, 16]),
        (16, [14, 8, 6]),
        (17, [16, 14, 15]),
        (18, [16, 14, 8]),
        (19, [18, 16, 14]),
        (20, [18, 16, 19]),
        (21, [18, 16, 19])
    ]
    cart_indices = [8, 6, 14]
    
    # Generate reference data if no data_path provided
    if data_path is None:
        logger.info("Generating reference trajectory for coordinate transform...")
        system = testsystems.AlanineDipeptideImplicit(constraints=None)
        sim = app.Simulation(system.topology, system.system,
                             mm.LangevinIntegrator(temperature * unit.kelvin,
                                                   1. / unit.picosecond,
                                                   1. * unit.femtosecond),
                             mm.Platform.getPlatformByName('Reference'))
        sim.context.setPositions(system.positions)
        sim.minimizeEnergy()
        state = sim.context.getState(getPositions=True)
        position = state.getPositions(True).value_in_unit(unit.nanometer)
        
        tmp_dir = tempfile.gettempdir()
        data_path = tmp_dir + '/aldp_transform_ref.pt'
        transform_data = torch.tensor(position.reshape(1, 66).astype(np.float64))
        torch.save(transform_data, data_path)
        logger.info(f"Reference data saved to {data_path}")
    else:
        # Load the provided data
        if Path(data_path).suffix == '.pt':
            transform_data = torch.load(data_path, weights_only=True)
        else:
            raise NotImplementedError(f"Unsupported data format: {data_path}")
    
    # Set distribution mode (same as boltzmann.py logic)
    mode = "mixed" if transform_type == 'mixed' else "internal"
    
    # Create coordinate transform
    coordinate_transform = bg.flows.CoordinateTransform(
        transform_data, 66, z_matrix, cart_indices, mode=mode,
        ind_circ_dih=ind_circ_dih, shift_dih=shift_dih,
        shift_dih_params=shift_dih_params, default_std=default_std
    )
    
    return coordinate_transform


class PTTemperedPairDataset(Dataset):
    """
    Dataset that loads temperature pairs from parallel tempering simulation data.
    
    This dataset extracts equilibrated samples from PT simulations and creates
    (high_temp, low_temp) pairs for training normalizing flows.
    
    IMPORTANT: This class now applies CoordinateTransform to raw Cartesian data
    to avoid scale mismatch issues that cause numerical instability.
    
    Parameters
    ----------
    pt_data_path : str or Path
        Path to the PT simulation data file (.pt format)
    temp_low_idx : int
        Index of the lower temperature in the PT temperature ladder
    temp_high_idx : int
        Index of the higher temperature in the PT temperature ladder
    n_samples : int, optional
        Number of sample pairs to extract. If None, uses all available post-burn-in samples
    burn_in_override : int, optional
        Override burn-in period from PT data. If None, uses value from PT data
    subsample_factor : int, optional
        Factor for subsampling the trajectory to reduce correlation (default: 1, no subsampling)
    apply_coordinate_transform : bool, optional
        Whether to apply coordinate transform to fix scale mismatch (default: True)
    cache_cleaned_data : bool, optional
        Whether to cache the coordinate-transformed data to disk (default: True)
    coordinate_transform_ref : str, optional
        Path to reference data for coordinate transform initialization
    """
    
    def __init__(
        self, 
        pt_data_path: Union[str, Path],
        temp_low_idx: int,
        temp_high_idx: int,
        n_samples: Optional[int] = None,
        burn_in_override: Optional[int] = None,
        subsample_factor: int = 1,
        apply_coordinate_transform: bool = True,
        cache_cleaned_data: bool = True,
        coordinate_transform_ref: Optional[str] = "./datasets/aldp/position_min_energy.pt"
    ):
        self.pt_data_path = Path(pt_data_path)
        self.temp_low_idx = temp_low_idx
        self.temp_high_idx = temp_high_idx
        self.subsample_factor = subsample_factor
        self.apply_coordinate_transform = apply_coordinate_transform
        self.cache_cleaned_data = cache_cleaned_data
        self.coordinate_transform_ref = coordinate_transform_ref
        self.burn_in_override = burn_in_override
        self.n_samples = n_samples
        
        # Generate cache path for cleaned data
        cache_suffix = f"_clean_T{temp_low_idx}to{temp_high_idx}_sub{subsample_factor}"
        self.cache_path = self.pt_data_path.parent / f"{self.pt_data_path.stem}{cache_suffix}.pt"
        
        # Load or create cleaned data
        if self.cache_cleaned_data and self.cache_path.exists() and self.apply_coordinate_transform:
            logger.info(f"Loading cached cleaned data from {self.cache_path}")
            self._load_from_cache()
        else:
            logger.info(f"Processing raw PT data from {self.pt_data_path}")
            self._load_and_process_raw_data()
            
            if self.cache_cleaned_data and self.apply_coordinate_transform:
                logger.info(f"Caching cleaned data to {self.cache_path}")
                self._save_to_cache()
    
    def _load_from_cache(self):
        """Load pre-processed data from cache."""
        cached_data = torch.load(self.cache_path, map_location='cpu')
        
        # Extract data
        self.low_temp_samples = cached_data['low_temp_samples']
        self.high_temp_samples = cached_data['high_temp_samples'] 
        self.temperatures = cached_data['temperatures']
        self.n_samples = cached_data['n_samples']
        self.burn_in = cached_data['burn_in']
        
        # Log dataset info
        T_low = self.temperatures[self.temp_low_idx].item()
        T_high = self.temperatures[self.temp_high_idx].item()
        logger.info(f"Created PT dataset: T={T_high:.3f} → T={T_low:.3f}")
        logger.info(f"  Total samples: {self.n_samples}")
        logger.info(f"  Sample shape: {self.low_temp_samples.shape}")
        logger.info(f"  Burn-in used: {self.burn_in} steps")
        logger.info(f"  Subsample factor: {self.subsample_factor}")
    
    def _save_to_cache(self):
        """Save processed data to cache."""
        cached_data = {
            'low_temp_samples': self.low_temp_samples,
            'high_temp_samples': self.high_temp_samples,
            'temperatures': self.temperatures,
            'n_samples': self.n_samples,
            'burn_in': self.burn_in,
            'subsample_factor': self.subsample_factor,
            'transform_applied': self.apply_coordinate_transform,
            'coordinate_transform_ref': self.coordinate_transform_ref
        }
        torch.save(cached_data, self.cache_path)
        
    def _load_and_process_raw_data(self):
        """Load and process raw PT data, applying coordinate transform if requested."""
        
        # Load PT simulation data
        logger.info(f"Loading PT data from {self.pt_data_path}")
        try:
            self.pt_data = torch.load(self.pt_data_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load PT data from {self.pt_data_path}: {e}")
        
        # Extract metadata
        self.trajectory = self.pt_data['trajectory']  # [n_temps, n_chains, n_steps, dim]
        self.temperatures = self.pt_data['temperatures']
        self.config = self.pt_data['config']
        
        # Determine burn-in period
        self.burn_in = self.burn_in_override if self.burn_in_override is not None else self.pt_data.get('burn_in', 0)
        
        # Validate temperature indices
        n_temps = self.trajectory.shape[0]
        if self.temp_low_idx >= n_temps or self.temp_high_idx >= n_temps:
            raise ValueError(f"Temperature indices ({self.temp_low_idx}, {self.temp_high_idx}) "
                           f"exceed available temperatures (0-{n_temps-1})")
        
        if self.temp_low_idx >= self.temp_high_idx:
            raise ValueError(f"Low temperature index ({self.temp_low_idx}) must be less than "
                           f"high temperature index ({self.temp_high_idx})")
        
        # Extract equilibrated samples (post burn-in)
        if self.burn_in >= self.trajectory.shape[2]:
            logger.warning(f"Burn-in period ({self.burn_in}) >= trajectory length ({self.trajectory.shape[2]}). "
                          "Using all samples.")
            self.burn_in = 0
        
        # Apply burn-in and subsampling
        start_idx = self.burn_in
        equilibrated_traj = self.trajectory[:, :, start_idx::self.subsample_factor, :]  # [n_temps, n_chains, n_eq_steps, dim]
        
        # Extract samples for the specific temperature pair
        low_temp_samples = equilibrated_traj[self.temp_low_idx]   # [n_chains, n_eq_steps, dim]
        high_temp_samples = equilibrated_traj[self.temp_high_idx] # [n_chains, n_eq_steps, dim]
        
        # Flatten chain and time dimensions: [n_chains * n_eq_steps, dim]
        low_temp_samples = low_temp_samples.reshape(-1, low_temp_samples.shape[-1])
        high_temp_samples = high_temp_samples.reshape(-1, high_temp_samples.shape[-1])
        
        # Limit to n_samples if specified  
        available_samples = min(len(low_temp_samples), len(high_temp_samples))
        if self.n_samples is not None and self.n_samples < available_samples:
            # Randomly sample n_samples indices
            indices = torch.randperm(available_samples)[:self.n_samples]
            low_temp_samples = low_temp_samples[indices]
            high_temp_samples = high_temp_samples[indices]
        else:
            self.n_samples = available_samples
        
        # Apply coordinate transform if requested (CRITICAL for numerical stability)
        if self.apply_coordinate_transform:
            logger.info("Applying coordinate transform to fix scale mismatch...")
            
            # Create coordinate transform
            coordinate_transform = create_coordinate_transform(
                data_path=self.coordinate_transform_ref, 
                transform_type="cartesian"
            )
            
            # Transform both temperature samples
            logger.info("Transforming low temperature samples...")
            low_temp_clean, _ = coordinate_transform.forward(low_temp_samples.double())
            low_temp_clean = low_temp_clean.float()
            low_temp_clean = remove_mean(low_temp_clean, n_particles=22, n_dimensions=3)
            
            logger.info("Transforming high temperature samples...")
            high_temp_clean, _ = coordinate_transform.forward(high_temp_samples.double())
            high_temp_clean = high_temp_clean.float()
            high_temp_clean = remove_mean(high_temp_clean, n_particles=22, n_dimensions=3)
            
            self.low_temp_samples = low_temp_clean
            self.high_temp_samples = high_temp_clean
            
            logger.info("Coordinate transform applied successfully")
        else:
            logger.warning("Coordinate transform disabled - may cause numerical instability!")
            self.low_temp_samples = low_temp_samples
            self.high_temp_samples = high_temp_samples
        
        # Log dataset info
        T_low = self.temperatures[self.temp_low_idx].item()
        T_high = self.temperatures[self.temp_high_idx].item()
        logger.info(f"Created PT dataset: T={T_high:.3f} → T={T_low:.3f}")
        logger.info(f"  Total samples: {self.n_samples}")
        logger.info(f"  Sample shape: {self.low_temp_samples.shape}")
        logger.info(f"  Burn-in used: {self.burn_in} steps")
        logger.info(f"  Subsample factor: {self.subsample_factor}")
        logger.info(f"  Transform applied: {self.apply_coordinate_transform}")
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a (high_temp, low_temp) sample pair.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (high_temperature_sample, low_temperature_sample)
        """
        return self.high_temp_samples[idx], self.low_temp_samples[idx]
    
    @property
    def temperature_pair(self) -> Tuple[float, float]:
        """Return the (T_high, T_low) temperature pair."""
        return (self.temperatures[self.temp_high_idx].item(), 
                self.temperatures[self.temp_low_idx].item())
    
    def get_metadata(self) -> dict:
        """Return dataset metadata."""
        return {
            'pt_data_path': str(self.pt_data_path),
            'temperature_pair': self.temperature_pair,
            'temp_indices': (self.temp_low_idx, self.temp_high_idx),
            'n_samples': self.n_samples,
            'burn_in': self.burn_in,
            'subsample_factor': self.subsample_factor,
            'sample_shape': list(self.low_temp_samples.shape),
            'transform_applied': self.apply_coordinate_transform,
            'cache_path': str(self.cache_path) if self.cache_cleaned_data else None,
        }


def create_pt_dataset_for_pair(
    pt_data_path: Union[str, Path],
    temp_pair_idx: int,
    n_samples: Optional[int] = None,
    **kwargs
) -> PTTemperedPairDataset:
    """
    Convenience function to create a PT dataset for a specific temperature pair.
    
    Parameters
    ----------
    pt_data_path : str or Path
        Path to PT simulation data
    temp_pair_idx : int
        Index of the temperature pair (0 for first pair, 1 for second, etc.)
    n_samples : int, optional
        Number of samples to extract
    **kwargs
        Additional arguments passed to PTTemperedPairDataset
        
    Returns
    -------
    PTTemperedPairDataset
        Dataset for the specified temperature pair
    """
    # Temperature pair indices: (0,1), (1,2), (2,3), etc.
    temp_low_idx = temp_pair_idx
    temp_high_idx = temp_pair_idx + 1
    
    return PTTemperedPairDataset(
        pt_data_path=pt_data_path,
        temp_low_idx=temp_low_idx,
        temp_high_idx=temp_high_idx,
        n_samples=n_samples,
        **kwargs
    )


def get_available_temperature_pairs(pt_data_path: Union[str, Path]) -> list:
    """
    Get information about available temperature pairs from PT data.
    
    Parameters
    ----------
    pt_data_path : str or Path
        Path to PT simulation data
        
    Returns
    -------
    list
        List of dicts with temperature pair information
    """
    try:
        pt_data = torch.load(pt_data_path, map_location='cpu')
        temperatures = pt_data['temperatures']
        n_temps = len(temperatures)
        
        pairs = []
        for i in range(n_temps - 1):
            pairs.append({
                'pair_idx': i,
                'temp_indices': (i, i + 1),
                'temperatures': (temperatures[i].item(), temperatures[i + 1].item()),
                'description': f"T={temperatures[i+1].item():.3f} → T={temperatures[i].item():.3f}"
            })
        
        return pairs
    
    except Exception as e:
        logger.error(f"Failed to load PT data from {pt_data_path}: {e}")
        return [] 