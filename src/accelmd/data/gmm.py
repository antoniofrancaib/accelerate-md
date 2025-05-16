"""
Dataset classes for tempered GMM data for temperature transition models.
"""

import torch
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TemperedGMMPairDataset(Dataset):
    """
    Dataset of GMM sample pairs from high temperature to low temperature.
    
    Args:
        gmm (GMM): Base GMM distribution (T=1)
        n_samples (int): Number of samples to generate
        temp_high (float): High temperature
        temp_low (float): Low temperature (typically 1.0)
        temp_scaling_method (str): Method for temperature scaling ('sqrt' or 'linear')
    """
    def __init__(self, gmm, n_samples, temp_high=10.0, temp_low=1.0, temp_scaling_method='sqrt'):
        """Initialize the dataset with high-temperature and low-temperature samples."""
        self.temp_high = temp_high
        self.temp_low = temp_low
        self.temp_scaling_method = temp_scaling_method
        
        # Create a high-temperature version of the GMM using the tempered_version method
        hi_temp_gmm = gmm.tempered_version(temp_high, temp_scaling_method)
        
        # Sample from both distributions
        with torch.no_grad():
            self.hi_temp_samples = hi_temp_gmm.sample((n_samples,)).float()
            self.low_temp_samples = gmm.sample((n_samples,)).float()
        
        logger.info(f"Created dataset with {n_samples} sample pairs from T={temp_high:.4f} to T={temp_low:.4f} using '{self.temp_scaling_method}' scaling")
    
    def __len__(self):
        """Return the number of sample pairs in the dataset."""
        return len(self.hi_temp_samples)
    
    def __getitem__(self, idx):
        """Return a (high_temp, low_temp) sample pair at the given index."""
        return self.hi_temp_samples[idx], self.low_temp_samples[idx]


def create_tempered_gmm(gmm, temp, temp_scaling_method='sqrt'):
    """
    Create a tempered version of a GMM at the specified temperature.
    
    Args:
        gmm: The base GMM distribution (T=1)
        temp: Temperature for the tempered GMM
        temp_scaling_method: Method for temperature scaling ('sqrt' or 'linear')
        
    Returns:
        A tempered GMM distribution
    """
    # Simply use the tempered_version method from the GMM class
    return gmm.tempered_version(temp, temp_scaling_method) 