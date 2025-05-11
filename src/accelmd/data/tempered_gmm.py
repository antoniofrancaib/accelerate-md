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
    """
    def __init__(self, gmm, n_samples, temp_high=10.0, temp_low=1.0):
        """Initialize the dataset with high-temperature and low-temperature samples."""
        self.temp_high = temp_high
        self.temp_low = temp_low
        
        # For a proper implementation, we should modify the GMM's variance
        # High temp -> wider variance
        # Low temp -> narrower variance (typically the original GMM)
        with torch.no_grad():
            # Sample from high temp (we'll use scaled variance)
            # This is an approximation - scaling sqrt(temp) is more accurate for GMMs
            # but we need proper tempered sampling which is outside scope
            means = gmm.locs
            hi_scale_trils = torch.sqrt(torch.tensor(temp_high)) * gmm.scale_trils
            
            # Create a temporary high-temp GMM
            mix = torch.distributions.Categorical(gmm.cat_probs)
            comp_hi = torch.distributions.MultivariateNormal(
                loc=means,
                scale_tril=hi_scale_trils,
                validate_args=False
            )
            hi_gmm = torch.distributions.MixtureSameFamily(
                mixture_distribution=mix,
                component_distribution=comp_hi,
                validate_args=False
            )
            
            # Sample from both
            self.hi_temp_samples = hi_gmm.sample((n_samples,)).float()
            self.low_temp_samples = gmm.sample((n_samples,)).float()
        
        logger.info(f"Created dataset with {n_samples} sample pairs from T={temp_high:.4f} to T={temp_low:.4f}")
    
    def __len__(self):
        """Return the number of sample pairs in the dataset."""
        return len(self.hi_temp_samples)
    
    def __getitem__(self, idx):
        """Return a (high_temp, low_temp) sample pair at the given index."""
        return self.hi_temp_samples[idx], self.low_temp_samples[idx]


def create_tempered_gmm(gmm, temp):
    """
    Create a tempered version of a GMM at the specified temperature.
    
    Args:
        gmm: The base GMM distribution (T=1)
        temp: Temperature for the tempered GMM
        
    Returns:
        A tempered GMM distribution
    """
    with torch.no_grad():
        means = gmm.locs
        scaled_scale_trils = torch.sqrt(torch.tensor(temp)) * gmm.scale_trils
        
        mix = torch.distributions.Categorical(gmm.cat_probs)
        comp = torch.distributions.MultivariateNormal(
            loc=means,
            scale_tril=scaled_scale_trils,
            validate_args=False
        )
        tempered_gmm = torch.distributions.MixtureSameFamily(
            mixture_distribution=mix,
            component_distribution=comp,
            validate_args=False
        )
    
    return tempered_gmm 