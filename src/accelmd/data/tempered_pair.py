import torch
from torch.utils.data import Dataset

class TemperedPairDataset(Dataset):
    """Generic dataset of (high→low) or (low→high) pairs for any target.

    Parameters
    ----------
    low_target : object
        Target distribution at the *lower* temperature (usually T=1).
    high_target : object
        Target distribution at the *higher* temperature.
    n_samples : int
        Number of (x_high, x_low) pairs to draw.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the low-T samples (default 0).
    """

    def __init__(self, low_target, high_target, n_samples: int, noise_std: float = 0.0):
        self.low_tgt = low_target
        self.high_tgt = high_target
        self.N = int(n_samples)
        self.noise = float(noise_std)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):  # pylint: disable=unused-argument
        # Draw one fresh sample from each distribution
        x_hi = self.high_tgt.sample((1,)).squeeze(0)
        x_lo = self.low_tgt.sample((1,)).squeeze(0)
        if self.noise > 0.0:
            x_lo = x_lo + torch.randn_like(x_lo) * self.noise
        return x_hi, x_lo 