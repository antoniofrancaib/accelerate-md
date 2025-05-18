import torch
import torch.nn as nn
import numpy as np

class RealNVPLoss(nn.Module):
    """Negative log-likelihood for RealNVP (standard normal prior)."""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, sldj: torch.Tensor):
        # Prior log-probability of z under N(0,I)
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        # Sum over dimensions and add log-det-Jacobian
        ll = prior_ll.view(z.size(0), -1).sum(-1) + sldj
        # Return mean NLL
        return -ll.mean()
