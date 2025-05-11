import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add the project root to the path to import TarFlow
# TODO: Replace with proper packaging of tarflow (pip install -e tarflow)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ============================================================
#                   Simple 2-D RealNVP Flow
# ============================================================

class _RealNVPCoupling(nn.Module):
    """An *affine* coupling layer for 2-D RealNVP.

    We assume a *fixed* mask so that exactly one of the two input
    dimensions is transformed while the other is left untouched.
    """

    def __init__(self, mask: torch.Tensor, hidden_dim: int = 128):
        super().__init__()
        self.register_buffer("mask", mask)

        in_dim = mask.numel()  # 2 for our 2-D use-case

        # Simple 2-layer MLPs for scale (s) and translation (t)
        def _net():
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_dim),
            )

        self.s_net = _net()
        self.t_net = _net()

    def forward(self, x: torch.Tensor):
        x_masked = x * self.mask  # part that remains unchanged
        s = self.s_net(x_masked) * (1.0 - self.mask)
        t = self.t_net(x_masked) * (1.0 - self.mask)

        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        logdet = (s * (1.0 - self.mask)).sum(dim=1)
        return y, logdet

    def inverse(self, y: torch.Tensor):
        y_masked = y * self.mask
        s = self.s_net(y_masked) * (1.0 - self.mask)
        t = self.t_net(y_masked) * (1.0 - self.mask)

        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = -(s * (1.0 - self.mask)).sum(dim=1)
        return x, logdet


class RealNVPFlow(nn.Module):
    """A minimal RealNVP implementation tailored to 2-D data.

    Args:
        n_couplings (int): Number of coupling layers (>=2 recommended).
        hidden_dim  (int): Hidden units of the *s* and *t* subnetworks.
    """

    def __init__(self, n_couplings: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.dim = 2  # fixed for current task

        # Alternating binary masks [1,0] and [0,1]
        masks = []
        base_mask = torch.tensor([1.0, 0.0])
        for k in range(n_couplings):
            masks.append(base_mask.clone())
            base_mask = 1.0 - base_mask  # flip bits

        self.couplings = nn.ModuleList([
            _RealNVPCoupling(mask, hidden_dim) for mask in masks
        ])

        # Constant term of the standard normal prior
        self.register_buffer(
            "log_z",
            -0.5 * self.dim * torch.log(torch.tensor(2 * np.pi)),
        )

    # ---------------------------------------------------------------------
    # Forward / inverse
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        z = x
        logdet = torch.zeros(x.shape[0], device=x.device)
        for c in self.couplings:
            z, ld = c.forward(z)
            logdet += ld
        return z, logdet

    def inverse(self, z: torch.Tensor):
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device)
        for c in reversed(self.couplings):
            x, ld = c.inverse(x)
            logdet += ld
        return x, logdet

    # ---------------------------------------------------------------------
    # Convenience wrappers
    # ---------------------------------------------------------------------
    def log_prob(self, x: torch.Tensor):
        z, logdet = self.forward(x)
        log_p_z = -0.5 * (z ** 2).sum(dim=1) + self.log_z
        return log_p_z + logdet

    def sample(self, n: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.inverse(z)
        return x


# ----------------------------------------------------------------------
# Factory helpers
# ----------------------------------------------------------------------

def create_realnvp_flow(config: dict | None = None):
    """Instantiate a :class:`RealNVPFlow` from a (possibly empty) config."""
    if config is None:
        config = {}
    n_couplings = int(config.get("n_couplings", 6))
    hidden_dim = int(config.get("hidden_dim", 128))
    return RealNVPFlow(n_couplings=n_couplings, hidden_dim=hidden_dim) 