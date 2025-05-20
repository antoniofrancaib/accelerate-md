"""
RealNVP normalizing flow implementation for temperature transitions.
"""

import torch
import torch.nn as nn
import numpy as np


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
            net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_dim),
            )
            # Zero-initialize the final layer to make the flow start as identity
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        self.s_net = _net()
        self.t_net = _net()

    def forward(self, x: torch.Tensor):
        x_masked = x * self.mask  # part that remains unchanged
        s = self.s_net(x_masked) * (1.0 - self.mask)
        # Clamp the scale to avoid numerical overflow in exp(s)
        s = torch.clamp(s, min=-5.0, max=5.0)
        t = self.t_net(x_masked) * (1.0 - self.mask)

        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        logdet = (s * (1.0 - self.mask)).sum(dim=1)
        return y, logdet

    def inverse(self, y: torch.Tensor):
        y_masked = y * self.mask
        s = self.s_net(y_masked) * (1.0 - self.mask)
        # Clamp the scale for stability
        s = torch.clamp(s, min=-5.0, max=5.0)
        t = self.t_net(y_masked) * (1.0 - self.mask)

        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = -(s * (1.0 - self.mask)).sum(dim=1)
        return x, logdet


class RealNVPFlow(nn.Module):
    """A minimal RealNVP implementation tailored to 2-D data by default.

    Args:
        n_couplings (int): Number of coupling layers (>=2 recommended).
        hidden_dim  (int): Hidden units of the *s* and *t* subnetworks.
        use_permutation (bool): Whether to use coordinate permutation between blocks.
        dim (int): Dimensionality of input/output (defaults to 2).
    """

    def __init__(self, n_couplings: int = 14, hidden_dim: int = 256, 
                use_permutation: bool = True, dim: int = 2):
        super().__init__()
        self.dim = dim  # Can be configured, defaults to 2
        self.use_permutation = use_permutation

        # Alternating binary masks for arbitrary dimension
        masks = []
        for k in range(n_couplings):
            # Create alternating mask patterns
            # Even indices: first half of dimensions is 1, second half is 0
            # Odd indices: first half of dimensions is 0, second half is 1
            mask = torch.zeros(dim)
            if k % 2 == 0:
                mask[:dim//2] = 1.0
            else:
                mask[dim//2:] = 1.0

            # Add a random boolean to decide whether to flip the mask – helps break symmetry
            if torch.rand(1) < 0.5:
                mask = 1.0 - mask
            masks.append(mask)

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
            if self.use_permutation:
                z = z.flip(1) if self.dim != 2 else torch.stack([z[:, 1], z[:, 0]], dim=1)

        return z, logdet

    def inverse(self, z: torch.Tensor):
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device)
        
        # We need to apply operations in reverse order for the inverse
        n_couplings = len(self.couplings)
        for rev_idx in reversed(range(n_couplings)):
            # --- undo the permutation BEFORE the coupling ---
            if self.use_permutation:
                if self.dim == 2:
                    x = torch.stack([x[:, 1], x[:, 0]], dim=1)  # 2‑D swap
                else:
                    x = x.flip(1)                               # general reverse
            # coupling layer
            c = self.couplings[rev_idx]
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


def create_realnvp_flow(config: dict | None = None):
    """Instantiate a :class:`RealNVPFlow` from a (possibly empty) config."""
    if config is None:
        config = {}
    n_couplings = int(config.get("n_couplings", 14))
    hidden_dim = int(config.get("hidden_dim", 256))
    use_permutation = bool(config.get("use_permutation", True))
    # Get dimension from config, default to 2
    dim = int(config.get("dim", 2))
    
    return RealNVPFlow(
        n_couplings=n_couplings, 
        hidden_dim=hidden_dim, 
        use_permutation=use_permutation,
        dim=dim
    ) 