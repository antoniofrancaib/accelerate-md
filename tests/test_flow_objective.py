import torch
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main.models.flow import TempTransitionFlow
from main.targets.gmm import GMM
from scripts.train_flow import compute_loss

def test_loss_has_grad():
    """Ensure that compute_loss produces a scalar with non-zero gradients."""
    flow = TempTransitionFlow(hidden_dim=32, num_blocks=1)
    x = torch.randn(8, 2, requires_grad=True)
    gmm = GMM(dim=2, n_mixes=3, loc_scaling=2.0)
    temp = 1.0  # using base temperature keeps maths simple

    loss = compute_loss(flow, x, gmm, temp, torch.device('cpu'))
    loss.backward()

    # Collect gradient magnitudes
    grads = [p.grad.abs().sum().item() for p in flow.parameters() if p.grad is not None]

    # At least one parameter should receive gradient signal
    assert any(g > 0 for g in grads), "No gradients flowing through loss!"
    # Loss should be a scalar tensor
    assert loss.ndim == 0, "Loss should be a scalar tensor" 