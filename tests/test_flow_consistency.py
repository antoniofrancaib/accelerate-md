import torch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.models.flow import TempTransitionFlow

def test_forward_inverse_identity():
    """Test that forward followed by inverse approximately recovers the input."""
    # Use hidden_dim=128 to satisfy in_channels % head_channels == 0
    # (where head_channels default to 64)
    flow = TempTransitionFlow(hidden_dim=128, num_blocks=1).eval()
    x = torch.randn(10, 2)
    z, _ = flow.forward(x)
    x_rec = flow.inverse(z)
    # Small reconstruction error on randomly-initialised flow 
    # (should be exact if flow is bijective)
    assert torch.allclose(x, x_rec, rtol=1e-4, atol=1e-4) 