import torch
import sys
import os
import math

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.models.flow import TempTransitionFlow
from scripts.train_flow import compute_loss
from main.targets.gmm import GMM

def test_loss_computed_correctly():
    """Test that the loss function computes a valid negative log likelihood."""
    # Create a flow model
    flow = TempTransitionFlow(hidden_dim=128, num_blocks=1)
    
    # Create random input
    torch.manual_seed(42)
    x = torch.randn(10, 2)
    
    # Dummy GMM and temperature (temp=1 ⇒ no scaling)
    gmm = GMM(dim=2, n_mixes=3, loc_scaling=2.0)
    temp = 1.0
    
    # Compute loss with our function
    device = torch.device('cpu')
    loss = compute_loss(flow, x, gmm, temp, device)
    
    # Check loss is finite and scalar
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.numel() == 1, f"Loss should be a scalar, got shape {loss.shape}"

def test_loss_computation():
    """Test that the loss is computed correctly per the formula in the task."""
    # Create a flow model with simpler architecture
    flow = TempTransitionFlow(hidden_dim=128, num_blocks=1, layers_per_block=1)
    flow.eval()  # Use eval mode for deterministic behavior
    
    # Create random data
    torch.manual_seed(42)
    x = torch.randn(10, 2)
    
    # Dummy GMM and temperature (temp=1 ⇒ no scaling)
    gmm = GMM(dim=2, n_mixes=3, loc_scaling=2.0)
    temp = 1.0
    
    # Use CPU device
    device = torch.device('cpu')
    
    # Compute loss
    loss = compute_loss(flow, x, gmm, temp, device)
    
    # The loss should be a scalar tensor with a finite value
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert torch.isfinite(loss)
    
    # Verify the different components of the loss
    with torch.enable_grad():
        # Enable grad to make sure we can compute gradients if needed
        x = x.detach().requires_grad_(True)
        
        # Get the components separately
        y, logdet = flow.inverse_and_logdet(x)
        
        # Compute base distribution log prob (standard normal)
        expected_loss = -(gmm.log_prob(y) / temp + logdet).mean()
        
        # The loss should match our manual computation
        assert torch.allclose(loss, expected_loss), f"Expected {expected_loss.item()}, got {loss.item()}"
        
        # The loss should require gradients
        assert loss.requires_grad

def test_loss_positive():
    """Test that the loss function returns a positive value."""
    flow = TempTransitionFlow(hidden_dim=128, num_blocks=1)  # Make sure hidden_dim is divisible by head_channels (64)
    x_src = torch.randn(64, 2)
    gmm = GMM(dim=2, n_mixes=3, loc_scaling=2.0)
    temp = 1.0
    loss = compute_loss(flow, x_src, gmm, temp, torch.device('cpu'))
    
    # Loss should be finite
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}" 