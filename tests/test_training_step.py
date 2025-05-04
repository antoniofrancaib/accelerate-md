import torch
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.models.flow import TempTransitionFlow

def test_flow_model_basic_operations():
    """Test basic operations of the flow model."""
    # Create a flow model with valid parameters
    model = TempTransitionFlow(hidden_dim=128, num_blocks=1)
    
    # Test forward pass
    x = torch.randn(10, 2)
    z, logdet = model.forward(x)
    assert z.shape == (10, 2), f"Latent shape mismatch: {z.shape}"
    assert logdet.shape == (10,), f"Logdet shape mismatch: {logdet.shape}"
    
    # Test inverse pass
    x_recon = model.inverse(z)
    assert x_recon.shape == (10, 2), f"Reconstructed shape mismatch: {x_recon.shape}"
    
    # Test log probability
    logp = model.log_prob(x)
    assert logp.shape == (10,), f"Log probability shape mismatch: {logp.shape}"
    
    # Test sampling
    samples = model.sample(5)
    assert samples.shape == (5, 2), f"Samples shape mismatch: {samples.shape}" 