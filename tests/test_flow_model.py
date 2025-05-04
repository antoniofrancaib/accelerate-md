import torch
import sys
import os
import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.models.flow import TempTransitionFlow

def test_model_creation():
    """Test that we can create models with valid parameters."""
    # Default params should work
    model = TempTransitionFlow()
    assert model is not None
    
    # Custom params that satisfy in_channels % head_channels == 0
    for hidden_dim in [64, 128, 192, 256]:
        model = TempTransitionFlow(hidden_dim=hidden_dim)
        assert model is not None
        
    # Different numbers of blocks and layers
    model = TempTransitionFlow(hidden_dim=128, num_blocks=2, layers_per_block=3)
    assert model is not None

def test_model_invalid_params():
    """Test that model creation fails with invalid parameters."""
    # hidden_dim must be divisible by head_channels (64 by default)
    with pytest.raises(AssertionError):
        TempTransitionFlow(hidden_dim=32)
    
    with pytest.raises(AssertionError):
        TempTransitionFlow(hidden_dim=100) 