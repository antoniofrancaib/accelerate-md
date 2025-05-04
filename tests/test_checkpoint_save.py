import torch
import tempfile
import os
import shutil
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.models.flow import TempTransitionFlow

def test_checkpoint(tmp_path):
    """Test saving and loading model checkpoints."""
    # Create a model and save it
    # Use hidden_dim=128 to satisfy in_channels % head_channels == 0
    # (where head_channels default to 64)
    model = TempTransitionFlow(hidden_dim=128, num_blocks=1)
    pth = tmp_path / "flow0.pth"
    torch.save(model.state_dict(), pth)
    
    # Verify the file exists
    assert pth.exists()
    
    # Create a new model and load the saved weights
    model2 = TempTransitionFlow(hidden_dim=128, num_blocks=1)
    model2.load_state_dict(torch.load(pth))
    
    # Compare weights between the two models
    for (n1, w1), (n2, w2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(w1, w2) 