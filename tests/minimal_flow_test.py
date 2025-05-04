import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.models.flow import TempTransitionFlow

def simple_loss(flow, x):
    """Simple negative log likelihood loss."""
    z, logdet = flow.forward(x)
    # Base distribution log prob (standard normal)
    log_p_z = -0.5 * torch.sum(z**2, dim=1) - z.shape[1] * 0.5 * torch.log(torch.tensor(2 * torch.pi))
    # Full log prob with change of variables
    log_p_x = log_p_z + logdet
    # Negative log likelihood
    return -log_p_x.mean()

def test_flow_training():
    """Test that a flow model can be trained to improve its loss."""
    print("Creating flow model...")
    flow = TempTransitionFlow(hidden_dim=128, num_blocks=1, layers_per_block=1)
    flow.train()
    
    print("Generating data...")
    torch.manual_seed(42)
    x_train = torch.randn(100, 2)
    
    # Get base loss
    initial_loss = simple_loss(flow, x_train).item()
    print(f"Initial loss: {initial_loss:.6f}")
    
    # Train for a few iterations
    print("Training...")
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.05)
    losses = []
    
    for epoch in range(30):
        optimizer.zero_grad()
        loss = simple_loss(flow, x_train)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        print(f"Epoch {epoch}, Loss: {current_loss:.6f}")
    
    # Final evaluation
    final_loss = simple_loss(flow, x_train).item()
    print(f"Final loss: {final_loss:.6f}")
    
    # Verify loss decreased
    improved = any(losses[i] < losses[i-1] for i in range(1, len(losses)))
    print(f"Did loss improve at any point? {improved}")
    
    # Other tests
    print("\nTesting inverse transform...")
    z, _ = flow.forward(x_train[:5])
    x_recon = flow.inverse(z)
    print(f"Original: {x_train[:2]}")
    print(f"Reconstructed: {x_recon[:2]}")
    
    print("\nTesting log_prob...")
    log_p = flow.log_prob(x_train[:5])
    print(f"Log probs: {log_p}")

if __name__ == "__main__":
    test_flow_training() 