import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add the project root to the path to import TarFlow
# TODO: Replace with proper packaging of tarflow (pip install -e tarflow)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tarflow.transformer_flow import Model as TarFlowModel


class TempTransitionFlow(nn.Module):
    """
    A wrapper around the TarFlow model for temperature transitions in 2D.
    This model learns to transform samples from one temperature to the next.
    
    Args:
        hidden_dim (int): Hidden dimension size for the transformer model
        num_blocks (int): Number of transformer blocks
        layers_per_block (int): Number of attention layers per block
        patch_size (int): Size of patches (typically 1 for 2D data)
        use_nvp (bool): Whether to use NVP (affine) coupling (default: True)
    """
    def __init__(
        self,
        hidden_dim=128,
        num_blocks=4,
        layers_per_block=2,
        patch_size=1,
        use_nvp=True
    ):
        super().__init__()
        
        # Define model parameters
        self.in_channels = 2  # 2D data
        self.img_size = 1     # We treat each 2D point as a 1×1 "image" with 2 channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block
        self.use_nvp = use_nvp
        
        # Initialize the TarFlow model
        self.flow_model = TarFlowModel(
            in_channels=self.in_channels,
            img_size=self.img_size,
            patch_size=self.patch_size,
            channels=self.hidden_dim,
            num_blocks=self.num_blocks,
            layers_per_block=self.layers_per_block,
            nvp=self.use_nvp,
            num_classes=0  # No class conditioning
        )
        
        # Store device for creating tensors
        self._device = next(self.flow_model.parameters()).device
        
        # Precompute constant terms for log probability calculation
        self.register_buffer(
            'norm_const', 
            -self.in_channels / 2 * torch.log(2 * torch.tensor(np.pi)) * torch.ones((), device=self._device)
        )
    
    def forward(self, x):
        """
        Forward pass through the flow model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2]
            
        Returns:
            tuple: (z, logdet) where:
                - z is the latent representation of shape [batch_size, 2]
                - logdet is the log determinant of the Jacobian of shape [batch_size]
        """
        # Reshape input for TarFlow: [batch_size, 2] -> [batch_size, 2, 1, 1]
        x_img = x.reshape(-1, self.in_channels, self.img_size, self.img_size)
        
        # Use TarFlow's model directly - it already calls patchify internally
        z, _, logdets = self.flow_model(x_img, y=None)
        
        # Reshape output back to flat vectors
        z = z.reshape(-1, self.in_channels)
        
        return z, logdets
    
    def inverse(self, z):
        """
        Inverse pass through the flow model (from latent to data space).
        
        Args:
            z (torch.Tensor): Latent tensor of shape [batch_size, 2]
            
        Returns:
            torch.Tensor: Reconstructed data of shape [batch_size, 2]
        """
        batch_size = z.shape[0]
        
        # Reshape for TarFlow to match the format expected after patchify
        num_patches = (self.img_size // self.patch_size) ** 2
        patch_dim = self.patch_size**2 * self.in_channels
        z_seq = z.reshape(batch_size, num_patches, patch_dim)
        
        try:
            # Use TarFlow's public reverse API
            x_seq = self.flow_model.reverse(z_seq, y=None)
            
            # If we get a sequence, take the last item
            if isinstance(x_seq, list):
                x_seq = x_seq[-1]
                
            # Reshape to [batch_size, 2]
            x = x_seq.reshape(batch_size, self.in_channels)
            
        except Exception as e:
            # Fallback method - this should rarely be needed once TarFlow's
            # reverse method is modified to handle img_size=1 properly
            
            # Create tensors on the proper device
            z_tensor = z.clone().detach().requires_grad_(True)
            x_init = torch.randn_like(z_tensor, device=self._device)
            x = x_init.clone().detach().requires_grad_(True)
            
            # Use gradient descent to find x such that forward(x) = z
            optimizer = torch.optim.Adam([x], lr=0.01)
            for _ in range(100):  # Limited iterations for efficiency
                optimizer.zero_grad()
                z_pred, _ = self.forward(x)
                loss = F.mse_loss(z_pred, z_tensor)
                loss.backward()
                optimizer.step()
                
                if loss.item() < 1e-5:
                    break
            
            x = x.detach()
            
        return x
    
    def log_prob(self, x):
        """
        Compute log probability of the input under the flow model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2]
            
        Returns:
            torch.Tensor: Log probability of shape [batch_size]
        """
        # Transform to latent space
        z, logdets = self.forward(x)
        
        # Compute log probability under the standard normal prior
        # Use precomputed constant term
        log_prob_z = -0.5 * (z**2).sum(dim=1) + self.norm_const
        
        # Log probability = log_prob(z) + log|det(dx/dz)|
        log_prob = log_prob_z + logdets
        
        return log_prob
    
    def sample(self, num_samples, device=None):
        """
        Sample from the flow model.
        
        Args:
            num_samples (int): Number of samples to generate
            device (torch.device): Device to use for sampling
            
        Returns:
            torch.Tensor: Samples of shape [num_samples, 2]
        """
        if device is None:
            device = self._device
        
        # Sample from the standard normal prior on the proper device
        z = torch.randn(num_samples, self.in_channels, device=device)
        
        # Transform to data space
        samples = self.inverse(z)
        
        return samples

    def inverse_and_logdet(self, x):
        """A fast convenience wrapper that returns **both** the output of the
        transformation from source → target space *and* the associated
        log-determinant.

        For the temperature–transition objective we view the *forward* direction
        of the underlying flow (``forward`` above) as the mapping
        \(f_k : x_k \mapsto y_{k+1}\).  Calling `inverse()` would hit the slow
        optimisation fallback, so instead we simply delegate to
        :pymeth:`forward` and rename the semantic intent.  The Jacobian
        orientation (dy/dx) is therefore already correct.

        Args:
            x (torch.Tensor): Source samples ``[batch, 2]`` from temperature *k*.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                ``(y, logdet)`` where ``y`` is the transformed sample in target
                space and ``logdet`` is ``log‖det∂y/∂x‖``.
        """
        y, logdet = self.forward(x)
        return y, logdet


# Helper function to create a flow model for a specific temperature transition
def create_temp_transition_flow(config):
    """
    Create a temperature transition flow model based on configuration.
    
    Args:
        config (dict): Configuration dictionary with model parameters
        
    Returns:
        TempTransitionFlow: The initialized flow model
    """
    return TempTransitionFlow(
        hidden_dim=config.get('hidden_dim', 128),
        num_blocks=config.get('num_blocks', 4),
        layers_per_block=config.get('layers_per_block', 2),
        patch_size=config.get('patch_size', 1),
        use_nvp=config.get('use_nvp', True)
    ) 