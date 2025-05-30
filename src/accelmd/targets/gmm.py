import matplotlib.pyplot as plt
import itertools
from typing import List
import numpy as np
import torch

from src.accelmd.targets.base import TargetDistribution


class GMM(torch.nn.Module, TargetDistribution):
    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1, seed=0,
                 n_test_set_samples=1000, device="cpu"):
        torch.nn.Module.__init__(self)
        TargetDistribution.__init__(self, dim=dim, is_se3=False, n_particles=1)
        self.seed = seed
        torch.manual_seed(self.seed)
        self.n_mixes = n_mixes
        self.n_test_set_samples = n_test_set_samples
        mean = (torch.rand((n_mixes, dim)) - 0.5)*2 * loc_scaling
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling
        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        self.register_buffer("scale_trils", torch.diag_embed(torch.nn.functional.softplus(log_var)))
        self.device = device
        self.to(self.device)
        self.all_metric_plots = {
            "marginal_pair": lambda samples, label, **kwargs: plt.scatter(
                samples[:, 0].detach().cpu(),
                samples[:, 1].detach().cpu(),
                label=label, **kwargs
            )
        }

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()
    
    @property
    def distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs.to(self.device))
        com = torch.distributions.MultivariateNormal(self.locs.to(self.device),
                                                     scale_tril=self.scale_trils.to(self.device),
                                                     validate_args=False)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com,
                                                     validate_args=False)
    
    def tempered_version(self, temperature=1.0, scaling_method='sqrt'):
        """
        Create a tempered version of this GMM distribution at the specified temperature.
        
        Args:
            temperature (float): Temperature for scaling (default: 1.0)
            scaling_method (str): Method for scaling covariance with temperature
                                 'sqrt' - Scale by sqrt(temperature) (more theoretically correct for GMMs)
                                 'linear' - Scale directly by temperature
        
        Returns:
            A PyTorch distribution representing the tempered GMM
        """
        with torch.no_grad():
            # Scale the covariance according to the specified method
            if scaling_method == 'sqrt':
                # Sqrt scaling is more theoretically correct for GMMs
                scaled_scale_trils = torch.sqrt(torch.tensor(temperature)) * self.scale_trils
            elif scaling_method == 'linear':
                # Linear scaling directly scales variance by temperature
                scaled_scale_trils = torch.tensor(temperature) * self.scale_trils
            else:
                print(f"Warning: Unknown scaling method '{scaling_method}', defaulting to 'sqrt'")
                scaled_scale_trils = torch.sqrt(torch.tensor(temperature)) * self.scale_trils
            
            # Create the tempered distribution
            mix = torch.distributions.Categorical(self.cat_probs.to(self.device))
            comp = torch.distributions.MultivariateNormal(
                loc=self.locs.to(self.device),
                scale_tril=scaled_scale_trils.to(self.device),
                validate_args=False
            )
            return torch.distributions.MixtureSameFamily(
                mixture_distribution=mix,
                component_distribution=comp,
                validate_args=False
            )
    
    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples, ))
    
    def log_prob(self, x: torch.Tensor):
        log_prob = self.distribution.log_prob(x)
        mask = torch.zeros_like(log_prob)
        mask[log_prob < -1e9] = - torch.tensor(float("inf"))
        log_prob = log_prob + mask
        return log_prob
    
    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

    def plot_samples(self, samples_list: List[torch.Tensor], labels_list: List[str], metric_to_plot="marginal_pair", **kwargs):
        for label, samples in zip(labels_list, samples_list):
            if samples is None:
                continue
            self.all_metric_plots[metric_to_plot](samples, label, **kwargs)



def setup_quadratic_function(x: torch.Tensor, seed: int = 0):
    # Useful for porting this problem to non torch libraries.
    torch.random.manual_seed(seed)
    # example function that we may want to calculate expectations over
    x_shift = 2 * torch.randn(x.shape[-1])
    A = 2 * torch.rand((x.shape[-1], x.shape[-1])).to(x.device)
    b = torch.rand(x.shape[-1]).to(x.device)
    torch.seed()  # set back to random number
    if x.dtype == torch.float64:
        return x_shift.double(), A.double(), b.double()
    else:
        assert x.dtype == torch.float32
        return x_shift, A, b


def quadratic_function(x: torch.Tensor, seed: int = 0):
    x_shift, A, b = setup_quadratic_function(x, seed)
    x = x + x_shift
    return torch.einsum("bi,ij,bj->b", x, A, x) + torch.einsum("i,bi->b", b, x)


def MC_estimate_true_expectation(samples, expectation_function):
    f_samples = expectation_function(samples)
    return torch.mean(f_samples)


def relative_mae(true_expectation, est_expectation):
    return torch.abs((est_expectation - true_expectation) / true_expectation)


def plot_contours(log_prob_func,
                  samples = None,
                  ax = None,
                  bounds = (-5.0, 5.0),
                  grid_width_n_points = 20,
                  n_contour_levels = None,
                  log_prob_min = -1000.0,
                  device='cpu',
                  plot_marginal_dims=[0, 1],
                  s=2,
                  alpha=0.6,
                  plt_show=True,
                  title=None):
    """Plot contours of a log_prob_func that is defined on 2D or higher.
    
    For dimensions > 2, only the dimensions specified in plot_marginal_dims
    are visualized (default: first two dimensions).
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    
    # For visualization, we need to create points in a 2D grid
    # But for evaluation, we might need to embed these in higher dimensions
    dim1, dim2 = plot_marginal_dims
    
    # Get the dimensionality from the samples if provided
    full_dim = 2  # Default if no samples
    if samples is not None:
        full_dim = samples.shape[1]
    
    # Create grid points in the visualization dimensions
    grid_coords = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)), device=device)
    
    # If we're dealing with higher dimensions, create full-dimensional points
    if full_dim > 2:
        # Create zero tensor of shape [grid_size², full_dim]
        x_points = torch.zeros((grid_coords.shape[0], full_dim), device=device)
        # Place the grid coordinates in the appropriate dimensions
        x_points[:, dim1] = grid_coords[:, 0]
        x_points[:, dim2] = grid_coords[:, 1]
    else:
        # Standard 2D case
        x_points = grid_coords
    
    log_p_x = log_prob_func(x_points).cpu().detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    
    # Reshape grid coordinates for plotting
    x_points_dim1 = grid_coords[:, 0].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()
    x_points_dim2 = grid_coords[:, 1].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()
    
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)
    if title is not None:
        ax.set_title(title)
    if samples is not None:
        samples_np = samples.cpu().numpy() if hasattr(samples, 'cpu') else samples
        samples_np = np.clip(samples_np, bounds[0], bounds[1])
        ax.scatter(samples_np[:, plot_marginal_dims[0]], samples_np[:, plot_marginal_dims[1]], s=s, alpha=alpha)
    if plt_show:
        plt.show()