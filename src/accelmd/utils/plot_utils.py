import mdtraj as md
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import nglview as nv
from scipy.stats import gaussian_kde


EPSILON_ = 1e-5


def plot_Ramachandran(ax, phi_data: np.ndarray, psi_data: np.ndarray):
    """
    Plot Ramachandran diagram of phi/psi dihedral angles.
    
    Args:
        ax: Matplotlib axes object
        phi_data: Phi dihedral angles in radians
        psi_data: Psi dihedral angles in radians
    
    Returns:
        Modified axes object
    """
    if phi_data.shape != psi_data.shape:
        raise ValueError("phi_data and psi_data must have the same shape")
    
    ax.hist2d(
        phi_data.flatten(),
        psi_data.flatten(),
        bins=100,
        norm=LogNorm(vmin=0.0001, vmax=1.0),
        range=[[-np.pi, np.pi], [-np.pi, np.pi]],
        density=True
    )
    ax.set_xticks(np.arange(-np.pi, np.pi + np.pi/2, step=(np.pi/2)))
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks(np.arange(-np.pi, np.pi + np.pi/2, step=(np.pi/2)))
    ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_xlabel('$\\phi$', fontsize=24)


def plot_phi_psi(ax, phi_data: np.ndarray, psi_data: np.ndarray,
                 phi_test: np.ndarray, psi_test: np.ndarray, nbins: int = 200):
    def filter_nan(x, y):
        is_nan = np.logical_or(np.isnan(x), np.isnan(y))
        not_nan = np.logical_not(is_nan)
        x, y = x[not_nan], y[not_nan]
        return x, y

    phi_data, psi_data = filter_nan(phi_data, psi_data)
    phi_test, psi_test = filter_nan(phi_test, psi_test)

    htest_phi, _ = np.histogram(phi_test.flatten(), nbins, range=[-np.pi, np.pi], density=True)
    hdata_phi, _ = np.histogram(phi_data.flatten(), nbins, range=[-np.pi, np.pi], density=True)
    htest_psi, _ = np.histogram(psi_test.flatten(), nbins, range=[-np.pi, np.pi], density=True)
    hdata_psi, _ = np.histogram(psi_data.flatten(), nbins, range=[-np.pi, np.pi], density=True)
   
def plot_energy_hist(ax, energy_data: np.ndarray, energy_test: np.ndarray, 
                     weights: np.ndarray, nbins: int = 200):
    """Plot histogram comparison of energy distributions."""
    range_limits = (energy_test.min() - 10,energy_test.max() + 100)
    ax.hist(energy_test, bins=100, alpha=0.5, range=range_limits, density=True, label="MD")
    ax.hist(energy_data, bins=100, alpha=0.5, range=range_limits, density=True, label="Model")
    ax.hist(energy_data, bins=100, alpha=0.5, range=range_limits, density=True, label="Model-reweighted", weights=weights, histtype='step', linewidth=5)
    ax.set_xlabel("Energy  / $k_B T$", fontsize=45)
    ax.set_ylabel('Density', fontsize=30)
    ax.legend(fontsize=30)
    return ax
    ax[1].set_ylabel('$p(\psi)$', fontsize=24)
    ax[1].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'])

    return ax


def plot_energy_hist(ax, energy_data: np.ndarray, energy_test: np.ndarray, 
                     is_weights: np.ndarray, nbins: int = 200):
    range_limits = (energy_test.min() - 10,energy_test.max() + 100)
    ax.hist(energy_test, bins=100, alpha=0.5, range=range_limits, density=True, label="MD")
    ax.hist(energy_data, bins=100, alpha=0.5, range=range_limits, density=True, label="Model")
    ax.hist(energy_data, bins=100, alpha=0.5, range=range_limits, density=True, label="Model-reweighed", weights=is_weights, histtype='step', linewidth=5)
    ax.set_xlabel("Energy  / $k_B T$", fontsize=45)
    ax.set_ylabel('Density', fontsize=30)
    # ax.set_xticks(fontsize=25)
    # ax.set_yticks(fontsize=25)
    ax.legend(fontsize=30)
    return ax


def show_md_traj(md_traj: md.Trajectory, scaling: float = 30.):
    md_traj.xyz /= scaling
    view = nv.show_mdtraj(md_traj)
    return view


def free_energy_proj(samples: np.ndarray, weights=None, kBT: float = 1.0, bw_method: float = 0.18):
    """
    Compute the free energy projection using kernel density estimation."
    """
    grid = np.linspace(samples.min(), samples.max(), 100)
    fes = -kBT * gaussian_kde(samples, bw_method, weights).logpdf(grid)
    fes -= fes.min()
    return grid, fes


def plot_free_energy_projection(ax, angles: np.ndarray, log_w: Optional[np.ndarray] = None, **kwargs):
    """
    Compute and plot the free energy curves for different transformations and weightings.
    """
    # Generate transformed phi values for left and right wrapping.
    phi_right = angles.copy().flatten()
    phi_left = angles.copy().flatten()
    phi_right[angles < 0] += 2 * np.pi
    phi_left[angles > np.pi / 2] -= 2 * np.pi
    weights = np.exp(log_w) + EPSILON_ if log_w is not None else None  # EPSILON_ is added to avoid numerical issues in the KDE estimation.
    grid_left, fes_left = free_energy_proj(phi_left, weights=weights)
    grid_right, fes_right = free_energy_proj(phi_right, weights=weights)

    # Extract relevant portions of grid and free energy based on a middle cutoff.
    middle = 0
    idx_left = (grid_left >= -np.pi) & (grid_left < middle)
    idx_right = (grid_right <= np.pi) & (grid_right > middle)
    grid_left, fes_left  = grid_left[idx_left], fes_left[idx_left]
    grid_right, fes_right = grid_right[idx_right], fes_right[idx_right]
    
    ax.plot(np.hstack([grid_left, grid_right]), np.hstack([fes_left, fes_right]), **kwargs)
    return ax


def plot_phi_psi(ax, phi_data: np.ndarray, psi_data: np.ndarray,
                 phi_test: np.ndarray, psi_test: np.ndarray, nbins: int = 200):
    def filter_nan(x, y):
        is_nan = np.logical_or(np.isnan(x), np.isnan(y))
        not_nan = np.logical_not(is_nan)
        x, y = x[not_nan], y[not_nan]
        return x, y

    phi_data, psi_data = filter_nan(phi_data, psi_data)
    phi_test, psi_test = filter_nan(phi_test, psi_test)

    htest_phi, _ = np.histogram(phi_test.flatten(), nbins, range=[-np.pi, np.pi], density=True)
    hdata_phi, _ = np.histogram(phi_data.flatten(), nbins, range=[-np.pi, np.pi], density=True)
    htest_psi, _ = np.histogram(psi_test.flatten(), nbins, range=[-np.pi, np.pi], density=True)
    hdata_psi, _ = np.histogram(psi_data.flatten(), nbins, range=[-np.pi, np.pi], density=True)
   
    x = np.linspace(-np.pi, np.pi, nbins)
    ax[0].plot(x, htest_phi, linewidth=3)
    ax[0].plot(x, hdata_phi, linewidth=3)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].set_xlabel('$\phi$', fontsize=24)
    ax[0].set_ylabel('$p(\phi)$', fontsize=24)
    ax[0].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'])

    ax[1].plot(x, htest_psi, linewidth=3)
    ax[1].plot(x, hdata_psi, linewidth=3)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].set_xlabel('$\psi$', fontsize=24)
    ax[1].set_ylabel('$p(\psi)$', fontsize=24)
    ax[1].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'])

    return ax


def plot_energy_hist(ax, energy_data: np.ndarray, energy_test: np.ndarray, 
                     is_weights: np.ndarray, nbins: int = 200):
    range_limits = (energy_test.min() - 10,energy_test.max() + 100)
    ax.hist(energy_test, bins=100, alpha=0.5, range=range_limits, density=True, label="MD")
    ax.hist(energy_data, bins=100, alpha=0.5, range=range_limits, density=True, label="Model")
    ax.hist(energy_data, bins=100, alpha=0.5, range=range_limits, density=True, label="Model-reweighed", weights=is_weights, histtype='step', linewidth=5)
    ax.set_xlabel("Energy  / $k_B T$", fontsize=45)
    ax.set_ylabel('Density', fontsize=30)
    # ax.set_xticks(fontsize=25)
    # ax.set_yticks(fontsize=25)
    ax.legend(fontsize=30)
    return ax


def show_md_traj(md_traj: md.Trajectory, scaling: float = 30.):
    md_traj.xyz /= scaling
    view = nv.show_mdtraj(md_traj)
    return view


def free_energy_proj(samples: np.ndarray, weights=None, kBT: float = 1.0, bw_method: float = 0.18):
    """
    Compute the free energy projection using kernel density estimation."
    """
    grid = np.linspace(samples.min(), samples.max(), 100)
    fes = -kBT * gaussian_kde(samples, bw_method, weights).logpdf(grid)
    fes -= fes.min()
    return grid, fes


def plot_free_energy_projection(ax, angles: np.ndarray, log_w: Optional[np.ndarray] = None, **kwargs):
    """
    Compute and plot the free energy curves for different transformations and weightings.
    """
    # Generate transformed phi values for left and right wrapping.
    phi_right = angles.copy().flatten()
    phi_left = angles.copy().flatten()
    phi_right[angles < 0] += 2 * np.pi
    phi_left[angles > np.pi / 2] -= 2 * np.pi
    weights = np.exp(log_w) + EPSILON_ if log_w is not None else None  # EPSILON_ is added to avoid numerical issues in the KDE estimation.
    grid_left, fes_left = free_energy_proj(phi_left, weights=weights)
    grid_right, fes_right = free_energy_proj(phi_right, weights=weights)

    # Extract relevant portions of grid and free energy based on a middle cutoff.
    middle = 0
    idx_left = (grid_left >= -np.pi) & (grid_left < middle)
    idx_right = (grid_right <= np.pi) & (grid_right > middle)
    grid_left, fes_left  = grid_left[idx_left], fes_left[idx_left]
    grid_right, fes_right = grid_right[idx_right], fes_right[idx_right]
    
    ax.plot(np.hstack([grid_left, grid_right]), np.hstack([fes_left, fes_right]), **kwargs)
    return ax


def plot_phi_psi(ax, phi_data: np.ndarray, psi_data: np.ndarray,
                 phi_test: np.ndarray, psi_test: np.ndarray, nbins: int = 200):
    def filter_nan(x, y):
        is_nan = np.logical_or(np.isnan(x), np.isnan(y))
        not_nan = np.logical_not(is_nan)
        x, y = x[not_nan], y[not_nan]
        return x, y

    phi_data, psi_data = filter_nan(phi_data, psi_data)
    phi_test, psi_test = filter_nan(phi_test, psi_test)

    htest_phi, _ = np.histogram(phi_test.flatten(), nbins, range=[-np.pi, np.pi], density=True)
    hdata_phi, _ = np.histogram(phi_data.flatten(), nbins, range=[-np.pi, np.pi], density=True)
    htest_psi, _ = np.histogram(psi_test.flatten(), nbins, range=[-np.pi, np.pi], density=True)
    hdata_psi, _ = np.histogram(psi_data.flatten(), nbins, range=[-np.pi, np.pi], density=True)
   
    x = np.linspace(-np.pi, np.pi, nbins)
    ax[0].plot(x, htest_phi, linewidth=3)
    ax[0].plot(x, hdata_phi, linewidth=3)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].set_xlabel('$\phi$', fontsize=24)
    ax[0].set_ylabel('$p(\phi)$', fontsize=24)
    ax[0].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'])

    ax[1].plot(x, htest_psi, linewidth=3)
    ax[1].plot(x, hdata_psi, linewidth=3)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].set_xlabel('$\psi$', fontsize=24)
    ax[1].set_ylabel('$p(\psi)$', fontsize=24)
    ax[1].set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'])

    return ax


def plot_energy_hist(ax, energy_data: np.ndarray, energy_test: np.ndarray, 
                     is_weights: np.ndarray, nbins: int = 200):
    range_limits = (energy_test.min() - 10,energy_test.max() + 100)
    ax.hist(energy_test, bins=100, alpha=0.5, range=range_limits, density=True, label="MD")
    ax.hist(energy_data, bins=100, alpha=0.5, range=range_limits, density=True, label="Model")
    ax.hist(energy_data, bins=100, alpha=0.5, range=range_limits, density=True, label="Model-reweighed", weights=is_weights, histtype='step', linewidth=5)
    ax.set_xlabel("Energy  / $k_B T$", fontsize=45)
    ax.set_ylabel('Density', fontsize=30)
    # ax.set_xticks(fontsize=25)
    # ax.set_yticks(fontsize=25)
    ax.legend(fontsize=30)
    return ax


def show_md_traj(md_traj: md.Trajectory, scaling: float = 30.):
    md_traj.xyz /= scaling
    view = nv.show_mdtraj(md_traj)
    return view


def free_energy_proj(samples: np.ndarray, weights=None, kBT: float = 1.0, bw_method: float = 0.18):
    """
    Compute the free energy projection using kernel density estimation."
    """
    grid = np.linspace(samples.min(), samples.max(), 100)
    fes = -kBT * gaussian_kde(samples, bw_method, weights).logpdf(grid)
    fes -= fes.min()
    return grid, fes


def plot_free_energy_projection(ax, angles: np.ndarray, log_w: Optional[np.ndarray] = None, **kwargs):
    """
    Compute and plot the free energy curves for different transformations and weightings.
    """
    # Generate transformed phi values for left and right wrapping.
    phi_right = angles.copy().flatten()
    phi_left = angles.copy().flatten()
    phi_right[angles < 0] += 2 * np.pi
    phi_left[angles > np.pi / 2] -= 2 * np.pi
    weights = np.exp(log_w) + EPSILON_ if log_w is not None else None  # EPSILON_ is added to avoid numerical issues in the KDE estimation.
    grid_left, fes_left = free_energy_proj(phi_left, weights=weights)
    grid_right, fes_right = free_energy_proj(phi_right, weights=weights)

    # Extract relevant portions of grid and free energy based on a middle cutoff.
    middle = 0
    idx_left = (grid_left >= -np.pi) & (grid_left < middle)
    idx_right = (grid_right <= np.pi) & (grid_right > middle)
    grid_left, fes_left  = grid_left[idx_left], fes_left[idx_left]
    grid_right, fes_right = grid_right[idx_right], fes_right[idx_right]
    
    ax.plot(np.hstack([grid_left, grid_right]), np.hstack([fes_left, fes_right]), **kwargs)
    return ax