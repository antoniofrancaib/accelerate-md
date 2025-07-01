import torch
from torch import nn
import numpy as np

import boltzgen as bg
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
import mdtraj
import tempfile

from typing import Optional, Dict

import abc
import yaml

from .base import Distribution
from .utils import torch_to_mdtraj, filter_chirality, compute_phi_psi
from ..utils.se3_utils import remove_mean, interatomic_dist, compute_intersection, compute_correlation
from ..utils.plot_utils import plot_Ramachandran, plot_phi_psi, plot_energy_hist, plot_free_energy_projection
from ..metrics.kld import kl_divergence
from ..metrics.wasserstein import wasserstein_distance_1d


class AlanineDipeptide(Distribution):
    def __init__(
        self,
        data_path=None,
        temperature=1000,
        energy_cut=1.e+8,
        energy_max=1.e+20,
        n_threads=4,
        internal_transform=False,  # 'True' for internal coordinate
        ind_circ_dih=[],
        shift_dih=False,
        shift_dih_params={'hist_bins': 100},
        default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2},
        env='vacuum'
    ):
        """
        Boltzmann distribution of Alanine dipeptide
        :param data_path: Path to the trajectory file used to initialize the
            transformation, if None, a trajectory is generated
        :type data_path: String
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        :param transform: Which transform to use, can be mixed or internal
        :type transform: String
        """

        # Define molecule parameters
        if internal_transform:
            ndim = 60
        else:
            ndim = 66
            self.n_particles = 22
            self.n_dimensions = 3
        self.dim = ndim
        z_matrix = [
            (0, [1, 4, 6]),
            (1, [4, 6, 8]),
            (2, [1, 4, 0]),
            (3, [1, 4, 0]),
            (4, [6, 8, 14]),
            (5, [4, 6, 8]),
            (7, [6, 8, 4]),
            (9, [8, 6, 4]),
            (10, [8, 6, 4]),
            (11, [10, 8, 6]),
            (12, [10, 8, 11]),
            (13, [10, 8, 11]),
            (15, [14, 8, 16]),
            (16, [14, 8, 6]),
            (17, [16, 14, 15]),
            (18, [16, 14, 8]),
            (19, [18, 16, 14]),
            (20, [18, 16, 19]),
            (21, [18, 16, 19])
        ]
        cart_indices = [8, 6, 14]

        # System setup
        if env == 'vacuum':
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == 'implicit':
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError('This environment is not implemented.')
        sim = app.Simulation(system.topology, system.system,
                             mm.LangevinIntegrator(temperature * unit.kelvin,
                                                   1. / unit.picosecond,
                                                   1. * unit.femtosecond),
                             mm.Platform.getPlatformByName('Reference'))

        # Generate trajectory for coordinate transform if no data path is specified
        if data_path is None:
            sim = app.Simulation(system.topology, system.system,
                                 mm.LangevinIntegrator(temperature * unit.kelvin,
                                                       1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                                 platform=mm.Platform.getPlatformByName('Reference'))
            sim.context.setPositions(system.positions)
            sim.minimizeEnergy()
            state = sim.context.getState(getPositions=True)
            position = state.getPositions(True).value_in_unit(unit.nanometer)
            tmp_dir = tempfile.gettempdir()
            data_path = tmp_dir + '/aldp.pt'
            torch.save(torch.tensor(position.reshape(1, 66).astype(np.float64)), data_path)

            del (sim)

        else:
            assert data_path[-2:] == "h5", "You must using the .h5 file"
            # Load data for transform
        traj = mdtraj.load(data_path)
        traj.center_coordinates()

        # superpose on the backbone
        ind = traj.top.select("backbone")
        traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

        # Gather the training data into a pytorch Tensor with the right shape
        data = traj.xyz
        n_atoms = data.shape[1]
        n_dim = n_atoms * 3
        data_npy = data.reshape(-1, n_dim)
        data = torch.from_numpy(data_npy.astype("float32"))

        if internal_transform:
            self.coordinate_transform = bg.flows.CoordinateTransform(
                data,
                ndim,
                z_matrix,
                cart_indices,
                mode="internel",
                ind_circ_dih=ind_circ_dih,
                shift_dih=shift_dih,
                shift_dih_params=shift_dih_params,
                default_std=default_std
            )
            data = self.coordinate_transform.inverse(data)[0]
        else:
            data = remove_mean(data, n_particles=22, n_dimensions=3)

        if n_threads > 1:
            if not internal_transform:
                self.p = bg.distributions.BoltzmannParallel(system, temperature,
                                                            energy_cut=energy_cut, energy_max=energy_max, n_threads=n_threads)
            else:
                self.p = bg.distributions.TransformedBoltzmannParallel(system,
                                                                       temperature, energy_cut=energy_cut, energy_max=energy_max,
                                                                       transform=self.coordinate_transform, n_threads=n_threads)
        else:
            if not internal_transform:
                self.p = bg.distributions.Boltzmann(system, temperature,
                                                    energy_cut=energy_cut, energy_max=energy_max)
            else:
                self.p = bg.distributions.TransformedBoltzmann(sim.context,
                                                               temperature, energy_cut=energy_cut, energy_max=energy_max,
                                                               transform=self.coordinate_transform)

        self.bonds = [
            (0, 1), (1, 2), (1, 3), (1, 4),
            (4, 5), (4, 6), (6, 7), (6, 8),
            (8, 9), (8, 10), (10, 11), (10, 12),
            (10, 13), (8, 14), (14, 15), (14, 16),
            (16, 17), (16, 18), (18, 19), (18, 20),
            (18, 21)
        ]
        structure_elements = [
            'C',
            'O',
            'N',
            'H',
            'H1',
            'H2',
            'H3',
            'CH3',
            'CA',
            'CB',
            'HA',
            'HB1',
            'HB2',
            'HB3'
        ]
        chemical_elements = ['H', 'C', 'N', 'O']
        self.topology = traj.topology
        table, _ = traj.topology.to_dataframe()
        structure_to_index = {element: idx for idx, element in enumerate(structure_elements)}
        chemical_to_index = {element: idx for idx, element in enumerate(chemical_elements)}
        atom_structure_elements = table["name"].values
        atom_chemical_elements = table["element"].values
        atom_structure_types = [structure_to_index[element] for element in atom_structure_elements]
        atom_chemical_types = [chemical_to_index[element] for element in atom_chemical_elements]
        self.atom_structure_types = atom_structure_types
        self.atom_chemical_types = atom_chemical_types
        super().__init__()
        self.has_data = True
        self.register_buffer("data", data)
        self.register_buffer("data_mean", torch.mean(self.data.view(-1, self.dim), dim=0))
        self.register_buffer("data_var", torch.var(self.data.view(-1, self.dim), dim=0))

    def build_dist(self):
        """Builds the inner dist object"""
        pass
    
    def log_prob(self, x: torch.tensor):
        return self.p.log_prob(x)
    
    def filter_chirality(self, samples, ref_samples):
        if self.dim == 60:
            print("Don't need to filter chirality for internal coordinate")
            return samples
        sample_shape = samples.shape[1:]
        samples = samples.view(samples.shape[0], self.n_particles, self.n_dimensions)
        target_traj = torch_to_mdtraj(ref_samples.view(-1, self.n_particles, self.n_dimensions), self.topology)
        samples, D_form_counter = filter_chirality(samples, target_traj)
        print(f"Number of D-form samples are changed from {D_form_counter[0] / samples.shape[0]} to {D_form_counter[1] / samples.shape[0]}")
        return samples.view(-1, *sample_shape)
    
    def compute_psi_phi(self, samples, ref_samples=None, return_ref_angles=False):
        """
        Compute the torsional angles: psi and phi
        """
        if ref_samples is None:
            ref_samples = self.sample((samples.shape[0],))
        traj = torch_to_mdtraj(samples.view(-1, self.n_particles, self.n_dimensions), self.topology)
        phi, psi = compute_phi_psi(traj)
        ref_traj = torch_to_mdtraj(ref_samples.view(-1, self.n_particles, self.n_dimensions), self.topology)
        ref_phi, ref_psi = compute_phi_psi(ref_traj)

        def filter_nan(x, y):
            is_nan = np.logical_or(np.isnan(x), np.isnan(y))
            not_nan = np.logical_not(is_nan)
            x, y = x[not_nan], y[not_nan]
            return torch.tensor(x).view(-1, 1), torch.tensor(y).view(-1, 1)

        phi, ref_phi = filter_nan(phi, ref_phi)
        psi, ref_psi = filter_nan(psi, ref_psi)
        if not return_ref_angles:
            return phi, psi
        else:
            return phi, psi, ref_phi, ref_psi
    
    def compute_ramachandran_kld(self, samples, ref_samples):
        samples = self.filter_chirality(samples, ref_samples)
        phi_source, psi_source, phi_target, psi_target = self.compute_psi_phi(samples, ref_samples, return_ref_angles=True)
        dihedral_source = torch.cat([phi_source, psi_source], dim=1)
        dihedral_target = torch.cat([phi_target, psi_target], dim=1)

        rama_kld = {
            "kld_phi": kl_divergence(phi_source, phi_target, num_bins=200, ranges=[[-np.pi, np.pi]]).item(),
            "kld_psi": kl_divergence(psi_source, psi_target, num_bins=200, ranges=[[-np.pi, np.pi]]).item(),
            "kld_ramachandran": kl_divergence(dihedral_source, dihedral_target, num_bins=64, ranges=[[-np.pi, np.pi], [-np.pi, np.pi]]).item()
        }
        return rama_kld
        
    def compute_metrics(self, samples, weights=None, ref_samples=None, compute_standard_metrics=False,
                        skip_costly_metrics=True, bins=128):
        """Compute various metrics based on samples"""
        # Warn about the weights
        if weights:
            raise ValueError('Weights are not supported for LJ.')
        # Get the standard statistics
        ret = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                      compute_standard_metrics=compute_standard_metrics,
                                      skip_costly_metrics=skip_costly_metrics
                                      )
        # Get reference samples
        if ref_samples is None:
            ref_samples = self.sample((samples.shape[0],)).to(samples.device)
        # Compute the various histograms
        ref_dist_hist, (dist_min, dist_max) = self.compute_interatomic_histograms(ref_samples, bins)
        samples_dist_hist, _ = self.compute_interatomic_histograms(samples, bins, range=(dist_min, dist_max))
        ref_en_hist, (en_min, en_max), en_ref = self.compute_energy_histograms(ref_samples, bins, return_en=True)
        samples_en_hist, _, en_samples = self.compute_energy_histograms(samples, bins,
                                                                        range=(en_min, en_max), return_en=True)
        # Compute the histogram distances
        ret['correlation_dist_hist'] = compute_correlation(ref_dist_hist, samples_dist_hist).item()
        ret['intersection_dist_hist'] = compute_intersection(ref_dist_hist, samples_dist_hist).item()
        ret['correlation_en_hist'] = compute_correlation(ref_en_hist, samples_dist_hist).item()
        ret['intersection_en_hist'] = compute_intersection(samples_en_hist, samples_dist_hist).item()
        # Compute the energy wasserstein distance
        ret['energy_w2'] = wasserstein_distance_1d(en_ref, en_samples).item()

        ret.update(self.compute_ramachandran_kld(samples, ref_samples))
        return ret
    
    def compute_interatomic_histograms(self, x, bins, range=None):
        """Compute the histograms of interatomic distances"""
        x = x.view((-1, self.n_particles, self.n_dimensions))
        dists = interatomic_dist(x).detach().cpu().flatten()
        return torch.histogram(dists, bins=bins, density=True, range=range)[0], (dists.min().item(), dists.max().item())

    def compute_energy_histograms(self, x, bins, range=None, return_en=False):
        """Compute the histograms of energies"""
        ens = -self.log_prob(x).detach().cpu().flatten()
        hist = torch.histogram(ens, bins=bins, density=True, range=range)[0]
        en_min, en_max = ens.min().item(), ens.max().item()
        if return_en:
            return hist, (en_min, en_max), ens
        else:
            return hist, (en_min, en_max)

    def plot_samples(self, ax, samples, label="model", plot_type="ramachandran"):
        ax.set_title(label)
        if plot_type in ["ramachandran", "marginal_angles", "fep_psi", "fep_phi"]:
            ref_samples = self.sample((samples.shape[0],))
            samples = self.filter_chirality(samples, ref_samples)
            phi_source, psi_source, phi_target, psi_target = self.compute_psi_phi(samples, ref_samples, return_ref_angles=True)
            if plot_type == "ramachandran":
                plot_Ramachandran(ax, phi_source, psi_source)
            elif plot_type == "marginal_angles":
                plot_phi_psi(ax, phi_source, psi_source, phi_target, psi_target)
            elif plot_type == "fep_phi":
                plot_free_energy_projection(ax, phi_source)
            elif plot_type == "fep_psi":
                 plot_free_energy_projection(ax, psi_source)
        else:
            bins = 128
            if plot_type == 'energies':
                hist_fn = self.compute_energy_histograms
                xlabel = "Energy"
            else:
                hist_fn = self.compute_interatomic_histograms
                xlabel = "Interatomic Distance"
            # Sample the true distribution
            if self.has_data:
                true_samples = self.sample((samples.shape[0],))
                true_hist, (val_min, val_max) = hist_fn(true_samples, bins)
            model_hist = hist_fn(samples, bins, range=(val_min, val_max))[0]
            hist_pairwise_linespace = torch.linspace(val_min, val_max, bins)
            width = torch.min(torch.diff(hist_pairwise_linespace))
            ax.bar(hist_pairwise_linespace, true_hist, label='True', align='edge', alpha=0.5, width=width)
            ax.bar(hist_pairwise_linespace, model_hist, label=label, align='edge', alpha=0.5, width=width)
            ax.set_ylabel('Density')
            ax.set_xlabel(xlabel)
            ax.legend()