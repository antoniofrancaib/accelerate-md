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

from typing import Optional, Dict, Union, Tuple, List, Any

import abc
import yaml

class TargetDistribution(abc.ABC):

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn = None,
                            batch_size: Optional[int] = None) -> Dict:
        """
        Check performance metrics using samples & log weights from the model, as well as it's
        probability density function (if defined).
        Args:
            samples: Samples from the trained model.
            log_w: Log importance weights from the trained model.
            log_q_fn: Log probability density function of the trained model, if defined.
            batch_size: If performance metrics are aggregated over many points that require network
                forward passes, batch_size ensures that the forward passes don't overload GPU
                memory by doing all the points together.

        Returns:
            info: A dictionary of performance measures, specific to the defined
            target_distribution, that evaluate how well the trained model approximates the target.
        """
        raise NotImplementedError


    def sample(self, shape):
        raise NotImplementedError



class AldpBoltzmann(nn.Module, TargetDistribution):
    def __init__(self, data_path=None, temperature=1000, energy_cut=1.e+8,
                 energy_max=1.e+20, n_threads=4, transform='internal',
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2},
                 env='vacuum'):
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
        super(AldpBoltzmann, self).__init__()
        
        # Store parameters for potential subclasses
        self.transform_type = transform
        self.data_path = data_path
        self.temperature = temperature
        self.energy_cut = energy_cut
        self.energy_max = energy_max
        self.n_threads = n_threads
        self.shift_dih = shift_dih
        self.shift_dih_params = shift_dih_params
        self.default_std = default_std
        self.env = env
        self.ind_circ_dih = ind_circ_dih

        # Define molecule parameters
        ndim = 66
        if transform == 'mixed':
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
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
            cart_indices = [6, 8, 9, 10, 14]
        # elif transform == 'internal':
        elif transform == 'internal' or transform == 'cartesian':
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

        if data_path[-2:] == 'h5':
            # Load data for transform
            traj = mdtraj.load(data_path)
            traj.center_coordinates()

            # superpose on the backbone
            ind = traj.top.select("backbone")
            traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

            # Gather the training data into a pytorch Tensor with the right shape
            transform_data = traj.xyz
            n_atoms = transform_data.shape[1]
            n_dim = n_atoms * 3
            transform_data_npy = transform_data.reshape(-1, n_dim)
            transform_data = torch.from_numpy(transform_data_npy.astype("float64"))
        elif data_path[-2:] == 'pt':
            transform_data = torch.load(data_path)
        else:
            raise NotImplementedError('Loading data or this format is not implemented.')

        # Set distribution
        mode = "mixed" if transform == 'mixed' else "internal"
        self.coordinate_transform = bg.flows.CoordinateTransform(transform_data,
                                        ndim, z_matrix, cart_indices, mode=mode,
                                        ind_circ_dih=ind_circ_dih, shift_dih=shift_dih,
                                        shift_dih_params=shift_dih_params,
                                        default_std=default_std)

        # Setup distributions based on the transform type
        if n_threads > 1:
            if transform == 'cartesian':
                self.p = bg.distributions.BoltzmannParallel(system, temperature, 
                                energy_cut=energy_cut, energy_max=energy_max, n_threads=n_threads)
            else:
                self.p = bg.distributions.TransformedBoltzmannParallel(system,
                                temperature, energy_cut=energy_cut, energy_max=energy_max,
                                transform=self.coordinate_transform, n_threads=n_threads)
        else:
            if transform == 'cartesian':
                self.p = bg.distributions.Boltzmann(system, temperature, 
                    energy_cut=energy_cut, energy_max=energy_max)
            else:
                self.p = bg.distributions.TransformedBoltzmann(sim.context,
                                temperature, energy_cut=energy_cut, energy_max=energy_max,
                                transform=self.coordinate_transform)

    def log_prob(self, x: torch.tensor):
        return self.p.log_prob(x)

    def performance_metrics(self, samples, log_w, log_q_fn, batch_size):
        return {}

    def sample(self, shape):
        """Sample from the target distribution.
        
        Args:
            shape: Either an integer or a tuple of integers specifying 
                  the shape of the sample to generate.
        
        Returns:
            torch.Tensor: Samples with shape (*shape, dim) where dim is the
                         dimensionality of the configuration space.
        """
        # Actual implementation handled by potential subclasses
        if hasattr(self, 'data'):
            # If this class has data (AldpPotential or AldpPotentialCart)
            if isinstance(shape, int):
                indices = torch.randint(0, self.data.shape[0], (shape,), device=self.data.device)
            elif isinstance(shape, (list, tuple)):
                indices = torch.randint(0, self.data.shape[0], shape, device=self.data.device)
            else:
                raise ValueError("Shape must be an integer or a tuple of integers")
            return self.data[indices].to(self.data.device).float()
        else:
            # Default implementation for base class - abstract to subclasses
            raise NotImplementedError("This method should be implemented by a subclass")

    def tempered_version(self, temperature=1.0, scaling_method='sqrt'):
        """Return a simple proxy (no variance scaling) for higher temperatures.
        This is a placeholder to ensure compatibility with code expecting
        `.tempered_version`. For rigorous physics, implement proper scaling.
        """
        return self



def get_aldp_target(config: dict, device):
    """Build ALDP target from a config dict under cfg['target'] block."""
    # Expect keys similar to those in original YAML structure
    system_cfg = {
        "temperature": config.get("temperature", 300),
        "energy_cut": config.get("energy_cut", 1.e8),
        "energy_max": config.get("energy_max", 1.e20),
        "n_threads": config.get("n_threads", 4),
        "transform": config.get("transform", "internal"),
        "shift_dih": config.get("shift_dih", False),
        "env": config.get("env", "implicit"),
    }
    data_path = config.get("data_path")

    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    
    # Choose the right implementation based on the transform type
    if system_cfg["transform"] == "cartesian":
        from .cartesian import AldpPotentialCart as TargetClass
    else:
        from .potential import AldpPotential as TargetClass
        
    target = TargetClass(
        data_path=data_path,
        temperature=system_cfg["temperature"],
        energy_cut=system_cfg["energy_cut"],
        energy_max=system_cfg["energy_max"],
        n_threads=system_cfg["n_threads"],
        transform=system_cfg["transform"],
        ind_circ_dih=ind_circ_dih,
        shift_dih=system_cfg["shift_dih"],
        env=system_cfg["env"],
        device=device
    )
    return target.to(device)

# Keep backward compatibility with old signature (config_path: str)
# ---------------------------------------------------------------

_original_get_aldp_target = get_aldp_target  # overwrite below if path str

def get_aldp_target(config_or_path, device):  # type: ignore
    """Dispatch based on type: accepts dict (new) or YAML path (legacy)."""
    if isinstance(config_or_path, dict):
        return _original_get_aldp_target(config_or_path, device)
    # Legacy: treat as path
    import yaml
    with open(config_or_path, 'r') as stream:
        legacy_cfg = yaml.load(stream, yaml.FullLoader)
    return _original_get_aldp_target(legacy_cfg, device)
