import torch
import yaml
from PIL import Image
import os
import mdtraj
from pathlib import Path
from typing import List

from src.accelmd.targets.base import TargetDistribution
from src.accelmd.targets.aldp.boltzmann import AldpBoltzmann as Boltzmann
from src.accelmd.utils.aldp_utils import evaluate_aldp, filter_chirality
from src.accelmd.utils.se3_utils import remove_mean


class AldpPotentialCart(Boltzmann, TargetDistribution):
    def __init__(self, data_path=None, temperature=1000, energy_cut=1.e+8,
                 energy_max=1.e+20, n_threads=4, transform='cartesian',
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2},
                 env='vacuum', device='cpu'):
        Boltzmann.__init__(
            self,
            data_path=data_path,
            temperature=temperature,
            energy_cut=energy_cut,
            energy_max=energy_max,
            n_threads=n_threads,
            transform=transform,
            ind_circ_dih=ind_circ_dih,
            shift_dih=shift_dih,
            shift_dih_params=shift_dih_params,
            default_std=default_std,
            env=env
        )
        TargetDistribution.__init__(self, dim=66, is_se3=True, n_particles=22)
        self.dim = 66
        self.device = device
        self.to(device)
        self.is_molecule = True
        
        # Use train.pt directly - position_min_energy.pt was already loaded above for the transforms
        self.min_energy_pos_path = data_path  # Store for later use
        
        # Load from train.pt instead of train.h5
        train_data = torch.load(os.path.join('datasets', 'aldp', 'train.pt')).to(device)
        
        # If we have internal coordinates from train.pt, transform to cartesian
        if train_data.shape[-1] != self.dim:
            # Transform from internal to cartesian coordinates
            cartesian_data, _ = self.coordinate_transform.forward(train_data)
        else:
            cartesian_data = train_data
            
        self.data = cartesian_data
        self.data = remove_mean(self.data, n_particles=22, n_dimensions=3)
        
        self.bonds = [
            (0, 1), (1, 2), (1, 3), (1, 4),
            (4, 5), (4, 6), (6, 7), (6, 8),
            (8, 9), (8, 10), (10, 11), (10, 12),
            (10, 13), (8, 14), (14, 15), (14, 16),
            (16, 17), (16, 18), (18, 19), (18, 20),
            (18, 21)
        ]
        
        # Setup structure and chemical elements (using hardcoded values since we don't need mdtraj)
        structure_elements = ['C', 
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
                              'HB3']
        chemical_elements = ['H', 'C', 'N', 'O']
        
        # Create dummy atom types (22 atoms)
        atom_structure_types = torch.zeros(22, dtype=torch.long, device=device)
        atom_chemical_types = torch.zeros(22, dtype=torch.long, device=device)
        self.atom_structure_types = atom_structure_types
        self.atom_chemical_types = atom_chemical_types
    
    def get_min_energy_position(self):
        x_init = torch.load(self.min_energy_pos_path, map_location=self.device)
        return x_init
    
    def log_prob(self, x):
        if x.ndim == 2:
            log_prob = super().log_prob(x)
        else:
            x_shape = x.shape
            x_reshaped = x.reshape(-1, self.dim)
            log_prob= super().log_prob(x).log_prob(x_reshaped)
            log_prob = log_prob.reshape(x_shape[:-1])
        return log_prob

    def sample(self, num_samples):
        if isinstance(num_samples, int):
            indices = torch.randint(0, self.data.shape[0], (num_samples,), device=self.device)
        elif isinstance(num_samples, list) or isinstance(num_samples, tuple):
            indices = torch.randint(0, self.data.shape[0], num_samples, device=self.device)
        else:
            raise ValueError
        return self.data[indices].to(self.device).float()
    
    def eval(self, samples: torch.Tensor, true_samples: torch.Tensor, iter: int, metric_dir, plot_dir, batch_size: int):
        samples = self.coordinate_transform.inverse(samples.clone().detach())[0]
        true_samples = self.coordinate_transform.inverse(true_samples.clone().detach())[0]
        evaluate_aldp(samples, true_samples, self.log_prob, self.coordinate_transform, iter, metric_dir, plot_dir, batch_size)

    def plot_samples(self, samples_list: List[torch.Tensor], labels_list: List[str], iter: int, metric_dir, plot_dir, batch_size: int):
        source = samples_list[0]
        target = samples_list[1]
        self.eval(source, target, iter, metric_dir, plot_dir, batch_size)
        marginal_angle = Image.open(os.path.join(plot_dir, 'marginals_%s_%07i.png' % ("angle", iter + 1)))
        marginal_bond = Image.open(os.path.join(plot_dir, 'marginals_%s_%07i.png' % ("bond", iter + 1)))
        marginal_dih = Image.open(os.path.join(plot_dir, 'marginals_%s_%07i.png' % ("dih", iter + 1)))
        phi_psi = Image.open(os.path.join(plot_dir, '%s_%07i.png' % ("phi_psi", iter + 1)))
        ramachandran = Image.open(os.path.join(plot_dir, '%s_%07i.png' % ("ramachandran", iter + 1)))
        images = {
            'marginal_angle': marginal_angle,
            'marginal_bond': marginal_bond,
            'marginal_dih': marginal_dih,
            'phi_psi': phi_psi,
            'ramachandran': ramachandran
        }
        return images
    
    def filter_chirality_cartesian(self, x):
        assert x.shape[-1] == self.dim, "Data should be in Cartesian coordinate"
        internal_x = self.coordinate_transform.inverse(x.clone().detach())[0]
        ind_L = filter_chirality(internal_x)
        filtered_x = self.coordinate_transform.forward(internal_x[ind_L])[0]
        return filtered_x

    def reflect_d_to_l_cartesian(self, x, reflect_ind: int = 0):
        assert x.shape[-1] == self.dim, "Data should be in Cartesian coordinate"
        internal_x = self.coordinate_transform.inverse(x.clone().detach())[0]
        ind_L = filter_chirality(internal_x)
        L_x = x[ind_L]
        D_x = x[~ind_L].view(-1, self.n_particles, self.n_dimensions)
        D_x[..., reflect_ind] *= -1.0  # reflect one axis
        reflected_x = torch.cat([L_x, D_x.reshape(-1, self.dim)], dim=0)
        return reflected_x

def get_aldp_potential(config_path, device):

    with open(config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    # Target distribution
    transform_mode = 'mixed' if not 'transform' in config['system'] \
        else config['system']['transform']
    shift_dih = False if not 'shift_dih' in config['system'] \
        else config['system']['shift_dih']
    env = 'vacuum' if not 'env' in config['system'] \
        else config['system']['env']
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    target = AldpPotentialCart(data_path=config['data']['transform'],
                           temperature=config['system']['temperature'],
                           energy_cut=config['system']['energy_cut'],
                           energy_max=config['system']['energy_max'],
                           n_threads=config['system']['n_threads'],
                           ind_circ_dih=ind_circ_dih,
                           shift_dih=shift_dih,
                           env=env,
                           device=device)
    return target