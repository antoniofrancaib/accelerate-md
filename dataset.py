import os
import os.path as osp
import mdtraj as md
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from typing import List, Dict, Tuple
import pdb
from training.se3_utils import scale

import boltzgen as bg
from openmmtools import testsystems


"""
Data Storage Format:  
data/  
 ├── 2AA/  
 │   ├── raw/
 │      ├── topology/
 │      ├── data/
 │   ├── processed/
 │      ├── topology.pt: dictionary
 │      ├── AA.pt
 │      ├── AA.pt
 │      ├── ......
 ├── 4AA/  
 │   ├── raw/  
 │   ├── processed/
 ...

 Each folder contains .pt files, where each .pt file includes information of one kind of peptide
    1. mdtraj
    2. atom/state info
    3. bond: List


Data():
    x: atom types
    coord: Cartesian coordinate
    edge_index: bonds
    batch: index

TransformedData():
    x: atom types
    coord: Cartesian coordinate
    edge_index: fully connected graph
    edge_mask: bonds
    batch: index
"""


class FullyConnectedGraph(BaseTransform):
    def __call__(self, data: Data):

        if data.edge_index is not None:
            original_edge_index = data.edge_index  # Shape: (2, num_original_edges)

        # Retrieve device from an existing tensor (prefer `x` if present) – fallback to CPU.
        device = data.x.device if hasattr(data, "x") and torch.is_tensor(data.x) else (
            data.coord.device if hasattr(data, "coord") and torch.is_tensor(data.coord) else torch.device("cpu")
        )

        adj = torch.ones((data.num_atoms, data.num_atoms), device=device) - torch.eye(
            data.num_atoms, device=device
        )
        fully_connected_edge_index = adj.nonzero(as_tuple=False).t()  # Shape: (2, num_edges)

        data.edge_index = fully_connected_edge_index

        N = data.num_atoms
        original_edges_flat = original_edge_index[0] * N + original_edge_index[1]
        fully_connected_edges_flat = fully_connected_edge_index[0] * N + fully_connected_edge_index[1]
        if data.edge_index is not None:
            data.edge_mask = torch.isin(fully_connected_edges_flat, original_edges_flat)
        else:
            data.edge_mask = torch.zeros(fully_connected_edges_flat.shape[0], dtype=torch.bool, device=device) 

        return data


class ALDPDataset(InMemoryDataset):

    def __init__(self, root="data", transform=None, pre_transform=None, **kwargs):
        self.peptide = "AL"
        self.edge_index = [
            (0, 1), (1, 2), (1, 3), (1, 4),
            (4, 5), (4, 6), (6, 7), (6, 8),
            (8, 9), (8, 10), (10, 11), (10, 12),
            (10, 13), (8, 14), (14, 15), (14, 16),
            (16, 17), (16, 18), (18, 19), (18, 20),
            (18, 21)
        ]
        self.atom_types = [
                0,  # H1 0
                1,  # CH3 1
                2,  # H2 2
                3,  # H3 3
                4,  # C 4
                5,  # O 5
                6,  # N 6
                7,  # H 7
                8,  # CA 8
                9,  # HA 9
                10,  # CB 10
                11,  # HB1 11
                12,  # HB2 12
                13,  # HB3 13
                4,  # C 14
                5,  # O 15
                6,  # N 16
                7,  # H 17
                4,  # C 18
                0,  # H1 19
                2,  # H2 20
                3   # H3 21
            ]
        self.atom_types = torch.tensor(self.atom_types, dtype=torch.int32).view(-1, 1).long()

        edge_index = torch.tensor(self.edge_index, dtype=torch.int64).t().contiguous()
        self.edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)

        super(ALDPDataset, self).__init__(osp.join(root, self.peptide), transform, pre_transform)

        # Safer loading – restrict to tensors where possible.
        try:
            self.data, self.slices, self.topology = torch.load(
                self.processed_paths[0], weights_only=True, map_location="cpu"
            )
        except Exception:  # fall back for legacy pickle files containing non-tensor objects
            self.data, self.slices, self.topology = torch.load(
                self.processed_paths[0], map_location="cpu"
            )

    def process(self):
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            if path[-2:] == 'h5':
                traj = md.load(path)
                traj.center_coordinates()
                ind = traj.top.select("backbone")
                traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)
                coords = torch.from_numpy(traj.xyz.astype("float32"))
                coords = coords - coords.mean(dim=1, keepdim=True)
                # coords, min_val, max_val = scale(coords)
            else:
                raise NotImplementedError
            data_list = []
            progress_bar = tqdm(range(coords.shape[0]), desc="Processing data")
            for i in progress_bar:
                sample = Data(
                    x=self.atom_types,
                    coord=coords[i],
                    edge_index=self.edge_index,
                    num_atoms=coords[i].shape[0],
                    peptide=self.peptide,
                    # min_val=min_val,
                    # max_val=max_val,
                )

                data_list.append(sample)

            data, slices = self.collate(data_list)
            torch.save((data, slices, traj.topology), processed_path)
    
    @property
    def raw_file_names(self):
        return ['train.h5']

    @property
    def processed_file_names(self):
        return ["processed_data.pt"]

    def len(self):
        return len(self.slices['x']) - 1

    def init_energy_func(self, n_threads: int = 1):
        assert n_threads > 0, "Number of threads should be greater than 0"

        temperature = 300
        energy_cut = 1.0e8
        energy_max = 1.0e20

        system = testsystems.AlanineDipeptideImplicit(constraints=None)
        if n_threads > 1:
            self.p = bg.distributions.BoltzmannParallel(
                system,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                n_threads=n_threads,
            )
        else:
            self.p = bg.distributions.Boltzmann(
                system, temperature, energy_cut=energy_cut, energy_max=energy_max
            )

    @property
    def energy(self):
        return lambda x: -self.p.log_prob(x)

    def log_prob(self, x: torch.Tensor):
        return self.p.log_prob(x)

    def score(self, x: torch.Tensor):
        with torch.enable_grad():
            x.requires_grad_(True)
            logp = self.log_prob(x)
            score = torch.autograd.grad(logp.sum(), x)[0]
            x.requires_grad_(False)
        return score.clone().detach()
    

# Get the train, val, and test set defined in TBG
# ------------------------------------------------------------------
def get_train_peptides() -> List[str]:
    return ['AA', 'AE', 'AF', 'AG', 'AI', 'AK', 'AL', 'AQ', 'AS', 'AW', 
            'AY', 'CA', 'CD', 'CI', 'CL', 'CP', 'CQ', 'CT', 'CY', 'DA', 
            'DC', 'DE', 'DF', 'DG', 'DI', 'DM', 'DN', 'DP', 'DS', 'DV', 
            'DY', 'EA', 'EC', 'ED', 'EE', 'EF', 'EG', 'EI', 'EM', 'EN', 
            'EP', 'EQ', 'ER', 'FC', 'FD', 'FE', 'FG', 'FI', 'FK', 'FL', 
            'FM', 'FN', 'FP', 'FQ', 'FR', 'FT', 'FV', 'FW', 'GA', 'GC', 
            'GE', 'GF', 'GG', 'GH', 'GI', 'GK', 'GL', 'GM', 'GV', 'GW', 
            'GY', 'HA', 'HE', 'HG', 'HL', 'HM', 'HN', 'HW', 'HY', 'IA', 
            'IC', 'IF', 'II', 'IN', 'IP', 'IR', 'IT', 'KF', 'KH', 'KK', 
            'KM', 'KP', 'KT', 'KW', 'KY', 'LC', 'LD', 'LF', 'LG', 'LH', 
            'LI', 'LK', 'LL', 'LN', 'LP', 'LQ', 'LS', 'LT', 'MG', 'MH', 
            'ML', 'MM', 'MP', 'MT', 'NL', 'NM', 'NN', 'NP', 'NR', 'NS', 
            'QA', 'QD', 'QE', 'QH', 'QI', 'QL', 'QN', 'QP', 'QR', 'QS', 
            'QT', 'QV', 'QY', 'RG', 'RI', 'RN', 'RR', 'RS', 'RW', 'SC', 
            'SE', 'SF', 'SG', 'SH', 'SL', 'SN', 'SP', 'SR', 'SS', 'SV', 
            'TC', 'TL', 'TN', 'TQ', 'TV', 'VA', 'VC', 'VD', 'VE', 'VH', 
            'VK', 'VM', 'VN', 'VR', 'VT', 'VW', 'WA', 'WD', 'WE', 'WG', 
            'WK', 'WL', 'WM', 'WN', 'WP', 'WQ', 'WR', 'WS', 'WV', 'WW', 
            'YE', 'YF', 'YG', 'YK', 'YM', 'YN', 'YP', 'YQ', 'YR', 'YT', 
            'YV', 'YY']
        # The topology, i.e. .pdb file, of the following dipeptides are missing
        # ['GS', 'IH', 'QK', 'RM', 'VS', 'VY', 'YC', 'YH']


def get_val_peptides() -> List[str]:
    return ['AV', 'CC', 'CE', 'CF', 'CG', 'CH', 'CM', 'CV', 'DD', 'DQ', 
            'DR', 'DT', 'EH', 'ES', 'FY', 'GD', 'HD', 'HF', 'HH', 'HQ', 
            'HS', 'HV', 'ID', 'IE', 'IL', 'IS', 'IV', 'IW', 'IY', 'KA', 
            'KL', 'KV', 'LA', 'LE', 'LR', 'MD', 'MF', 'MN', 'MQ', 'MR', 
            'MS', 'NA', 'ND', 'NG', 'NH', 'NI', 'NT', 'NV', 'NW', 'QC', 
            'RA', 'RD', 'RE', 'RH', 'RK', 'RP', 'SA', 'SI', 'SK', 'SW', 
            'TG', 'TH', 'TM', 'TP', 'TR', 'TS', 'TW', 'VF', 'VG', 'VI', 
            'VL', 'VP', 'VQ', 'WC', 'WH', 'WT', 'WY', 'YS']

def get_test_peptides() -> List[str]:
    return ['AC', 'AT', 'ET', 'GN', 'GP', 'HT', 'IM', 'KG', 'KQ', 'KS', 
            'LW', 'NF', 'NY', 'RL', 'RV', 'TD']


def get_peptide_hash_dict() -> Tuple:
    peptides = get_train_peptides() + get_val_peptides() + get_test_peptides()
    peptides = sorted(peptides)
    peptide_hash_dict = {}
    for i, peptide in enumerate(peptides):
        peptide_hash_dict[peptide] = i
    return peptide_hash_dict


# A dataset class for any kind of dipeptides, including single or few kinds of them
# ------------------------------------------------------------------
class DipeptideDataset(InMemoryDataset):
    def __init__(self, peptides: List, root="data/2AA", mode="full", transform=None, pre_transform=None):
        assert isinstance(peptides, list), "peptides must be a list"
        self.mode = mode
        self.peptides = peptides
        self.peptide_hash_dict = get_peptide_hash_dict()
        if os.path.exists(os.path.join(root, "processed/topology.pt")):
            try:
                self.topo_dict = torch.load(
                    os.path.join(root, "processed/topology.pt"),
                    weights_only=True,
                    map_location="cpu",
                )
            except Exception:
                self.topo_dict = torch.load(
                    os.path.join(root, "processed/topology.pt"), map_location="cpu"
                )
        else:
            self.topo_dict = {}
        super().__init__(root, transform, pre_transform)

        self.data = None
        self.slices = {}

        for processed_path in self.processed_paths:
            try:
                data, slices = torch.load(processed_path, weights_only=True, map_location="cpu")
            except Exception:
                data, slices = torch.load(processed_path, map_location="cpu")

            # Merge data
            if self.data is None:
                self.data = data
                offsets = {key: 0 for key in data.keys()}
            else:
                for key in data.keys():
                    if key == "edge_index":
                        self.data[key] = torch.cat([self.data[key], data[key]], dim=1)
                    elif isinstance(data[key], list):
                        self.data[key] = self.data[key] + data[key]
                    elif isinstance(data[key], torch.Tensor):
                        self.data[key] = torch.cat([self.data[key], data[key]], dim=0)

            # Merge slices (with offset adjustment)
            for key, slice_ in slices.items():
                if key not in self.slices:
                    self.slices[key] = slice_
                else:
                    self.slices[key] = torch.cat([self.slices[key], slice_[1:] + offsets[key]], dim=0)
                offsets[key] = self.slices[key][-1].item()

    @property
    def processed_file_names(self):
        return [f"processed_{peptide}_data.pt" for peptide in self.peptides]

    @property
    def raw_file_names(self):
        return [f"data/{peptide}.pt" for peptide in self.peptides] + [f"topology/{peptide}-traj-state0.pdb" for peptide in self.peptides]

    def file_paths_for_peptide(self, peptide):
        # This is an internal safe accessor
        return (
            os.path.join(self.raw_dir, "data", f"{peptide}.pt"),
            os.path.join(self.raw_dir, "topology", f"{peptide}-traj-state0.pdb")
        )

    @property
    def atom_dict(self) -> Dict:
        if self.mode == "full":
            return {'C': 0, 'CA': 1, 'CB': 2, 'CD': 3, 'CD1': 4, 'CD2': 5, 'CE': 6, 'CE1': 7, 'CE2': 8, 'CE3': 9, 
                    'CG': 10, 'CG1': 11, 'CG2': 12, 'CH2': 13, 'CZ': 14, 'CZ2': 15, 'CZ3': 16, 'H': 17, 'HA': 18, 
                    'HB': 19, 'HD': 20, 'HD1': 21, 'HD2': 22, 'HE': 23, 'HE1': 24, 'HE2': 25, 'HE3': 26, 'HG': 27, 
                    'HG1': 28, 'HG2': 29, 'HH': 30, 'HH1': 31, 'HH2': 32, 'HZ': 33, 'HZ2': 34, 'HZ3': 35, 'N': 36, 
                    'ND1': 37, 'ND2': 38, 'NE': 39, 'NE1': 40, 'NE2': 41, 'NH1': 42, 'NH2': 43, 'NZ': 44, 'O': 45, 
                    'OD': 46, 'OE': 47, 'OG': 48, 'OG1': 49, 'OH': 50, 'OXT': 51, 'SD': 52, 'SG': 53}
        elif self.mode == "backbone" or self.mode == "element":
            return {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4}
        else:
            raise ValueError("Invalid mode")

    @property
    def amino_dict(self) -> Dict:
        return {
            "ALA": 0,
            "ARG": 1,
            "ASN": 2,
            "ASP": 3,
            "CYS": 4,
            "GLN": 5,
            "GLU": 6,
            "GLY": 7,
            "HIS": 8,
            "ILE": 9,
            "LEU": 10,
            "LYS": 11,
            "MET": 12,
            "PHE": 13,
            "PRO": 14,
            "SER": 15,
            "THR": 16,
            "TRP": 17,
            "TYR": 18,
            "VAL": 19,
        }
    
    @property
    def num_atom_types(self):
        if self.mode == "full":
            return [len(self.atom_dict), len(self.amino_dict), 2]
        elif self.mode == "backbone" or self.mode is None:
            return [len(self.atom_dict)]

    def get_atom_types(self, peptide: str) -> torch.Tensor:
        topology = self.topo_dict[peptide]
        atom_types = []

        if self.mode is None or self.mode == "backbone":
            # using only H, C, N, O, and S as atom types
            for atom_name in topology.atoms:
                atom_types.append(self.atom_dict[atom_name.name[0]])
            
            # specify the backbone structure with "backbone" mode
            if self.mode == "backbone":
                # In backbone mode we keep the original atom type indices (0-4) –
                # no out-of-range remapping.
                backbone_idxs = topology.select("backbone")
                atom_types[backbone_idxs] = [5, 6, 7, 8, 9, 10, 11, 12]
            atom_types = torch.tensor(atom_types).view(-1, 1)

        # using all 54 classes as atom types, as well as the amino types and their orders
        elif self.mode == "full":
            amino_idx = []
            amino_types = []
            for i, amino in enumerate(topology.residues):
                for atom_name in amino.atoms:
                    amino_idx.append(i)
                    amino_types.append(self.amino_dict[amino.name])
                    if atom_name.name[0] == "H" and atom_name.name[-1] in ("1", "2", "3"):
                        if self.amino_dict[amino.name] in (8, 13, 17, 18) and atom_name.name[:2] in (
                            "HE",
                            "HD",
                            "HZ",
                            "HH",
                        ):
                            pass
                        else:
                            atom_name.name = atom_name.name[:-1]
                    if atom_name.name[:2] == "OE" or atom_name.name[:2] == "OD":
                        atom_name.name = atom_name.name[:-1]
                    atom_types.append(self.atom_dict[atom_name.name])
            atom_types = torch.tensor(atom_types).view(-1, 1)
            amino_types = torch.tensor(amino_types).view(-1, 1)
            amino_idx = torch.tensor(amino_idx).view(-1, 1)

            atom_types = torch.cat((atom_types, amino_types, amino_idx), dim=-1)

        else:
            raise NotImplementedError("Mode not implemented")

        return atom_types
    
    def get_bonds(self, peptide: str) -> torch.Tensor:
        bonds = [(bond.atom1.index, bond.atom2.index) for bond in self.topo_dict[peptide].bonds]
        bonds = torch.tensor(bonds, dtype=torch.int32)
        return bonds

    def process(self):
        progress_bar = tqdm(zip(self.peptides, self.processed_paths), desc="Processing data")
        for peptide, processed_path in progress_bar:
            data_path, topo_path = self.file_paths_for_peptide(peptide)

            try:
                data = torch.load(data_path, weights_only=True, map_location="cpu")
            except Exception:
                data = torch.load(data_path, map_location="cpu")
            
            if peptide not in self.topo_dict.keys():
                self.topo_dict[peptide] = md.load(topo_path).topology
            
            data_list = []
            sub_progress_bar = tqdm(range(len(data)), desc=f"Processing {peptide}")
            x = self.get_atom_types(peptide)
            bonds = self.get_bonds(peptide)
            edge_index = torch.tensor(bonds, dtype=torch.int64).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
            num_atoms = len(list(self.topo_dict[peptide].atoms))
            coords = data.view(-1, num_atoms, 3)
            for i in sub_progress_bar:
                pep_data = Data(x=x.clone(),
                                coord=coords[i],
                                edge_index=edge_index,
                                num_atoms=num_atoms,
                                peptide=self.peptide_hash_dict[peptide])
                data_list.append(pep_data)

            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)
        torch.save(self.topo_dict, os.path.join(self.root, "processed/topology.pt"))

    def len(self):
        return self.slices["x"].shape[0] - 1


if __name__ == "__main__":
    # dataset = ALDPDataset(root="data")
    # print(dataset[0])
    # print(Batch.from_data_list([dataset[0], dataset[1]]))
    # print(len(dataset))

    dataset = DipeptideDataset(root="../data/2AA", peptides=["AL", "AA"], transform=FullyConnectedGraph())
    dataset2 = DipeptideDataset(root="../data/2AA", peptides=["AL"], transform=FullyConnectedGraph())
