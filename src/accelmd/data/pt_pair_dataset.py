"""PT temperature pair dataset for swap flow training.

Loads PT trajectory data for a specific temperature pair and provides
batched coordinate pairs for training swap flows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Dataset

from .molecular_data import filter_chirality, center_coordinates, random_rotation_augment

__all__ = ["PTTemperaturePairDataset"]


class PTTemperaturePairDataset(Dataset):
    """Dataset for PT coordinate pairs between adjacent temperatures.
    
    Loads PT simulation data and extracts coordinate pairs for training
    flows that map between adjacent temperature replicas.
    
    Parameters
    ----------
    pt_data_path : str
        Path to the PT trajectory file (e.g., 'datasets/pt_dipeptides/AX/pt_AX.pt')
    molecular_data_path : str  
        Path to molecular data directory containing adj_list.pt and atom_types.pt
    temp_pair : Tuple[int, int]
        Indices of the temperature pair (e.g., (0, 1) for T0 -> T1)
    subsample_rate : int, optional
        Take every Nth frame (default: 100)
    device : str, optional
        Device for tensors, currently kept as "cpu" (default: "cpu")
    filter_chirality : bool, optional
        Whether to filter out incorrect chirality conformations (default: False)
    center_coordinates : bool, optional
        Whether to center coordinates (default: True)
    augment_coordinates : bool, optional
        Whether to apply random rotation augmentation (default: False)
    random_subsample : bool, optional
        Whether to use random subsampling instead of deterministic (default: False)
    random_seed : int, optional
        Random seed for reproducible subsampling (default: None)
    """
    
    def __init__(
        self,
        pt_data_path: str,
        molecular_data_path: str,
        temp_pair: Tuple[int, int],
        subsample_rate: int = 100,
        device: str = "cpu",
        filter_chirality: bool = False,
        center_coordinates: bool = True,
        augment_coordinates: bool = False,
        random_subsample: bool = False,
        random_seed: int = None,
    ) -> None:
        self.pt_data_path = Path(pt_data_path)
        self.molecular_data_path = Path(molecular_data_path)
        self.temp_pair = temp_pair
        self.subsample_rate = subsample_rate
        self.device = device
        self.filter_chirality_enabled = filter_chirality
        self.center_coordinates_enabled = center_coordinates
        self.augment_coordinates_enabled = augment_coordinates
        self.random_subsample = random_subsample
        self.random_seed = random_seed
        
        # Load PT trajectory data and molecular structure
        self._load_pt_data()
        self._load_molecular_data()
        
    def _load_pt_data(self) -> None:
        """Load and preprocess PT trajectory data."""
        if not self.pt_data_path.exists():
            raise FileNotFoundError(f"PT data file not found: {self.pt_data_path}")
            
        # Load the PT data (should be a dict with trajectory info)
        try:
            pt_data = torch.load(self.pt_data_path, map_location="cpu", weights_only=True)
        except Exception:
            # Fallback for non-tensor data
            pt_data = torch.load(self.pt_data_path, map_location="cpu")
        
        # Extract coordinates for our temperature pair
        low_idx, high_idx = self.temp_pair
        
        # PT data should have structure like:
        # {"trajectory": tensor of shape [n_steps, n_temps, n_chains, n_atoms*3],
        #  "temperatures": tensor of temps, etc.}
        if isinstance(pt_data, dict):
            if "trajectory" in pt_data:
                traj = pt_data["trajectory"]  # [n_steps, n_temps, n_chains, n_atoms*3]
            else:
                # Try other common keys
                for key in ["coords", "coordinates", "traj"]:
                    if key in pt_data:
                        traj = pt_data[key]
                        break
                else:
                    raise ValueError(f"Could not find trajectory data in keys: {list(pt_data.keys())}")
        else:
            # Assume pt_data is directly the trajectory tensor
            traj = pt_data
            
        # Apply subsampling to the steps dimension
        if self.subsample_rate > 1:
            if self.random_subsample:
                # Conservative random subsampling: use deterministic subsampling first, then add randomness
                import numpy as np
                if self.random_seed is not None:
                    np.random.seed(self.random_seed)
                
                # First, do deterministic subsampling to get the same total number as before
                if traj.ndim == 4:
                    traj = traj[:, :, ::self.subsample_rate, :]  # subsample steps dimension
                else:
                    traj = traj[::self.subsample_rate]  # fallback for other formats
                
                # Then add randomness by shuffling the selected samples
                if traj.ndim == 4:
                    # Shape: [temps, chains, steps, coords] - shuffle the steps dimension
                    n_steps = traj.shape[2]
                    step_indices = np.arange(n_steps)
                    np.random.shuffle(step_indices)
                    traj = traj[:, :, step_indices, :]
                    print(f"Random shuffle: shuffled {n_steps} pre-selected samples (seed: {self.random_seed})")
                else:
                    # Shape: [steps*chains, temps, coords] - shuffle the samples dimension
                    n_samples = traj.shape[0]
                    sample_indices = np.arange(n_samples)
                    np.random.shuffle(sample_indices)
                    traj = traj[sample_indices]
                    print(f"Random shuffle: shuffled {n_samples} pre-selected samples (seed: {self.random_seed})")
            else:
                # Deterministic subsampling (original behavior)
                if traj.ndim == 4:
                    traj = traj[:, :, ::self.subsample_rate, :]  # subsample steps dimension
                else:
                    traj = traj[::self.subsample_rate]  # fallback for other formats
            
        # Extract coordinates for our temperature pair
        low_idx, high_idx = self.temp_pair
        

        # traj shape should be [n_temps, n_chains, n_steps, n_coords] 
        if traj.ndim == 4:
            # Standard format: [temps, chains, steps, coords]
            if high_idx >= traj.shape[0]:
                raise ValueError(f"Temperature index {high_idx} out of bounds for {traj.shape[0]} temperatures")
            source_coords = traj[low_idx, :, :, :].flatten(0, 1)   # [chains*steps, coords]
            target_coords = traj[high_idx, :, :, :].flatten(0, 1)  # [chains*steps, coords]
        elif traj.ndim == 3:
            # Alternative format: [steps*chains, temps, coords]
            source_coords = traj[:, low_idx, :]  # [steps*chains, coords]
            target_coords = traj[:, high_idx, :]  # [steps*chains, coords]
        else:
            raise ValueError(f"Unexpected trajectory shape: {traj.shape}")
        
        # Reshape from flat [N*3] to [N, 3] format expected by flow model
        n_atoms = source_coords.shape[1] // 3
        source_coords = source_coords.view(-1, n_atoms, 3)
        target_coords = target_coords.view(-1, n_atoms, 3)
        
        # Apply coordinate centering if enabled
        if self.center_coordinates_enabled:
            source_coords = center_coordinates(source_coords)
            target_coords = center_coordinates(target_coords)
        
        # Apply coordinate augmentation if enabled
        # IMPORTANT: Apply the SAME rotation to both source and target coordinates
        if self.augment_coordinates_enabled:
            # Combine coordinates for consistent rotation
            combined_coords = torch.stack([source_coords, target_coords], dim=1)  # [N, 2, n_atoms, 3]
            combined_shape = combined_coords.shape
            
            # Reshape to apply rotation to pairs together
            combined_flat = combined_coords.view(-1, n_atoms, 3)  # [2*N, n_atoms, 3]
            
            # Apply rotation (each pair gets the same rotation)
            augmented_flat = torch.zeros_like(combined_flat)
            for i in range(0, combined_flat.shape[0], 2):
                # Get source and target for this sample
                pair_coords = combined_flat[i:i+2]  # [2, n_atoms, 3]
                # Apply same rotation to both
                pair_augmented = random_rotation_augment(pair_coords)
                augmented_flat[i:i+2] = pair_augmented
            
            # Reshape back and extract source/target
            augmented_combined = augmented_flat.view(combined_shape)
            source_coords = augmented_combined[:, 0]  # [N, n_atoms, 3]
            target_coords = augmented_combined[:, 1]  # [N, n_atoms, 3]
            
            print(f"Applied coordinate augmentation to {len(source_coords)} coordinate pairs")
        
        # Apply chirality filtering if enabled
        # Note: chirality filtering is typically done on combined data, but for now
        # we apply it separately to source and target coordinates
        if self.filter_chirality_enabled:
            source_coords, source_chirality_stats = filter_chirality(source_coords)
            target_coords, target_chirality_stats = filter_chirality(target_coords)
            
            # Report chirality filtering results
            if source_chirality_stats[0] > 0 or target_chirality_stats[0] > 0:
                print(f"Chirality filtering: Source {source_chirality_stats}, Target {target_chirality_stats}")
                
            # Ensure same number of samples after filtering
            min_samples = min(len(source_coords), len(target_coords))
            source_coords = source_coords[:min_samples]
            target_coords = target_coords[:min_samples]
        
        # Store as instance variables  
        self.source_coords = source_coords.float()
        self.target_coords = target_coords.float()
        
        # Validate shapes match
        if self.source_coords.shape != self.target_coords.shape:
            raise ValueError(
                f"Source and target coordinate shapes don't match: "
                f"{self.source_coords.shape} vs {self.target_coords.shape}"
            )
            
        print(f"Loaded PT dataset: {len(self)} samples, coord shape: {self.source_coords.shape[1:]}")
    
    def _load_molecular_data(self) -> None:
        """Load molecular structure data (atom types and adjacency list)."""
        # Load atom types
        atom_types_path = self.molecular_data_path / "atom_types.pt"
        if not atom_types_path.exists():
            raise FileNotFoundError(f"Atom types file not found: {atom_types_path}")
        
        try:
            self.atom_types = torch.load(atom_types_path, map_location="cpu", weights_only=True)
        except Exception:
            self.atom_types = torch.load(atom_types_path, map_location="cpu")
        
        # Load adjacency list
        adj_list_path = self.molecular_data_path / "adj_list.pt"
        if not adj_list_path.exists():
            raise FileNotFoundError(f"Adjacency list file not found: {adj_list_path}")
        
        try:
            self.adj_list = torch.load(adj_list_path, map_location="cpu", weights_only=True)
        except Exception:
            self.adj_list = torch.load(adj_list_path, map_location="cpu")
        
        # Validate data shapes
        n_atoms = self.source_coords.shape[1]
        if len(self.atom_types) != n_atoms:
            raise ValueError(
                f"Atom types length ({len(self.atom_types)}) doesn't match number of atoms ({n_atoms})"
            )
        
        # Handle different adjacency list formats 
        if self.adj_list.ndim == 2:
            if self.adj_list.shape[0] == 2:
                # Format: [2, n_edges] -> transpose to [n_edges, 2]
                self.adj_list = self.adj_list.T
            # else: already in [n_edges, 2] format
        else:
            raise ValueError(f"Unexpected adjacency list shape: {self.adj_list.shape}")
        
        print(f"Loaded molecular data: {len(self.atom_types)} atoms, {len(self.adj_list)} edges")
        print(f"Atom types: {self.atom_types}")
        print(f"Adjacency list shape: {self.adj_list.shape}")
    
    def __len__(self) -> int:
        """Return number of coordinate pairs."""
        return len(self.source_coords)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single coordinate pair with molecular structure data."""
        return {
            "source_coords": self.source_coords[idx],
            "target_coords": self.target_coords[idx],
            "atom_types": self.atom_types,  # Same for all samples
            "adj_list": self.adj_list,      # Same for all samples
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.
        
        Parameters
        ----------
        batch : List[Dict[str, torch.Tensor]]
            List of samples from __getitem__
            
        Returns
        -------
        collated : Dict[str, torch.Tensor]
            Batched data with proper edge batch indices for message passing
        """
        batch_size = len(batch)
        
        # Stack coordinate data
        source_coords = torch.stack([sample["source_coords"] for sample in batch])
        target_coords = torch.stack([sample["target_coords"] for sample in batch])
        
        # Atom types are the same for all molecules, so we just stack them
        atom_types = torch.stack([sample["atom_types"] for sample in batch])
        
        # For message passing: replicate adjacency list for each molecule in batch
        adj_list = batch[0]["adj_list"]  # Same topology for all molecules
        n_edges = adj_list.shape[0]
        
        # Replicate adj_list for each molecule in the batch
        adj_list_batched = torch.cat([adj_list for _ in range(batch_size)], dim=0)
        
        # Create edge batch indices: [0, 0, 0, ..., 1, 1, 1, ..., B-1, B-1, B-1]
        edge_batch_idx = torch.repeat_interleave(
            torch.arange(batch_size), n_edges
        )
        
        return {
            "source_coords": source_coords,
            "target_coords": target_coords,
            "atom_types": atom_types,
            "adj_list": adj_list_batched,  # Now properly batched
            "edge_batch_idx": edge_batch_idx,
        } 