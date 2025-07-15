"""PT temperature pair dataset for swap flow training.

Loads PT trajectory data for a specific temperature pair and provides
batched coordinate pairs for training swap flows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Dataset

from .molecular_data import filter_chirality, center_coordinates

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
        Path to molecular data directory (for metadata, currently unused)
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
    ) -> None:
        self.pt_data_path = Path(pt_data_path)
        self.molecular_data_path = Path(molecular_data_path)
        self.temp_pair = temp_pair
        self.subsample_rate = subsample_rate
        self.device = device
        self.filter_chirality_enabled = filter_chirality
        self.center_coordinates_enabled = center_coordinates
        
        # Load PT trajectory data
        self._load_pt_data()
        
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
            
        # Apply subsampling to the steps dimension (dimension 2 for format [temps, chains, steps, coords])
        if self.subsample_rate > 1:
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
        
    def __len__(self) -> int:
        """Return number of coordinate pairs."""
        return len(self.source_coords)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single coordinate pair."""
        return {
            "source_coords": self.source_coords[idx],
            "target_coords": self.target_coords[idx],
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
        Dict[str, torch.Tensor]
            Batched tensors with keys matching trainer expectations
        """
        # Stack coordinate tensors
        source_coords = torch.stack([sample["source_coords"] for sample in batch])
        target_coords = torch.stack([sample["target_coords"] for sample in batch])
        
        return {
            "source_coords": source_coords,
            "target_coords": target_coords,
        } 