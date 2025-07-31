"""Multi-peptide dataset for PT swap flow training.

Combines multiple PTTemperaturePairDataset instances to enable training flows
on mixed peptide datasets while maintaining separate evaluation per peptide.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import torch
from torch.utils.data import Dataset

from .pt_pair_dataset import PTTemperaturePairDataset

__all__ = ["MultiPeptidePairDataset", "collate_padded", "RoundRobinLoader"]


class MultiPeptidePairDataset(Dataset):
    """Dataset for PT coordinate pairs from multiple peptides.
    
    Internally builds one PTTemperaturePairDataset per peptide and provides
    unified access via global indexing.
    
    Parameters
    ----------
    peptides : List[str]
        List of peptide codes (e.g., ['AA', 'AK', 'AS'])
    temp_pair : Tuple[int, int]
        Indices of the temperature pair (e.g., (0, 1) for T0 -> T1)
    base_data_dir : str
        Base directory containing peptide subdirectories
        (e.g., 'datasets/pt_dipeptides' containing AA/, AK/, AS/ subdirs)
    subsample_rate : int, optional
        Take every Nth frame (default: 100)
    device : str, optional
        Device for tensors (default: "cpu")
    filter_chirality : bool, optional
        Whether to filter out incorrect chirality conformations (default: False)
    center_coordinates : bool, optional
        Whether to center coordinates (default: True)
    augment_coordinates : bool, optional
        Whether to apply random rotation augmentation (default: False)
    """
    
    def __init__(
        self,
        peptides: List[str],
        temp_pair: Tuple[int, int],
        base_data_dir: str = "datasets/pt_dipeptides",
        subsample_rate: int = 100,
        device: str = "cpu",
        filter_chirality: bool = False,
        center_coordinates: bool = True,
        augment_coordinates: bool = False,
    ) -> None:
        self.peptides = peptides
        self.temp_pair = temp_pair
        self.base_data_dir = Path(base_data_dir)
        
        # Build individual datasets
        self.datasets: List[PTTemperaturePairDataset] = []
        self.peptide_names: List[str] = []
        self.cum_lens: List[int] = []
        
        total_len = 0
        for peptide in peptides:
            try:
                # Construct paths for this peptide
                peptide_dir = self.base_data_dir / peptide
                pt_data_path = peptide_dir / f"pt_{peptide}.pt"
                molecular_data_path = peptide_dir
                
                # Create dataset for this peptide
                dataset = PTTemperaturePairDataset(
                    pt_data_path=str(pt_data_path),
                    molecular_data_path=str(molecular_data_path),
                    temp_pair=temp_pair,
                    subsample_rate=subsample_rate,
                    device=device,
                    filter_chirality=filter_chirality,
                    center_coordinates=center_coordinates,
                    augment_coordinates=augment_coordinates,
                )
                
                self.datasets.append(dataset)
                self.peptide_names.append(peptide)
                total_len += len(dataset)
                self.cum_lens.append(total_len)
                
                print(f"Loaded {peptide}: {len(dataset)} samples")
                
            except Exception as e:
                print(f"Warning: Failed to load peptide {peptide}: {e}")
                # Continue with other peptides
                continue
        
        if not self.datasets:
            raise RuntimeError(f"Failed to load any peptides from {peptides}")
        
        self.total_length = total_len
        print(f"MultiPeptidePairDataset: {len(self.datasets)} peptides, {self.total_length} total samples")
    
    def __len__(self) -> int:
        """Return total number of coordinate pairs across all peptides."""
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single coordinate pair with molecular structure data and peptide ID."""
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range [0, {self.total_length})")
        
        # Find which peptide dataset contains this index
        peptide_idx = 0
        for i, cum_len in enumerate(self.cum_lens):
            if idx < cum_len:
                peptide_idx = i
                break
        
        # Calculate local index within the peptide dataset
        local_idx = idx
        if peptide_idx > 0:
            local_idx = idx - self.cum_lens[peptide_idx - 1]
        
        # Get sample from appropriate dataset
        sample = self.datasets[peptide_idx][local_idx]
        
        # Add peptide identity information
        sample["peptide_name"] = self.peptide_names[peptide_idx]
        sample["peptide_idx"] = peptide_idx
        
        return sample
    
    def get_peptide_datasets(self) -> List[PTTemperaturePairDataset]:
        """Return list of individual peptide datasets for uniform batching."""
        return self.datasets
    
    def get_peptide_names(self) -> List[str]:
        """Return list of peptide names."""
        return self.peptide_names


def collate_padded(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for heterogeneous batches with padding.
    
    Pads all molecular data to the maximum size in the batch and creates
    proper masking for variable-length molecules.
    
    Parameters
    ----------
    batch : List[Dict[str, torch.Tensor]]
        List of samples from MultiPeptidePairDataset.__getitem__
        
    Returns
    -------
    collated : Dict[str, torch.Tensor]
        Batched data with padding and masking
    """
    batch_size = len(batch)
    
    # Find maximum number of atoms in this batch
    n_atoms_list = [sample["source_coords"].shape[0] for sample in batch]
    n_max = max(n_atoms_list)
    
    # Initialize padded tensors
    source_coords_padded = torch.zeros(batch_size, n_max, 3)
    target_coords_padded = torch.zeros(batch_size, n_max, 3)
    atom_types_padded = torch.zeros(batch_size, n_max, dtype=torch.long)  # Use 0 for padding
    masked_elements = torch.ones(batch_size, n_max, dtype=torch.bool)  # True = padding
    
    # Store peptide information
    peptide_names = []
    peptide_indices = []
    
    # Collect all adjacency lists and their sizes
    adj_lists = []
    edge_counts = []
    
    for i, sample in enumerate(batch):
        n_atoms = sample["source_coords"].shape[0]
        
        # Copy coordinate data
        source_coords_padded[i, :n_atoms] = sample["source_coords"]
        target_coords_padded[i, :n_atoms] = sample["target_coords"]
        
        # Copy atom types
        atom_types_padded[i, :n_atoms] = sample["atom_types"]
        
        # Mark valid (non-padding) elements
        masked_elements[i, :n_atoms] = False
        
        # Store peptide information
        peptide_names.append(sample["peptide_name"])
        peptide_indices.append(sample["peptide_idx"])
        
        # Store adjacency list
        adj_lists.append(sample["adj_list"])
        edge_counts.append(sample["adj_list"].shape[0])
    
    # Create batched adjacency list with proper index offsets
    adj_list_batched = []
    edge_batch_idx = []
    
    node_offset = 0
    for i, (adj_list, n_edges) in enumerate(zip(adj_lists, edge_counts)):
        # Add node offset to adjacency list indices
        adj_list_offset = adj_list + node_offset
        adj_list_batched.append(adj_list_offset)
        
        # Create edge batch indices for this molecule
        edge_batch_idx.extend([i] * n_edges)
        
        # Update node offset for next molecule
        node_offset += n_max  # Use padded size for proper indexing
    
    # Concatenate all adjacency lists
    if adj_list_batched:
        adj_list_batched = torch.cat(adj_list_batched, dim=0)
        edge_batch_idx = torch.tensor(edge_batch_idx, dtype=torch.long)
    else:
        adj_list_batched = torch.empty(0, 2, dtype=torch.long)
        edge_batch_idx = torch.empty(0, dtype=torch.long)
    
    return {
        "source_coords": source_coords_padded,
        "target_coords": target_coords_padded,
        "atom_types": atom_types_padded,
        "adj_list": adj_list_batched,
        "edge_batch_idx": edge_batch_idx,
        "masked_elements": masked_elements,
        "peptide_names": peptide_names,
        "peptide_indices": peptide_indices,
    }


class RoundRobinLoader:
    """Round-robin loader for uniform batching across multiple peptides.
    
    Creates one DataLoader per peptide and yields batches from each in turn.
    This ensures each batch contains samples from only one peptide (no padding needed).
    
    Parameters
    ----------
    datasets : List[PTTemperaturePairDataset]
        List of peptide datasets
    batch_size : int
        Batch size for each DataLoader
    shuffle : bool
        Whether to shuffle within each dataset
    **kwargs
        Additional arguments passed to DataLoader
    """
    
    def __init__(
        self,
        datasets: List[PTTemperaturePairDataset],
        batch_size: int,
        shuffle: bool = True,
        peptide_names: List[str] = None,  # Add peptide names
        **kwargs
    ):
        from torch.utils.data import DataLoader
        
        self.peptide_names = peptide_names or [f"peptide_{i}" for i in range(len(datasets))]
        
        self.loaders = []
        for i, dataset in enumerate(datasets):
            # Create a custom collate function that includes peptide information
            peptide_name = self.peptide_names[i]
            
            def make_collate_fn(pep_name):
                def collate_with_peptide_info(batch):
                    # Use the original collate function
                    collated = PTTemperaturePairDataset.collate_fn(batch)
                    # Add peptide information for uniform batches
                    batch_size = collated["source_coords"].shape[0]
                    collated["peptide_names"] = [pep_name] * batch_size
                    collated["peptide_indices"] = [i] * batch_size
                    return collated
                return collate_with_peptide_info
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=make_collate_fn(peptide_name),
                **kwargs
            )
            self.loaders.append(loader)
        
        self.num_loaders = len(self.loaders)
        
        # Calculate total number of batches
        self.total_batches = sum(len(loader) for loader in self.loaders)
    
    def __iter__(self):
        # Create iterators for all loaders
        iterators = [iter(loader) for loader in self.loaders]
        active_iterators = list(range(self.num_loaders))
        
        current_idx = 0
        
        while active_iterators:
            # Get the actual loader index we're trying to access
            loader_idx = active_iterators[current_idx % len(active_iterators)]
            
            try:
                batch = next(iterators[loader_idx])
                
                # Add peptide information for uniform batching
                if self.peptide_names:
                    peptide_name = self.peptide_names[loader_idx]
                    batch['peptide_names'] = [peptide_name] * batch['source_coords'].shape[0]
                
                yield batch
                
                # Move to next loader
                current_idx += 1
                
            except StopIteration:
                # This specific loader is exhausted, remove it from active list
                active_iterators.remove(loader_idx)
                
                # Reset current_idx if it's now out of bounds
                if active_iterators:
                    if current_idx >= len(active_iterators):
                        current_idx = 0
                # If no active iterators left, the while loop will exit 
    
    def __len__(self):
        return self.total_batches 