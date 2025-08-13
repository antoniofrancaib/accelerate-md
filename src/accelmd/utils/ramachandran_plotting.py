"""Ramachandran plot generation utilities for PT swap flow visualization.

This module provides functionality to generate 2×2 Ramachandran plot grids
visualizing how swap flows transform molecular conformations between temperatures.
"""
from __future__ import annotations

import pathlib
from typing import Tuple, List, Optional

import torch
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .config import load_config, create_run_config, setup_device
from ..data.pt_pair_dataset import PTTemperaturePairDataset
from ..flows.pt_swap_flow import PTSwapFlow

# Conditional imports for advanced architectures
try:
    from ..flows.pt_swap_graph_flow import PTSwapGraphFlow
    GRAPH_AVAILABLE = True
except ImportError:
    PTSwapGraphFlow = None
    GRAPH_AVAILABLE = False

try:
    from ..flows.pt_swap_transformer_flow import PTSwapTransformerFlow
    from ..flows.transformer_block import TransformerConfig
    from ..flows.rff_position_encoder import RFFPositionEncoderConfig
    TRANSFORMER_AVAILABLE = True
except ImportError:
    PTSwapTransformerFlow = None
    TransformerConfig = None
    RFFPositionEncoderConfig = None
    TRANSFORMER_AVAILABLE = False
from ..data.molecular_data import filter_chirality
from .plot_utils import plot_Ramachandran

__all__ = ["generate_ramachandran_grid"]


def _phi_psi_angles(coords_nm: np.ndarray, topology: md.Topology) -> Tuple[np.ndarray, np.ndarray]:
    """Return (φ, ψ) arrays in radians for `coords_nm`.

    Parameters
    ----------
    coords_nm
        ndarray of shape [B, N, 3] in **nanometres** where N is number of atoms.
    topology
        MDTraj Topology instance for the peptide.
        
    Returns
    -------
    phi, psi : tuple of arrays
        Phi and psi dihedral angles in radians
    """
    traj = md.Trajectory(coords_nm.copy(), topology)
    phi = md.compute_phi(traj)[1]  # (B, n_phi)
    psi = md.compute_psi(traj)[1]  # (B, n_psi)
    
    # For dipeptides, typically we have 1 phi and 1 psi angle
    # Take the first phi and psi if multiple exist
    if phi.shape[1] > 0:
        phi = phi[:, 0]  # Take first phi angle
    else:
        phi = np.zeros(len(coords_nm))  # Fallback if no phi
        
    if psi.shape[1] > 0:
        psi = psi[:, 0]  # Take first psi angle  
    else:
        psi = np.zeros(len(coords_nm))  # Fallback if no psi
        
    return phi.flatten(), psi.flatten()


def _load_topology(molecular_data_path: str, config: dict) -> Optional[md.Topology]:
    """Load molecular topology for phi/psi angle computation.
    
    Parameters
    ----------
    molecular_data_path
        Path to molecular data directory
    config
        Configuration dictionary containing target information
        
    Returns
    -------
    topology : md.Topology or None
        MDTraj topology object, or None if loading failed
    """
    target_cfg = config.get("target", {})
    peptide_code = config.get("peptide_code", "").upper()
    
    # Handle ALDP target specially - it uses OpenMMTools built-in system
    if peptide_code == "AX":
        print("Using ALDP target - loading topology from OpenMMTools")
        try:
            from openmmtools import testsystems
            from simtk import unit
            
            # Get ALDP test system 
            env = target_cfg.get("kwargs", {}).get("env", "vacuum")
            if env == "implicit":
                testsys = testsystems.AlanineDipeptideImplicit(constraints=None)
            else:
                testsys = testsystems.AlanineDipeptideVacuum(constraints=None)
            
            # Convert OpenMM topology to MDTraj topology
            topology = md.Topology.from_openmm(testsys.topology)
            print(f"Successfully loaded ALDP topology from OpenMMTools ({env} environment)")
            return topology
            
        except Exception as e:
            print(f"Error loading ALDP topology from OpenMMTools: {e}")
            return None
    else:
        # Handle dipeptide target - look for external PDB files
        pdb_files = list(pathlib.Path(molecular_data_path).glob("*.pdb"))
        if not pdb_files:
            # Fallback: try to find reference PDB from target config
            target_kwargs = target_cfg.get("kwargs", {})
            pdb_path = target_kwargs.get("pdb_path")
            print(f"No PDB files found in molecular data directory: {molecular_data_path}")
            if pdb_path:
                print(f"Trying fallback PDB path from config: {pdb_path}")
                if pathlib.Path(pdb_path).exists():
                    topology = md.load(pdb_path).topology
                    print(f"Successfully loaded topology from: {pdb_path}")
                    return topology
                else:
                    print(f"Fallback PDB path does not exist: {pdb_path}")
                    return None
            else:
                print("No pdb_path specified in target config kwargs")
                return None
        else:
            pdb_file = str(pdb_files[0])
            print(f"Found PDB file in molecular data directory: {pdb_file}")
            return md.load(pdb_file).topology


def _build_model_from_config(config: dict, pair: Tuple[int, int], device: str):
    """Build flow model from configuration.
    
    Parameters
    ----------
    config
        Configuration dictionary
    pair
        Temperature pair indices
    device
        Device for model
        
    Returns
    -------
    model : PTSwapFlow
        Initialized flow model
    """
    model_cfg = config["model"]
    temps = config["temperatures"]["values"]
    sys_cfg = config.get("system", {})
    
    # Determine target based on peptide_code
    peptide_code = config["peptide_code"].upper()
    if peptide_code == "AX":
        target_name = "aldp"
        target_kwargs_extra = {}
    else:
        target_name = "dipeptide"
        # For dipeptide target, we need PDB path and environment
        pdb_path = f"datasets/pt_dipeptides/{peptide_code}/ref.pdb"
        target_kwargs_extra = {
            "pdb_path": pdb_path,
            "env": "implicit"
        }
    
    # Add system-level energy parameters to target kwargs
    energy_cut = sys_cfg.get("energy_cut")
    energy_max = sys_cfg.get("energy_max")
    target_kwargs_extra.update({
        "energy_cut": float(energy_cut) if energy_cut is not None else None,
        "energy_max": float(energy_max) if energy_max is not None else None,
    })

    # Read num_atoms from molecular data
    try:
        atom_types_path = pathlib.Path(config["data"]["molecular_data_path"]) / "atom_types.pt"
        atom_types = torch.load(atom_types_path, map_location="cpu")
        num_atoms = int(atom_types.shape[0])
    except Exception as e:
        raise RuntimeError(f"Could not read num_atoms from {atom_types_path}: {e}")

    # Check architecture type to decide which model to build
    architecture = model_cfg.get("architecture", "simple")
    
    if architecture == "simple":
        model = PTSwapFlow(
            num_atoms=num_atoms,
            num_layers=model_cfg["flow_layers"],
            hidden_dim=model_cfg["hidden_dim"],
            source_temperature=temps[pair[0]],
            target_temperature=temps[pair[1]],
            target_name=target_name,
            target_kwargs=target_kwargs_extra,
            device=device,
        )
    elif architecture == "graph":
        if not GRAPH_AVAILABLE:
            raise ValueError("Graph architecture not available due to import errors")
        graph_cfg = model_cfg.get("graph", {})
        model = PTSwapGraphFlow(
            num_layers=model_cfg["flow_layers"],
            atom_vocab_size=graph_cfg.get("atom_vocab_size", 4),
            atom_embed_dim=graph_cfg.get("atom_embed_dim", 32),
            hidden_dim=graph_cfg.get("hidden_dim", model_cfg["hidden_dim"]),
            scale_range=graph_cfg.get("scale_range", 0.05),  # Conservative for stability
            scale_range_end=graph_cfg.get("scale_range_end", 0.15),  # Final scale range
            scale_range_schedule_epochs=graph_cfg.get("scale_range_schedule_epochs", 20),  # Annealing epochs
            max_neighbors=graph_cfg.get("max_neighbors", 20),  # Timewarp parameter
            distance_cutoff=graph_cfg.get("distance_cutoff", 8.0),  # Timewarp parameter
            temperature_conditioning=graph_cfg.get("temperature_conditioning", True),  # Temperature conditioning
            source_temperature=temps[pair[0]],
            target_temperature=temps[pair[1]],
            target_name=target_name,
            target_kwargs=target_kwargs_extra,
            device=device,
        )
    elif architecture == "transformer":
        if not TRANSFORMER_AVAILABLE:
            raise ValueError("Transformer architecture not available due to import errors")
        
        transformer_cfg = model_cfg.get("transformer", {})
        
        # Create transformer configuration
        transformer_config = TransformerConfig(
            n_head=transformer_cfg.get("n_head", 8),
            dim_feedforward=transformer_cfg.get("dim_feedforward", 2048),
            dropout=0.0,  # No dropout for deterministic likelihood
        )
        
        # Create RFF position encoder configuration
        rff_config = RFFPositionEncoderConfig(
            encoding_dim=transformer_cfg.get("rff_encoding_dim", 64),
            scale_mean=transformer_cfg.get("rff_scale_mean", 1.0),
            scale_stddev=transformer_cfg.get("rff_scale_stddev", 1.0),
        )
        
        model = PTSwapTransformerFlow(
            num_layers=model_cfg["flow_layers"],
            atom_vocab_size=transformer_cfg.get("atom_vocab_size", 4),
            atom_embed_dim=transformer_cfg.get("atom_embed_dim", 32),
            transformer_hidden_dim=transformer_cfg.get("transformer_hidden_dim", 128),
            mlp_hidden_layer_dims=transformer_cfg.get("mlp_hidden_layer_dims", [128, 128]),
            num_transformer_layers=transformer_cfg.get("num_transformer_layers", 2),
            source_temperature=temps[pair[0]],
            target_temperature=temps[pair[1]],
            target_name=target_name,
            target_kwargs=target_kwargs_extra,
            transformer_config=transformer_config,
            rff_position_encoder_config=rff_config,
            device=device,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def generate_ramachandran_grid(
    config_path: str,
    checkpoint_path: str,
    temp_pair: Tuple[int, int],
    output_path: str,
    n_samples: int = 5000,
    peptide_code_override: Optional[str] = None,
) -> bool:
    """Generate 2×2 Ramachandran plot grid for swap flow visualization.
    
    The grid layout:
    - top-left     : true T_low data
    - top-right    : mapped high→low (inverse flow)
    - bottom-left  : true T_high data
    - bottom-right : mapped low→high (forward flow)
    
    Parameters
    ----------
    config_path
        Path to YAML configuration file
    checkpoint_path
        Path to model checkpoint file
    temp_pair
        Tuple of (low_temp_idx, high_temp_idx)
    output_path
        Path where to save the plot
    n_samples
        Number of samples to use for plotting
    peptide_code_override : str, optional
        If provided, override the peptide_code from config for multi-peptide evaluation
        
    Returns
    -------
    success : bool
        True if plot was generated successfully, False otherwise
    """
    try:
        # Load configuration
        cfg_base = load_config(config_path)
        device = setup_device(cfg_base)
        pair = tuple(temp_pair)

        # Override peptide_code if provided (for multi-peptide mode)
        if peptide_code_override is not None:
            # Create modified config for this specific peptide
            cfg_base = cfg_base.copy()
            cfg_base["peptide_code"] = peptide_code_override
            
            # Build peptide-specific data paths
            peptide_dir = f"datasets/pt_dipeptides/{peptide_code_override}"
            cfg_base.setdefault("data", {})
            cfg_base["data"]["pt_data_path"] = f"{peptide_dir}/pt_{peptide_code_override}.pt"
            cfg_base["data"]["molecular_data_path"] = peptide_dir

        # Dataset & loader (CPU) – we just iterate sequentially until we have N.
        run_cfg = create_run_config(cfg_base, pair, device="cpu")
        dataset = PTTemperaturePairDataset(
            pt_data_path=run_cfg["data"]["pt_data_path"],
            molecular_data_path=run_cfg["data"]["molecular_data_path"],
            temp_pair=pair,
            subsample_rate=run_cfg["data"].get("subsample_rate", 100),
            device="cpu",
            filter_chirality=run_cfg["data"].get("filter_chirality", False),
            center_coordinates=run_cfg["data"].get("center_coordinates", True),
        )

        # Determine sample count per direction
        n_total = min(n_samples, len(dataset))

        coords_low: List[np.ndarray] = []
        coords_high: List[np.ndarray] = []

        for i in range(n_total):
            sample = dataset[i]
            coords_low.append(sample["source_coords"].numpy())  # (N,3)
            coords_high.append(sample["target_coords"].numpy())
        coords_low_np = np.stack(coords_low, axis=0)  # [B,N,3]
        coords_high_np = np.stack(coords_high, axis=0)

        # Load topology
        molecular_data_path = run_cfg["data"]["molecular_data_path"]
        topology = _load_topology(molecular_data_path, run_cfg)

        # Apply chirality filtering if coordinates are compatible
        if topology is not None:
            try:
                # Our filter_chirality works on tensors with shape [B, dim]
                coords_low_tensor = torch.from_numpy(coords_low_np.reshape(len(coords_low_np), -1))
                coords_high_tensor = torch.from_numpy(coords_high_np.reshape(len(coords_high_np), -1))
                
                # Apply chirality filter
                mask_low, _ = filter_chirality(coords_low_tensor)
                mask_high, _ = filter_chirality(coords_high_tensor)
                combined_mask = mask_low & mask_high
                
                # Apply mask to keep only valid chirality samples
                coords_low_np = coords_low_np[combined_mask.numpy()]
                coords_high_np = coords_high_np[combined_mask.numpy()]
                
                print(f"Chirality filtering: kept {combined_mask.sum().item()}/{len(combined_mask)} samples")
            except Exception as e:
                print(f"Warning: Chirality filtering failed: {e}")
                print("Proceeding without chirality filtering")

        # Build flow model and load checkpoint
        model = _build_model_from_config(run_cfg, pair, device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Map directions (keep tensors on device, then to CPU → numpy)
        with torch.no_grad():
            low_tensor = torch.from_numpy(coords_low_np).to(device)
            high_tensor = torch.from_numpy(coords_high_np).to(device)

            # Handle different flow architectures
            if GRAPH_AVAILABLE and isinstance(model, PTSwapGraphFlow):
                # For graph flows, we need molecular data
                sample = dataset[0]  # Get molecular data from first sample
                batch_size = low_tensor.shape[0]
                atom_types = sample["atom_types"].unsqueeze(0).repeat(batch_size, 1).to(device)
                
                # Replicate adjacency list for batch (same as collate_fn does)
                adj_list_single = sample["adj_list"]
                n_edges = adj_list_single.shape[0]
                adj_list = torch.cat([adj_list_single for _ in range(batch_size)], dim=0).to(device)
                
                # Create proper edge batch indices
                edge_batch_idx = torch.repeat_interleave(
                    torch.arange(batch_size), n_edges
                ).to(device)
                

                
                mapped_hi, _ = model.forward(low_tensor, atom_types=atom_types, adj_list=adj_list, edge_batch_idx=edge_batch_idx)
                mapped_lo, _ = model.inverse(high_tensor, atom_types=atom_types, adj_list=adj_list, edge_batch_idx=edge_batch_idx)
            elif TRANSFORMER_AVAILABLE and isinstance(model, PTSwapTransformerFlow):
                # For transformer flows, we need atom_types
                sample = dataset[0]  # Get molecular data from first sample
                atom_types = sample["atom_types"].unsqueeze(0).repeat(low_tensor.shape[0], 1).to(device)
                
                mapped_hi, _ = model.forward(low_tensor, atom_types=atom_types)   # low → high
                mapped_lo, _ = model.inverse(high_tensor, atom_types=atom_types)  # high → low
            else:
                # Simple flow architecture
                mapped_hi, _ = model.forward(low_tensor)   # low → high
                mapped_lo, _ = model.inverse(high_tensor)  # high → low

            mapped_hi_np = mapped_hi.cpu().numpy()
            mapped_lo_np = mapped_lo.cpu().numpy()

        # Ramachandran angles
        if topology is not None:
            phi_low, psi_low = _phi_psi_angles(coords_low_np, topology)
            phi_high, psi_high = _phi_psi_angles(coords_high_np, topology)
            phi_map_hi, psi_map_hi = _phi_psi_angles(mapped_hi_np, topology)
            phi_map_lo, psi_map_lo = _phi_psi_angles(mapped_lo_np, topology)
        else:
            print("Error: No topology available for Ramachandran plot generation")
            print("Cannot compute phi/psi angles without molecular topology")
            print("Please ensure a PDB file is available in the molecular data directory")
            print("or check that target.kwargs.pdb_path points to a valid PDB file")
            return False

        # Plot grid
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        plot_Ramachandran(axs[0, 0], phi_low, psi_low)
        axs[0, 0].set_title("T_low data")

        plot_Ramachandran(axs[0, 1], phi_map_lo, psi_map_lo)
        axs[0, 1].set_title("High → Low (flow)")

        plot_Ramachandran(axs[1, 0], phi_high, psi_high)
        axs[1, 0].set_title("T_high data")

        plot_Ramachandran(axs[1, 1], phi_map_hi, psi_map_hi)
        axs[1, 1].set_title("Low → High (flow)")

        for ax in axs.flat:
            ax.set_aspect('equal')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        out_path = pathlib.Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()  # Close the figure to free memory
        print(f"Saved Ramachandran grid to {out_path}")
        return True

    except Exception as e:
        print(f"Error generating Ramachandran plot: {e}")
        import traceback
        traceback.print_exc()
        return False 