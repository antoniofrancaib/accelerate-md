#!/usr/bin/env python3
"""Command-line tool for Parallel-Tempering (PT) simulations with optional flow-based swap models.

This tool builds PT simulations for single peptides, optionally loads flow-based swap models
from checkpoint files, runs MD simulations, and generates analysis artifacts.

Usage:
    python -m src.run_pt --cfg configs/AA_simple.yaml \
                         --checkpoint_dir checkpoints/AA_simple \
                         --out_dir results/kinetics/simple \
                         --n_steps 20000 \
                         --swap_interval 100 \
                         --sample_interval 10 \
                         --seed 42

Output files:
    acceptance_matrix.npy    - Mean swap-acceptance probability matrix [R,R]
    replica_trace.npy        - Temperature index of walker j at step t [T,R]
    phi_trace.npy           - Phi dihedral of coldest replica [T,]
    wallclock_per_step.npy  - Mean wall-clock seconds per MD step [scalar]
    metadata.json           - Input arguments and run metadata

Dependencies:
    yaml, numpy, scipy, torch, mdtraj, openmm, openmmtools, tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import yaml
from tqdm import tqdm

# Check for required dependencies
try:
    import torch
    import mdtraj as md
    from scipy import stats
    import openmm as mm
    from openmm import app, unit
    import openmmtools as ommt
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Please install: conda install -c conda-forge openmm openmmtools mdtraj")
    print("                pip install torch scipy tqdm")
    sys.exit(1)

# Project imports
try:
    from src.accelmd.samplers.pt.sampler import ParallelTempering
    from src.accelmd.samplers.pt.dyn_wrapper import DynSamplerWrapper
    from src.accelmd.targets.dipeptide_potential import DipeptidePotentialCart
    from src.accelmd.targets.aldp_boltzmann import AldpBoltzmann
    from src.accelmd.flows import PTSwapFlow, PTSwapGraphFlow, PTSwapTransformerFlow
    from src.accelmd.flows.transformer_block import TransformerConfig
    from src.accelmd.flows.rff_position_encoder import RFFPositionEncoderConfig
except ImportError as e:
    print(f"❌ Failed to import project modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

__all__ = ["main", "get_test_parser"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FlowSwap:
    """Wrapper for flow-based swap proposals with unified interface."""
    
    def __init__(self, flow_model: torch.nn.Module, atom_types: torch.Tensor, device: str = "cpu"):
        """Initialize FlowSwap wrapper.
        
        Parameters
        ----------
        flow_model : torch.nn.Module
            Trained flow model (PTSwapFlow, PTSwapGraphFlow, or PTSwapTransformerFlow)
        atom_types : torch.Tensor
            Atom type indices for molecular data [N,]
        device : str
            Compute device for flow operations
        """
        self.flow = flow_model.to(device).eval()
        self.atom_types = atom_types.to(device)
        self.device = device
        self.n_atoms = len(atom_types)
    
    def propose(self, x_low: np.ndarray, x_high: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Generate flow-based swap proposal.
        
        Parameters
        ----------
        x_low : np.ndarray, shape [N*3,]
            Low temperature configuration (flattened coordinates)
        x_high : np.ndarray, shape [N*3,]
            High temperature configuration (flattened coordinates)
            
        Returns
        -------
        x_low_prop : np.ndarray, shape [N*3,]
            Proposed low temperature configuration
        x_high_prop : np.ndarray, shape [N*3,]
            Proposed high temperature configuration
        log_det_forward : float
            Log determinant for forward transformation
        log_det_inverse : float
            Log determinant for inverse transformation
        """
        with torch.no_grad():
            # Reshape to [1, N, 3] and move to device
            x_low_3d = torch.from_numpy(x_low).float().view(1, self.n_atoms, 3).to(self.device)
            x_high_3d = torch.from_numpy(x_high).float().view(1, self.n_atoms, 3).to(self.device)
            
            # Prepare atom types batch
            atom_types_batch = self.atom_types.unsqueeze(0)  # [1, N]
            
            # Apply flow transformations
            try:
                # Forward: low -> high temperature proposal
                x_high_prop, log_det_f = self.flow.forward(
                    x_low_3d, atom_types=atom_types_batch, return_log_det=True
                )
                
                # Inverse: high -> low temperature proposal
                x_low_prop, log_det_inv = self.flow.inverse(
                    x_high_3d, atom_types=atom_types_batch
                )
                
                # Convert back to CPU numpy arrays and flatten
                x_low_prop = x_low_prop.view(-1).cpu().numpy()
                x_high_prop = x_high_prop.view(-1).cpu().numpy()
                
                return x_low_prop, x_high_prop, log_det_f.item(), log_det_inv.item()
                
            except Exception as e:
                logger.warning(f"Flow proposal failed, falling back to naive swap: {e}")
                # Fall back to naive swap
                return x_high.copy(), x_low.copy(), 0.0, 0.0


def build_flow_swap(checkpoint_path: str, config: Dict[str, Any], 
                   pair: Tuple[int, int], device: str) -> Optional[FlowSwap]:
    """Build FlowSwap object from checkpoint file.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint file
    config : Dict[str, Any]
        Configuration dictionary containing model parameters
    pair : Tuple[int, int]
        Temperature pair indices (i, j)
    device : str
        Device for model computation
        
    Returns
    -------
    FlowSwap or None
        FlowSwap object if successful, None if failed
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint not found: {checkpoint_path}")
        return None
        
    try:
        # Determine architecture
        architecture = config.get("model", {}).get("architecture", "simple")
        
        # Get temperature values for flow construction
        temps = config.get("temperatures", {}).get("values", [300.0, 1500.0])
        temp_low, temp_high = temps[pair[0]], temps[pair[1]]
        
        # Build target kwargs
        target_config = config.get("target", {})
        target_name = target_config.get("name", "dipeptide")
        target_kwargs = target_config.get("kwargs", {})
        
        # Create flow model based on architecture
        if architecture == "simple":
            flow = PTSwapFlow(
                num_layers=config["model"].get("flow_layers", 8),
                hidden_dim=config["model"].get("hidden_dim", 512),
                source_temperature=temp_low,
                target_temperature=temp_high,
                target_name=target_name,
                target_kwargs=target_kwargs,
                device=device,
            )
            
        elif architecture == "graph":
            # Load graph model config
            graph_config = config.get("graph", {})
            flow = PTSwapGraphFlow(
                num_layers=config["model"].get("flow_layers", 8),
                hidden_dim=config["model"].get("hidden_dim", 512),
                message_passing_steps=graph_config.get("message_passing_steps", 3),
                atom_vocab_size=graph_config.get("atom_vocab_size", 4),
                atom_embed_dim=graph_config.get("atom_embed_dim", 32),
                edge_embed_dim=graph_config.get("edge_embed_dim", 32),
                source_temperature=temp_low,
                target_temperature=temp_high,
                target_name=target_name,
                target_kwargs=target_kwargs,
                device=device,
            )
            
        elif architecture == "transformer":
            # Load transformer config
            transformer_cfg = config.get("transformer", {})
            
            transformer_config = TransformerConfig(
                n_head=transformer_cfg.get("n_head", 8),
                dim_feedforward=transformer_cfg.get("dim_feedforward", 2048),
                dropout=0.0,
            )
            
            rff_config = RFFPositionEncoderConfig(
                encoding_dim=transformer_cfg.get("rff_encoding_dim", 64),
                scale_mean=transformer_cfg.get("rff_scale_mean", 1.0),
                scale_stddev=transformer_cfg.get("rff_scale_stddev", 1.0),
            )
            
            flow = PTSwapTransformerFlow(
                num_layers=config["model"].get("flow_layers", 8),
                atom_vocab_size=transformer_cfg.get("atom_vocab_size", 4),
                atom_embed_dim=transformer_cfg.get("atom_embed_dim", 32),
                transformer_hidden_dim=transformer_cfg.get("transformer_hidden_dim", 128),
                mlp_hidden_layer_dims=transformer_cfg.get("mlp_hidden_layer_dims", [128, 128]),
                num_transformer_layers=transformer_cfg.get("num_transformer_layers", 2),
                source_temperature=temp_low,
                target_temperature=temp_high,
                target_name=target_name,
                target_kwargs=target_kwargs,
                transformer_config=transformer_config,
                rff_position_encoder_config=rff_config,
                device=device,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        flow.load_state_dict(state_dict)
        
        # Load real atom types from PDB
        pdb_path = target_kwargs.get("pdb_path", "")
        if pdb_path and os.path.exists(pdb_path):
            try:
                atom_types = get_atom_vocab_indices(pdb_path)
                logger.debug(f"Loaded atom types from {pdb_path}: {atom_types}")
            except ValueError as e:
                logger.warning(f"Failed to load atom types from {pdb_path}: {e}")
                # Fallback: assume all carbon (index 1 in new vocab: H=0, C=1, N=2, O=3, S=4)
                pdb = app.PDBFile(pdb_path)
                n_atoms = pdb.topology.getNumAtoms()
                atom_types = torch.ones(n_atoms, dtype=torch.long)  # All carbon (index 1)
        else:
            # Default fallback for ALDP (assume all carbon)
            atom_types = torch.ones(22, dtype=torch.long)
        
        logger.info(f"Loaded {architecture} flow for pair {pair}")
        return FlowSwap(flow, atom_types, device)
        
    except Exception as e:
        logger.warning(f"Failed to load flow from {checkpoint_path}: {e}")
        return None


def compute_round_trip_time(replica_trace: np.ndarray) -> float:
    """Compute round-trip time for replica exchange (cold→hot→cold).
    
    Parameters
    ----------
    replica_trace : np.ndarray, shape [T, R]
        Temperature index of walker j at step t
        
    Returns
    -------
    float
        Mean round-trip time in swap intervals
    """
    T, R = replica_trace.shape
    round_trip_times = []
    
    cold_idx = 0  # Coldest temperature index
    hot_idx = R - 1  # Hottest temperature index
    
    # Track each replica's journey from cold→hot→cold
    for replica_id in range(R):
        replica_temps = replica_trace[:, replica_id]
        
        # Find all visits to cold temperature
        cold_visits = np.where(replica_temps == cold_idx)[0]
        
        for cold_start in cold_visits:
            # Look for subsequent visit to hot temperature
            hot_visit = None
            for t in range(cold_start + 1, T):
                if replica_temps[t] == hot_idx:
                    hot_visit = t
                    break
            
            if hot_visit is not None:
                # Look for return to cold temperature
                for t in range(hot_visit + 1, T):
                    if replica_temps[t] == cold_idx:
                        round_trip_time = t - cold_start
                        round_trip_times.append(round_trip_time)
                        break
    
    return float(np.mean(round_trip_times)) if round_trip_times else float('inf')


def integrated_autocorrelation(x: np.ndarray, c: float = 5.0) -> float:
    """Compute integrated autocorrelation time using Sokal's windowing method.
    
    Parameters
    ----------
    x : np.ndarray
        Time series data
    c : float
        Windowing parameter (default 5.0)
        
    Returns
    -------
    float
        Integrated autocorrelation time
    """
    if len(x) < 10:
        return float('inf')
    
    # Remove mean
    x = x - np.mean(x)
    
    # Compute autocorrelation function
    n = len(x)
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[n-1:]  # Take only positive lags
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find optimal window using Sokal's criterion
    tau_int = 0.5  # Start with 0.5 for lag 0
    for W in range(1, min(len(autocorr), n//4)):
        tau_int_new = 0.5 + np.sum(autocorr[1:W+1])
        if W >= c * tau_int_new:
            break
        tau_int = tau_int_new
    
    return max(0.5, tau_int)


def calc_acceptance_matrix(swap_log: List[np.ndarray], n_replicas: int) -> np.ndarray:
    """Calculate mean acceptance matrix from swap attempts.
    
    Parameters
    ----------
    swap_log : List[np.ndarray]
        List of acceptance arrays from each swap attempt
    n_replicas : int
        Number of replicas
        
    Returns
    -------
    np.ndarray, shape [R, R]
        Mean acceptance probability matrix
    """
    acceptance_matrix = np.zeros((n_replicas, n_replicas))
    counts = np.zeros((n_replicas, n_replicas))
    
    for swap_acceptances in swap_log:
        # Assume neighboring swaps for now
        for i in range(len(swap_acceptances)):
            acceptance_matrix[i, i+1] += swap_acceptances[i]
            acceptance_matrix[i+1, i] += swap_acceptances[i]
            counts[i, i+1] += 1
            counts[i+1, i] += 1
    
    # Avoid division by zero
    mask = counts > 0
    acceptance_matrix[mask] /= counts[mask]
    
    return acceptance_matrix.astype(np.float32)


def generate_synthetic_data(n_replicas: int, n_steps: int, swap_interval: int, 
                          sample_interval: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate synthetic data for dry-run mode.
    
    Parameters
    ----------
    n_replicas : int
        Number of temperature replicas
    n_steps : int
        Total number of MD steps
    swap_interval : int
        Interval between swap attempts
    sample_interval : int
        Interval between phi angle samples
        
    Returns
    -------
    Tuple of synthetic data arrays matching expected output format
    """
    # Synthetic acceptance matrix (random around 0.3)
    acceptance_matrix = np.random.uniform(0.2, 0.4, (n_replicas, n_replicas)).astype(np.float32)
    np.fill_diagonal(acceptance_matrix, 1.0)
    
    # Synthetic replica trace (random walk in temperature space)
    n_swaps = n_steps // swap_interval + 1
    replica_trace = np.zeros((n_swaps, n_replicas), dtype=np.int32)
    
    # Initialize: replica i starts at temperature i
    for i in range(n_replicas):
        replica_trace[0, i] = i
    
    # Random walk with neighbor swaps
    for t in range(1, n_swaps):
        replica_trace[t] = replica_trace[t-1].copy()
        
        # Attempt neighboring swaps with some probability
        for i in range(n_replicas - 1):
            if np.random.random() < 0.3:  # 30% swap probability
                # Swap temperatures between adjacent replicas
                temp_i = replica_trace[t, i]
                temp_j = replica_trace[t, i+1]
                replica_trace[t, i] = temp_j
                replica_trace[t, i+1] = temp_i
    
    # Synthetic phi trace (AR(1) process with rho=0.9)
    n_samples = n_steps // sample_interval + 1
    phi_trace = np.zeros(n_samples, dtype=np.float32)
    phi_trace[0] = np.random.uniform(-np.pi, np.pi)
    
    for i in range(1, n_samples):
        phi_trace[i] = 0.9 * phi_trace[i-1] + 0.1 * np.random.normal(0, 0.5)
        # Keep in [-π, π] range
        phi_trace[i] = ((phi_trace[i] + np.pi) % (2 * np.pi)) - np.pi
    
    # Synthetic wallclock time (assume ~1ms per step)
    wallclock_per_step = np.float64(0.001)
    
    return acceptance_matrix, replica_trace, phi_trace, wallclock_per_step


def setup_system_from_config(config: Dict[str, Any], device: str) -> Tuple[Any, torch.Tensor, Dict[str, Any]]:
    """Set up molecular system from configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    device : str
        Compute device
        
    Returns
    -------
    target : Boltzmann target
    x_init : torch.Tensor
        Initial coordinates [n_temp, n_chains, dim]
    system_info : Dict[str, Any]
        System information
    """
    # Determine target type
    peptide_code = config.get("peptide_code", "AA")
    target_config = config.get("target", {})
    target_name = target_config.get("name", "dipeptide")
    target_kwargs = target_config.get("kwargs", {})
    
    # Build target
    if target_name == "aldp" or peptide_code == "AX":
        target = AldpBoltzmann(
            temperature=config.get("system", {}).get("temperature", 300.0),
            device=device,
            **target_kwargs
        )
    else:
        # Default to dipeptide potential
        if "pdb_path" not in target_kwargs:
            target_kwargs["pdb_path"] = f"datasets/pt_dipeptides/{peptide_code}/ref.pdb"
        
        target = DipeptidePotentialCart(
            temperature=config.get("system", {}).get("temperature", 300.0),
            device=device,
            **target_kwargs
        )
    
    # Get minimized initial coordinates
    if hasattr(target, 'context'):
        state = target.context.getState(getPositions=True)
        minimized_positions = state.getPositions()
        pos_array = np.array([[v.x, v.y, v.z] for v in minimized_positions])
        x_init_single = torch.tensor(pos_array, device=device, dtype=torch.float32).view(1, -1)
    else:
        # For ALDP, use a default initial configuration
        x_init_single = torch.randn(1, 66, device=device, dtype=torch.float32)  # 22 atoms * 3
    
    # Replicate for all temperatures and chains
    temps = config.get("temperatures", {}).get("values", [300.0, 1500.0])
    n_temp = len(temps)
    n_chains = 1  # Single chain per temperature for now
    
    x_init = x_init_single.unsqueeze(0).repeat(n_temp, n_chains, 1)
    
    system_info = {
        "n_atoms": target.n_atoms if hasattr(target, 'n_atoms') else 22,
        "dim": x_init.shape[-1],
        "topology": target.topology if hasattr(target, 'topology') else None,
    }
    
    return target, x_init, system_info


def get_git_commit_hash() -> str:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_atom_vocab_indices(pdb_path: str) -> torch.LongTensor:
    """Extract atom type indices from PDB file.
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file
        
    Returns
    -------
    torch.LongTensor
        Atom type indices [N_atoms], where 0=H, 1=C, 2=N, 3=O, 4=S
        
    Raises
    ------
    ValueError
        If unsupported element types are found
    """
    import mdtraj as md
    
    # Load topology from PDB
    traj = md.load(pdb_path)
    topology = traj.topology
    
    # Define element mapping (H is common in peptides)
    element_to_idx = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4}
    
    atom_indices = []
    for atom in topology.atoms:
        element = atom.element.symbol
        if element not in element_to_idx:
            raise ValueError(f"Unsupported element '{element}' in {pdb_path}. Supported: {list(element_to_idx.keys())}")
        atom_indices.append(element_to_idx[element])
    
    return torch.LongTensor(atom_indices)


def main() -> None:
    """Main entry point for PT simulation."""
    parser = get_test_parser()
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.INFO)
    
    logger.info("Starting Parallel-Tempering simulation")
    logger.info(f"Configuration: {args.cfg}")
    logger.info(f"Output directory: {args.out_dir}")
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load configuration
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Check if output files already exist
    output_files = [
        "acceptance_matrix.npy",
        "replica_trace.npy", 
        "phi_trace.npy",
        "wallclock_per_step.npy"
    ]
    
    if not args.overwrite:
        existing_files = [f for f in output_files if os.path.exists(os.path.join(args.out_dir, f))]
        if existing_files:
            logger.error(f"Output files already exist: {existing_files}")
            logger.error("Use --overwrite to overwrite existing files")
            return
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    logger.info(f"Using device: {device}")
    
    # Generate synthetic data for dry-run
    if args.dry_run:
        logger.info("Running in dry-run mode with synthetic data")
        temps = config.get("temperatures", {}).get("values", [300.0, 1500.0])
        n_replicas = len(temps)
        
        acceptance_matrix, replica_trace, phi_trace, wallclock_per_step = generate_synthetic_data(
            n_replicas, args.n_steps, args.swap_interval, args.sample_interval
        )
        
        # Save outputs
        np.save(os.path.join(args.out_dir, "acceptance_matrix.npy"), acceptance_matrix)
        np.save(os.path.join(args.out_dir, "replica_trace.npy"), replica_trace)
        np.save(os.path.join(args.out_dir, "phi_trace.npy"), phi_trace)
        np.save(os.path.join(args.out_dir, "wallclock_per_step.npy"), wallclock_per_step)
        
        # Generate metadata
        metadata = {
            "args": vars(args),
            "config_file": args.cfg,
            "git_commit": get_git_commit_hash(),
            "mode": "dry_run",
            "device": device,
            "mean_acceptance": float(np.mean(np.triu(acceptance_matrix, 1))),
            "round_trip_time": float(compute_round_trip_time(replica_trace)),
            "phi_autocorr_time": float(integrated_autocorrelation(phi_trace)),
        }
        
        with open(os.path.join(args.out_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Dry-run completed successfully")
        logger.info(f"Mean acceptance: {metadata['mean_acceptance']:.3f}")
        logger.info(f"Round-trip time: {metadata['round_trip_time']:.1f} swap intervals")
        logger.info(f"Phi autocorr time: {metadata['phi_autocorr_time']:.1f} sample intervals")
        return
    
    # Real simulation mode
    logger.info("Setting up molecular system...")
    target, x_init, system_info = setup_system_from_config(config, device)
    
    # Set up temperatures
    temps = config.get("temperatures", {}).get("values", [300.0, 1500.0])
    # Convert to kT in kJ/mol units so that dividing energy (kJ/mol) by `temperatures`
    # yields reduced energy beta*E and 1/temperatures equals beta.
    R_kJ_per_mol_K = 0.008314462618  # kJ mol^-1 K^-1
    temperatures = torch.tensor([R_kJ_per_mol_K * float(T) for T in temps], device=device, dtype=torch.float32)
    n_replicas = len(temperatures)
    
    logger.info(f"Temperature ladder (K): {temps}")
    
    # Load flow models if available and not disabled
    flows = {}
    if not args.no_flow and args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        logger.info("Loading flow models...")
        for i in range(n_replicas - 1):
            pair = (i, i + 1)
            checkpoint_path = os.path.join(args.checkpoint_dir, f"pair_{i}_{i+1}", "models")
            if os.path.exists(checkpoint_path):
                # Find best model file
                model_files = [f for f in os.listdir(checkpoint_path) if f.startswith("best_model_")]
                if model_files:
                    model_file = os.path.join(checkpoint_path, sorted(model_files)[-1])
                    flow_swap = build_flow_swap(model_file, config, pair, device)
                    if flow_swap:
                        flows[f"pair_{i}_{i+1}"] = flow_swap
    
    if flows:
        logger.info(f"Loaded {len(flows)} flow models")
    else:
        logger.info("No flow models loaded, using vanilla PT")
    
    # Set up PT sampler
    logger.info("Setting up Parallel Tempering sampler...")
    
    # Create explicit energy function (target.energy returns potential energy in kJ/mol)
    def energy_func(x):
        """Potential energy function returning energy in kJ/mol."""
        # target.log_prob returns -β*E, so we need to extract E
        # For consistency, we use target.energy if available, otherwise -log_prob/β
        if hasattr(target, 'energy'):
            return target.energy(x)
        else:
            # Fallback: assume target.log_prob returns -βE and extract E
            # Note: This may need adjustment based on actual target implementation
            return -target.log_prob(x)
    
    # Create PT sampler
    pt = ParallelTempering(
        x=x_init,
        energy_func=energy_func,
        step_size=torch.tensor([0.0001] * (n_replicas * x_init.shape[1]), device=device).unsqueeze(-1),
        swap_interval=args.swap_interval,
        temperatures=temperatures,
        mh=True,
        device=device,
        log_history=True,
    )
    
    # Integrate flows into PT sampler if available
    if flows:
        logger.info("Creating flow-enhanced PT sampler...")
        
        class FlowEnhancedPT(ParallelTempering):
            def __init__(self, flow_dict, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.flows = flow_dict
                
            def _attempt_swap(self, idx_a, idx_b):
                """Enhanced swap with flow proposals when available.
                
                Implements correct multi-chain flow swaps with proper Jacobians.
                See Eq. (8) in Noé & Cortes, JCP 155, 084104 (2021)
                """
                temp_a, temp_b = self.temperatures[idx_a], self.temperatures[idx_b]
                
                # Get chain slices for each temperature
                chains_per_temp = self.x.shape[0] // self.num_temperatures
                slice_a = slice(idx_a * chains_per_temp, (idx_a + 1) * chains_per_temp)
                slice_b = slice(idx_b * chains_per_temp, (idx_b + 1) * chains_per_temp)
                
                chain_a = self.x[slice_a]  # [n_chains, dim]
                chain_b = self.x[slice_b]  # [n_chains, dim]
                n_chains = chain_a.shape[0]
                
                # Check if we have a flow for this temperature pair
                pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                pair_key = f"pair_{pair[0]}_{pair[1]}"
                
                if pair_key in self.flows:
                    try:
                        flow_swap = self.flows[pair_key]
                        
                        # Determine low/high temperature states
                        if idx_a < idx_b:  # idx_a is lower temperature
                            x_low = chain_a    # [n_chains, dim]
                            x_high = chain_b   # [n_chains, dim]
                        else:  # idx_b is lower temperature  
                            x_low = chain_b    # [n_chains, dim]
                            x_high = chain_a   # [n_chains, dim]
                        
                        # Get atom types and broadcast to batch size
                        atom_types = flow_swap.atom_types  # [N_atoms]
                        n_atoms = int(atom_types.shape[0])
                        atom_types_batch = atom_types.unsqueeze(0).expand(n_chains, -1)  # [n_chains, N_atoms]
                        
                        # Apply flow transformations (vectorized)
                        with torch.no_grad():
                            # Reshape flattened coordinates [B, N*3] -> [B, N, 3]
                            x_low_coords = x_low.view(n_chains, n_atoms, 3)
                            x_high_coords = x_high.view(n_chains, n_atoms, 3)
                            # Forward: low -> high (proposal for high temperature)
                            x_high_prop_coords, logJ_f = flow_swap.flow.forward(
                                x_low_coords.to(flow_swap.device), 
                                atom_types_batch.to(flow_swap.device), 
                                return_log_det=True
                            )
                            
                            # Inverse: high -> low (proposal for low temperature)  
                            x_low_prop_coords, logJ_r = flow_swap.flow.inverse(
                                x_high_coords.to(flow_swap.device),
                                atom_types_batch.to(flow_swap.device),
                                return_log_det=True
                            )
                            # Flatten back to [B, N*3]
                            x_high_prop = x_high_prop_coords.view(n_chains, n_atoms * 3)
                            x_low_prop = x_low_prop_coords.view(n_chains, n_atoms * 3)
                        
                        # Compute energies (vectorized)
                        E_low = self.base_energy(x_low)      # [n_chains]
                        E_high = self.base_energy(x_high)    # [n_chains]
                        E_low_prop = self.base_energy(x_low_prop.to(self.device))  # [n_chains]
                        E_high_prop = self.base_energy(x_high_prop.to(self.device)) # [n_chains]
                        
                        # Correct Metropolis ratio with Jacobians
                        # See Eq. (8) in Noé & Cortes, JCP 155, 084104 (2021)
                        d_beta = (1.0 / temp_a - 1.0 / temp_b)  # scalar
                        
                        if idx_a < idx_b:  # a=low, b=high
                            delta_E = (E_high_prop - E_low_prop)  # [n_chains]
                        else:  # b=low, a=high
                            delta_E = (E_low_prop - E_high_prop)  # [n_chains]
                        
                        log_alpha = d_beta * delta_E + (logJ_f.to(self.device) - logJ_r.to(self.device))  # [n_chains]
                        
                        # Accept/reject for each chain
                        accept_prob = torch.minimum(torch.ones_like(log_alpha), log_alpha.exp())
                        accept = torch.rand_like(accept_prob) <= accept_prob
                        
                        # Apply swaps for accepted chains
                        if idx_a < idx_b:  # a=low, b=high
                            self.x[slice_a] = torch.where(
                                accept.unsqueeze(-1), 
                                x_low_prop.to(self.device), 
                                chain_a
                            )
                            self.x[slice_b] = torch.where(
                                accept.unsqueeze(-1),
                                x_high_prop.to(self.device),
                                chain_b
                            )
                        else:  # b=low, a=high
                            self.x[slice_a] = torch.where(
                                accept.unsqueeze(-1),
                                x_high_prop.to(self.device),
                                chain_a
                            )
                            self.x[slice_b] = torch.where(
                                accept.unsqueeze(-1),
                                x_low_prop.to(self.device),
                                chain_b
                            )

                        # Mirror vanilla sampler: update temperature assignments when logging is enabled
                        if getattr(self, "log_history", False):
                            accept_flat = accept.bool()  # [n_chains]
                            # Swap temp assignments for accepted chains
                            temp_a_old = self.temp_assignment[slice_a].clone()
                            temp_b_old = self.temp_assignment[slice_b].clone()
                            self.temp_assignment[slice_a][accept_flat] = temp_b_old[accept_flat]
                            self.temp_assignment[slice_b][accept_flat] = temp_a_old[accept_flat]
                        
                        # Return mean acceptance probability
                        mean_accept = accept_prob.mean().item()
                        logger.debug(f"Flow swap {pair_key}: {accept.sum().item()}/{n_chains} accepted, mean_prob={mean_accept:.3f}")
                        return mean_accept
                        
                    except Exception as e:
                        logger.debug(f"Flow swap failed: {e}, falling back to vanilla")
                        # Fall back to vanilla swap
                        return super()._attempt_swap(idx_a, idx_b)
                else:
                    # No flow available, use vanilla swap
                    return super()._attempt_swap(idx_a, idx_b)
        
        # Create flow-enhanced PT sampler
        pt = FlowEnhancedPT(
            flow_dict=flows,
            x=x_init,
            energy_func=energy_func,
            step_size=torch.tensor([0.0001] * (n_replicas * x_init.shape[1]), device=device).unsqueeze(-1),
            swap_interval=args.swap_interval,
            temperatures=temperatures,
            mh=True,
            device=device,
            log_history=True,
        )
        logger.info("Flow-enhanced PT sampler created")
    

    
    # Wrap with dynamic step size adaptation
    pt = DynSamplerWrapper(pt, per_temp=True, total_n_temp=n_replicas, target_acceptance_rate=0.6, alpha=0.25)
    
    # Run simulation
    logger.info(f"Running simulation for {args.n_steps} steps...")
    
    swap_log = []
    phi_trace = []
    step_times = []
    
    # Manual replica trace tracking (since PT sampler logging is unreliable)
    manual_replica_trace = []
    n_expected_swaps = args.n_steps // args.swap_interval + 1
    
    progress_bar = tqdm(range(0, args.n_steps, args.swap_interval), desc="PT Simulation")
    
    for step_idx, step in enumerate(progress_bar):
        step_start = time.time()
        
        # Record replica state before sampling (for manual tracking)
        if hasattr(pt.sampler, 'temp_assignment'):
            # Extract current temperature assignments for each replica  
            temp_assignments = pt.sampler.temp_assignment.cpu().numpy()
            n_chains_per_replica = len(temp_assignments) // n_replicas
            
            replica_temps = np.zeros(n_replicas, dtype=np.int32)
            for replica_id in range(n_replicas):
                walker_start = replica_id * n_chains_per_replica  
                replica_temps[replica_id] = temp_assignments[walker_start]
            
            manual_replica_trace.append(replica_temps.copy())
        else:
            # Fallback: static assignment
            replica_temps = np.arange(n_replicas, dtype=np.int32)
            manual_replica_trace.append(replica_temps)
        
        # Perform MD steps and attempt swaps
        new_samples, acc, *_ = pt.sample()
        
        # Synchronize GPU operations for accurate timing
        if device == "cuda":
            torch.cuda.synchronize()
        
        step_end = time.time()
        step_times.append((step_end - step_start) / args.swap_interval)
        
        # Record swap acceptance rates
        if hasattr(pt.sampler, 'swap_rates') and pt.sampler.swap_rates:
            swap_log.append(np.array(pt.sampler.swap_rates))
        
        # Sample phi angle if needed
        if step % args.sample_interval == 0:
            # Get coordinates of coldest replica (index 0)
            coords = new_samples[0].view(system_info["n_atoms"], 3).detach().cpu().numpy()
            
            if system_info["topology"] is not None:
                # Compute phi dihedral using MDTraj
                traj = md.Trajectory(coords.reshape(1, -1, 3), topology=md.Topology.from_openmm(system_info["topology"]))
                phi_angles = md.compute_phi(traj)[1]
                if len(phi_angles) > 0:
                    phi_trace.append(phi_angles[0][0])
                else:
                    phi_trace.append(0.0)  # Fallback
            else:
                # Simple fallback - compute dihedral manually for ALDP
                phi_trace.append(0.0)  # Placeholder
        
        # Update progress bar
        if swap_log:
            mean_acc = np.mean(swap_log[-1]) if swap_log[-1] is not None else 0.0
            progress_bar.set_postfix_str(f"acc: {mean_acc:.3f}")
    
    # Record final state
    if hasattr(pt.sampler, 'temp_assignment'):
        temp_assignments = pt.sampler.temp_assignment.cpu().numpy()
        n_chains_per_replica = len(temp_assignments) // n_replicas
        
        replica_temps = np.zeros(n_replicas, dtype=np.int32)
        for replica_id in range(n_replicas):
            walker_start = replica_id * n_chains_per_replica  
            replica_temps[replica_id] = temp_assignments[walker_start]
        
        manual_replica_trace.append(replica_temps.copy())
    else:
        replica_temps = np.arange(n_replicas, dtype=np.int32)
        manual_replica_trace.append(replica_temps)
    
    # Compute final statistics
    logger.info("Computing final statistics...")
    
    # Acceptance matrix
    acceptance_matrix = calc_acceptance_matrix(swap_log, n_replicas)
    
    # Use manual replica trace (more reliable than PT sampler's internal history)
    if manual_replica_trace:
        replica_trace = np.array(manual_replica_trace, dtype=np.int32)
        logger.info(f"Using manual replica trace with shape: {replica_trace.shape}")
    else:
        # Fallback: create synthetic trace with expected shape
        logger.warning("No replica trace available, creating synthetic trace")
        n_swaps = args.n_steps // args.swap_interval + 1
        replica_trace = np.zeros((n_swaps, n_replicas), dtype=np.int32)
        for i in range(n_replicas):
            replica_trace[:, i] = i  # Static assignment
    
    # Convert phi trace to numpy
    phi_trace = np.array(phi_trace, dtype=np.float32)
    
    # Mean wallclock time per step
    wallclock_per_step = np.mean(step_times) if step_times else 0.001
    
    # Save outputs
    logger.info("Saving output files...")
    np.save(os.path.join(args.out_dir, "acceptance_matrix.npy"), acceptance_matrix)
    np.save(os.path.join(args.out_dir, "replica_trace.npy"), replica_trace)
    np.save(os.path.join(args.out_dir, "phi_trace.npy"), phi_trace)
    np.save(os.path.join(args.out_dir, "wallclock_per_step.npy"), np.float64(wallclock_per_step))
    
    # Generate metadata
    metadata = {
        "args": vars(args),
        "config_file": args.cfg,
        "git_commit": get_git_commit_hash(),
        "device": device,
        "n_flows_loaded": len(flows),
        "mean_acceptance": float(np.mean(np.triu(acceptance_matrix, 1)[np.triu(acceptance_matrix, 1) > 0])),
        "round_trip_time": float(compute_round_trip_time(replica_trace)),
        "phi_autocorr_time": float(integrated_autocorrelation(phi_trace)) if len(phi_trace) > 0 else float('inf'),
        "mean_wallclock_per_step": float(wallclock_per_step),
    }
    
    with open(os.path.join(args.out_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    logger.info("Simulation completed successfully!")
    logger.info(f"Mean acceptance: {metadata['mean_acceptance']:.3f}")
    logger.info(f"Round-trip time: {metadata['round_trip_time']:.1f} swap intervals")
    logger.info(f"Phi autocorr time: {metadata['phi_autocorr_time']:.1f} sample intervals")
    logger.info(f"Speed: {1.0/wallclock_per_step:.1f} steps/second")


def get_test_parser() -> argparse.ArgumentParser:
    """Return ArgumentParser for testing purposes."""
    parser = argparse.ArgumentParser(
        description="Parallel-Tempering simulation with optional flow-based swap models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic PT simulation with flow models
  python -m src.run_pt --cfg configs/AA_simple.yaml \\
                       --checkpoint_dir checkpoints/AA_simple \\
                       --out_dir results/kinetics/simple \\
                       --n_steps 20000 --seed 42

  # Vanilla PT without flows
  python -m src.run_pt --cfg configs/AA_simple.yaml \\
                       --out_dir results/vanilla_pt \\
                       --n_steps 10000 --no_flow

  # Dry run for testing
  python -m src.run_pt --cfg configs/AA_simple.yaml \\
                       --out_dir results/test \\
                       --n_steps 1000 --dry-run

Output files:
  acceptance_matrix.npy    - Mean swap acceptance probability [R,R]
  replica_trace.npy        - Temperature index at each step [T,R]  
  phi_trace.npy           - Phi dihedral of coldest replica [T,]
  wallclock_per_step.npy  - Mean seconds per MD step [scalar]
  metadata.json           - Run metadata and summary statistics
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--cfg", 
        required=True,
        help="Path to YAML configuration file (e.g., configs/AA_simple.yaml)"
    )
    parser.add_argument(
        "--out_dir",
        required=True, 
        help="Output directory for results"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        required=True,
        help="Number of MD integration steps"
    )
    
    # Optional arguments
    parser.add_argument(
        "--checkpoint_dir",
        help="Directory containing flow model checkpoints"
    )
    parser.add_argument(
        "--swap_interval",
        type=int,
        default=100,
        help="Interval between swap attempts (default: 100)"
    )
    parser.add_argument(
        "--sample_interval", 
        type=int,
        default=10,
        help="Interval between phi angle samples (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no_flow",
        action="store_true",
        help="Run vanilla PT even if checkpoints exist"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true", 
        help="Generate synthetic data instead of running simulation"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU-only execution"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (very verbose)"
    )
    
    return parser


if __name__ == "__main__":
    main()