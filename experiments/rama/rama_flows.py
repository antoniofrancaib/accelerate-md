#!/usr/bin/env python3
"""
Generate 2x2 Ramachandran plots using transformer flows for one or more peptides.

Per peptide, the 2x2 subplots are:
- Top-left: original T0 samples
- Top-right: original T3 samples mapped to T0 via inverse flows (3→2→1→0)
- Bottom-left: original T3 samples
- Bottom-right: original T0 samples mapped to T3 via forward flows (0→1→2→3)

Defaults assume the multi-transformer MCMC setup and dataset layout under
datasets/pt_dipeptides/<PEP>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import yaml
import mdtraj as md
import matplotlib.pyplot as plt

# Project-root imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.accelmd.flows import PTSwapTransformerFlow  # type: ignore
from src.accelmd.flows.transformer_block import TransformerConfig  # type: ignore
from src.accelmd.flows.rff_position_encoder import RFFPositionEncoderConfig  # type: ignore
from src.accelmd.utils.plot_utils import plot_Ramachandran  # type: ignore
from src.run_pt import get_atom_vocab_indices  # type: ignore


def load_used_config(config_path: Path) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_experiments_cfg(experiments_cfg_path: Path) -> Dict:
    with open(experiments_cfg_path, "r") as f:
        return yaml.safe_load(f)


def resolve_transformer_checkpoint_map(exp_cfg: Dict) -> Dict[str, str]:
    """Return a mapping like {"pair_0_1": path, ...} for transformer checkpoints.

    Supports either:
      checkpoints.transformer.pair_0_1 = path
    or nested under a single subkey like:
      checkpoints.transformer.multi_peptide.pair_0_1 = path
    """
    ck_root = exp_cfg.get("checkpoints", {}).get("transformer", {})
    # Direct mapping
    if any(k.startswith("pair_") for k in ck_root.keys()):
        return {k: v for k, v in ck_root.items() if k.startswith("pair_") and v}
    # Nested one level (pick the first dict)
    for _, sub in ck_root.items():
        if isinstance(sub, dict) and any(k.startswith("pair_") for k in sub.keys()):
            return {k: v for k, v in sub.items() if k.startswith("pair_") and v}
    return {}


def build_transformer_flow_for_pair(
    used_cfg: Dict,
    pair: Tuple[int, int],
    device: str,
    pdb_path: Path,
) -> PTSwapTransformerFlow:
    temps: List[float] = used_cfg.get("temperatures", {}).get("values", [300.0, 1500.0])
    temp_low, temp_high = float(temps[pair[0]]), float(temps[pair[1]])

    target_cfg = used_cfg.get("target", {})
    target_name = target_cfg.get("name", "dipeptide")
    target_kwargs = dict(target_cfg.get("kwargs", {}))
    # Always inject PDB path for dipeptide target to ensure correct peptide
    if target_name == "dipeptide":
        target_kwargs["pdb_path"] = str(pdb_path)

    # Read transformer hyperparameters from training-style location
    model_cfg = used_cfg.get("model", {})
    transformer_cfg = model_cfg.get("transformer", used_cfg.get("transformer", {}))

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
        num_layers=model_cfg.get("flow_layers", 8),
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

    return flow


@torch.no_grad()
def apply_inverse_flow_sequence(
    coords_bxn3: torch.Tensor,
    atom_types_n: torch.LongTensor,
    flow_ckpts: List[Tuple[Tuple[int, int], Path]],
    used_cfg: Dict,
    pdb_path: Path,
    device: str = "cpu",
    batch_size: int = 128,
) -> torch.Tensor:
    device_t = torch.device(device)
    coords = coords_bxn3.to(device_t)
    atom_types_b = None

    for pair, ckpt in flow_ckpts:
        flow = build_transformer_flow_for_pair(used_cfg, pair, device, pdb_path)
        state_dict = torch.load(str(ckpt), map_location=device_t, weights_only=True)
        flow.load_state_dict(state_dict)
        flow = flow.to(device_t).eval()

        outputs: List[torch.Tensor] = []
        for start in range(0, coords.shape[0], batch_size):
            end = min(start + batch_size, coords.shape[0])
            batch = coords[start:end]
            if atom_types_b is None or atom_types_b.shape[0] != (end - start):
                atom_types_b = atom_types_n.to(device_t).unsqueeze(0).expand(end - start, -1)
            # Deterministic, position-only mapping with no padding
            masked = torch.zeros((end - start, atom_types_b.shape[1]), dtype=torch.bool, device=device_t)
            zeros_vel = torch.zeros_like(batch)
            mapped, _ = flow.inverse(
                batch,
                atom_types=atom_types_b,
                velocities=zeros_vel,
                masked_elements=masked,
            )
            outputs.append(mapped)
        coords = torch.cat(outputs, dim=0)

    return coords.cpu()


@torch.no_grad()
def apply_forward_flow_sequence(
    coords_bxn3: torch.Tensor,
    atom_types_n: torch.LongTensor,
    flow_ckpts: List[Tuple[Tuple[int, int], Path]],
    used_cfg: Dict,
    pdb_path: Path,
    device: str = "cpu",
    batch_size: int = 128,
) -> torch.Tensor:
    device_t = torch.device(device)
    coords = coords_bxn3.to(device_t)
    atom_types_b = None

    for pair, ckpt in flow_ckpts:
        flow = build_transformer_flow_for_pair(used_cfg, pair, device, pdb_path)
        state_dict = torch.load(str(ckpt), map_location=device_t, weights_only=True)
        flow.load_state_dict(state_dict)
        flow = flow.to(device_t).eval()

        outputs: List[torch.Tensor] = []
        for start in range(0, coords.shape[0], batch_size):
            end = min(start + batch_size, coords.shape[0])
            batch = coords[start:end]
            if atom_types_b is None or atom_types_b.shape[0] != (end - start):
                atom_types_b = atom_types_n.to(device_t).unsqueeze(0).expand(end - start, -1)
            # Deterministic, position-only mapping with no padding
            masked = torch.zeros((end - start, atom_types_b.shape[1]), dtype=torch.bool, device=device_t)
            zeros_vel = torch.zeros_like(batch)
            mapped, _ = flow.forward(
                batch,
                atom_types=atom_types_b,
                velocities=zeros_vel,
                masked_elements=masked,
                return_log_det=True,
            )
            outputs.append(mapped)
        coords = torch.cat(outputs, dim=0)

    return coords.cpu()


def compute_phi_psi_all(coords_bxn3: np.ndarray, pdb_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    traj = md.Trajectory(coords_bxn3, topology=md.load(str(pdb_path)).topology)
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    phi_vec = phi[:, 0] if phi.shape[1] > 0 else np.zeros(len(coords_bxn3))
    psi_vec = psi[:, 0] if psi.shape[1] > 0 else np.zeros(len(coords_bxn3))
    return phi_vec, psi_vec


def process_one_peptide(peptide: str, args: argparse.Namespace) -> Path:
    peptide = peptide.upper()
    device = "cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda"

    # Resolve dataset and PDB
    dataset_path = Path(args.dataset) if args.dataset else (PROJECT_ROOT / f"datasets/pt_dipeptides/{peptide}/pt_{peptide}.pt")
    pdb_path = Path(args.pdb) if args.pdb else (PROJECT_ROOT / f"datasets/pt_dipeptides/{peptide}/ref.pdb")

    data = torch.load(dataset_path, map_location="cpu")
    if not isinstance(data, torch.Tensor):
        raise RuntimeError(f"Expected a torch.Tensor at {dataset_path}, got {type(data)}")

    # Load experiments config (temperature ladder and checkpoints)
    exp_cfg = load_experiments_cfg(Path(args.experiments_cfg))
    temps: List[float] = exp_cfg.get("temperatures", {}).get("values", [])
    if not temps:
        raise RuntimeError("temperatures.values missing in experiments config")
    n_temp = len(temps)
    cold_idx, hot_idx = 0, n_temp - 1

    # Expect dataset shape [num_temp, num_chains, num_steps, dim]
    # Flatten chains and steps for sampling
    coords_low_all = data[cold_idx]    # [num_chains, num_steps, dim]
    coords_high_all = data[hot_idx]    # [num_chains, num_steps, dim]
    num_chains, num_steps, dim = coords_low_all.shape
    coords_low = coords_low_all.reshape(num_chains * num_steps, dim)
    coords_high = coords_high_all.reshape(num_chains * num_steps, dim)
    if args.subsample and args.subsample > 0:
        steps = min(args.subsample, coords_low.shape[0], coords_high.shape[0])
        idx_high = torch.linspace(0, coords_high.shape[0] - 1, steps=steps).long()
        idx_low = torch.linspace(0, coords_low.shape[0] - 1, steps=steps).long()
        coords_high = coords_high.index_select(0, idx_high)
        coords_low = coords_low.index_select(0, idx_low)

    n_atoms = md.load(str(pdb_path)).n_atoms
    dimd = coords_high.shape[-1]
    if dimd != n_atoms * 3:
        raise ValueError(f"Dim mismatch for {peptide}: dataset dim={dimd} but PDB atoms*3={n_atoms*3}")
    coords_high_bxn3 = coords_high.view(-1, n_atoms, 3).contiguous()
    coords_low_bxn3 = coords_low.view(-1, n_atoms, 3).contiguous()

    atom_types = get_atom_vocab_indices(str(pdb_path))
    # Load training config to build flows; override temperatures with experiments ladder
    used_cfg = load_used_config(Path(args.used_config))
    if "temperatures" not in used_cfg:
        used_cfg["temperatures"] = {}
    used_cfg["temperatures"]["values"] = temps

    # Resolve checkpoints for available pairs based on ladder length
    ck_map = resolve_transformer_checkpoint_map(exp_cfg)
    # Build pair lists dynamically
    pair_indices = [(i, i + 1) for i in range(n_temp - 1)]
    # Map to paths; skip pairs without a checkpoint
    def pair_key(i, j):
        return f"pair_{i}_{j}"

    seq_fwd = []
    for i, j in pair_indices:
        path = ck_map.get(pair_key(i, j))
        if path:
            seq_fwd.append(((i, j), PROJECT_ROOT / path))
    seq_inv = list(reversed(seq_fwd))  # reverse order for inverse mapping

    coords_3to0 = apply_inverse_flow_sequence(
        coords_bxn3=coords_high_bxn3,
        atom_types_n=atom_types,
        flow_ckpts=seq_inv,
        used_cfg=used_cfg,
        pdb_path=pdb_path,
        device=device,
        batch_size=args.batch,
    )
    coords_0to3 = apply_forward_flow_sequence(
        coords_bxn3=coords_low_bxn3,
        atom_types_n=atom_types,
        flow_ckpts=seq_fwd,
        used_cfg=used_cfg,
        pdb_path=pdb_path,
        device=device,
        batch_size=args.batch,
    )

    phi0, psi0 = compute_phi_psi_all(coords_low_bxn3.numpy(), pdb_path)
    phi3, psi3 = compute_phi_psi_all(coords_high_bxn3.numpy(), pdb_path)
    phi3to0, psi3to0 = compute_phi_psi_all(coords_3to0.numpy(), pdb_path)
    phi0to3, psi0to3 = compute_phi_psi_all(coords_0to3.numpy(), pdb_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{peptide}_rama_flows_2x2.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    plot_Ramachandran(axes[0, 0], phi0, psi0)
    axes[0, 0].set_title(f"{peptide}: Original T0", fontsize=12)
    plot_Ramachandran(axes[0, 1], phi3to0, psi3to0)
    axes[0, 1].set_title(f"{peptide}: T3 → T0 (flows)", fontsize=12)
    plot_Ramachandran(axes[1, 0], phi3, psi3)
    axes[1, 0].set_title(f"{peptide}: Original T3", fontsize=12)
    plot_Ramachandran(axes[1, 1], phi0to3, psi0to3)
    axes[1, 1].set_title(f"{peptide}: T0 → T3 (flows)", fontsize=12)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved Ramachandran plot to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="2x2 Ramachandran via transformer flows for one or more peptides")
    parser.add_argument("--peptide", nargs="+", default=["AS"], help="One or more peptide codes, e.g. --peptide AK KA KK")
    parser.add_argument("--dataset", help="Override dataset path (pt_<PEP>.pt); used only when single peptide provided")
    parser.add_argument("--pdb", help="Override PDB path (ref.pdb); used only when single peptide provided")
    parser.add_argument("--experiments_cfg", default=str(PROJECT_ROOT / "configs/experiments.yaml"))
    parser.add_argument("--used_config", default=str(PROJECT_ROOT / "configs/multi_transformer.yaml"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "experiments/rama"))
    parser.add_argument("--subsample", type=int, default=2000, help="Number of frames to subsample per peptide (0=all)")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for flow mapping")
    parser.add_argument("--cpu_only", action="store_true")
    args = parser.parse_args()

    for pep in args.peptide:
        process_one_peptide(pep, args)


if __name__ == "__main__":
    main()

