import os
import yaml
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

__all__ = [
    "load_config",
    "save_config",
    "setup_device",
    "print_config_summary",
    "setup_output_directories",
    "get_temperature_pairs",
    "get_model_config",
    "get_data_config",
    "get_training_config",
    "create_run_config",
    "get_energy_threshold",
    "set_openmm_threads",
]


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries (override wins)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# -----------------------------------------------------------------------------
# YAML I/O
# -----------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file and attach helper metadata.

    Args:
        path: Path to YAML file.

    Returns:
        A dictionary with the parsed configuration. The field `_config_path` is
        injected so downstream functions can locate the original YAML for
        provenance tracking.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    cfg["_config_path"] = os.path.abspath(path)

    # ------------------------------------------------------------------
    # Auto-fill dataset paths & model.num_atoms from `peptide_code`
    # ------------------------------------------------------------------
    if "peptide_code" in cfg:
        _autofill_from_peptide(cfg)

    # Apply system-level environment tweaks (e.g. OpenMM CPU thread count)
    set_openmm_threads(cfg)

    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Persist configuration dictionary as YAML (drops private keys)."""
    cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg_clean, fh, sort_keys=False)


# -----------------------------------------------------------------------------
# Convenience getters
# -----------------------------------------------------------------------------

def get_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("model", {})


def get_data_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pt_data_path": cfg["data"].get("pt_data_path"),
        "topology_path": cfg["data"].get("molecular_data_path"),
        "subsample_rate": cfg["data"].get("subsample_rate", 100),
    }


def get_training_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("training", {})


# -----------------------------------------------------------------------------
# Device & output directories
# -----------------------------------------------------------------------------

def setup_device(cfg: Dict[str, Any]) -> str:
    """Select compute device based on cfg["device"] (auto/cpu/cuda)."""
    requested = cfg.get("device", "auto")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        import torch
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but not available.")
    # auto
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_output_directories(cfg: Dict[str, Any]) -> None:
    base_dir = Path(cfg["output"]["base_dir"]).expanduser()
    experiment_dir = base_dir / cfg["experiment_name"]
    # Standard sub-dirs
    for sub in ["models", "logs", "plots", "metrics"]:
        (experiment_dir / sub).mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Temperature helpers
# -----------------------------------------------------------------------------

def get_temperature_pairs(cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return list of index pairs indicating adjacent temperatures to train."""
    return [tuple(pair) for pair in cfg["temperature_pairs"]]


# -----------------------------------------------------------------------------
# Per-run config (temperature-pair specific)
# -----------------------------------------------------------------------------

def create_run_config(cfg: Dict[str, Any], pair: Tuple[int, int], device: str) -> Dict[str, Any]:
    """Return a cloned config dict specialised for a single temperature pair.

    Adjusts output directories to live under
    `outputs/<experiment>/pair_<low>_<high>/` so that checkpoints and logs are
    neatly separated.
    """
    run_cfg: Dict[str, Any] = _deep_update({}, cfg)  # shallow copy
    run_cfg["temp_pair"] = pair
    run_cfg["device"] = device

    low, high = pair
    pair_dir_name = f"pair_{low}_{high}"
    base_dir = Path(cfg["output"]["base_dir"]).expanduser()
    run_cfg["output"] = {
        "base_dir": str(base_dir),
        "pair_dir": str(base_dir / cfg["experiment_name"] / pair_dir_name),
    }
    # create directories
    for sub in ["models", "logs", "plots", "metrics"]:
        (Path(run_cfg["output"]["pair_dir"]) / sub).mkdir(parents=True, exist_ok=True)

    return run_cfg


# -----------------------------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------------------------

def print_config_summary(cfg: Dict[str, Any]) -> None:
    import pprint
    print("\nCONFIG SUMMARY\n--------------")
    pprint.pprint({k: v for k, v in cfg.items() if not k.startswith("_")})


# -----------------------------------------------------------------------------
# System helpers (energy threshold, OpenMM env)
# -----------------------------------------------------------------------------

def get_energy_threshold(cfg: Dict[str, Any]) -> float | None:
    """Return a global energy threshold for batch clipping.

    Priority order:
    1. `system.energy_max` if present.
    2. `system.energy_cut` as fallback.
    Returns `None` if neither key exists.
    """
    sys_cfg = cfg.get("system", {})
    if sys_cfg.get("energy_max") is not None:
        return float(sys_cfg["energy_max"])
    if sys_cfg.get("energy_cut") is not None:
        return float(sys_cfg["energy_cut"])
    return None


def set_openmm_threads(cfg: Dict[str, Any]):
    """Set OPENMM_CPU_THREADS env var if `system.n_threads` is configured."""
    sys_cfg = cfg.get("system", {})
    n_threads = sys_cfg.get("n_threads")
    if n_threads is not None and n_threads > 0:
        os.environ.setdefault("OPENMM_CPU_THREADS", str(int(n_threads)))


# -----------------------------------------------------------------------------
# Peptide helper – infer paths & atom count
# -----------------------------------------------------------------------------

def _autofill_from_peptide(cfg: Dict[str, Any]):
    """Populate data paths and `model.num_atoms` given `cfg['peptide_code']`."""
    code = cfg["peptide_code"].strip()
    base_dir = f"data/pt_dipeptides/{code}"

    data_cfg = cfg.setdefault("data", {})
    data_cfg.setdefault("pt_data_path", f"{base_dir}/pt_{code}.pt")
    data_cfg.setdefault("molecular_data_path", base_dir)

    # Infer atom count if not set in model section.
    model_cfg = cfg.setdefault("model", {})
    atom_file = Path(base_dir) / "atom_types.pt"
    try:
        import torch
        if atom_file.is_file():
            atom_types = torch.load(atom_file, map_location="cpu")
            n_atoms = int(atom_types.shape[0])
            if model_cfg.get("num_atoms") != n_atoms:
                model_cfg["num_atoms"] = n_atoms
    except Exception:  # pragma: no cover – fallback silently
        pass

    # ------------------------------------------------------------------
    # Target defaults – Aldp for AX; generic dipeptide otherwise.
    # ------------------------------------------------------------------
    target_cfg = cfg.setdefault("target", {})
    if "name" not in target_cfg:
        if code.upper() == "AX":
            target_cfg["name"] = "aldp"
        else:
            target_cfg["name"] = "dipeptide"
            # All peptides reside in the 2AA-1-big dataset directory.
            pdb_guess = f"data/timewarp/2AA-1-big/train/{code}-traj-state0.pdb"
            target_cfg.setdefault("kwargs", {})
            target_cfg["kwargs"].setdefault("pdb_path", pdb_guess)
            target_cfg["kwargs"].setdefault("env", "implicit") 