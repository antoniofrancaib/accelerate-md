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