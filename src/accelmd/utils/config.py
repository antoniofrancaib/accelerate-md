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
    "get_training_peptides",
    "get_eval_peptides",
    "is_multi_peptide_mode",
    "get_multi_batching_mode",
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
    # Validate and process multi-peptide configuration
    # ------------------------------------------------------------------
    _validate_multi_peptide_config(cfg)
    
    # ------------------------------------------------------------------
    # Auto-fill dataset paths & model.num_atoms from `peptide_code` or multi-peptide mode
    # ------------------------------------------------------------------
    mode = cfg.get("mode", "single")
    if mode == "single":
        if "peptide_code" in cfg:
            _autofill_from_peptide(cfg)
    elif mode == "multi":
        # Multi-peptide mode handled separately in data loading
        # Validation already done by _validate_multi_peptide_config
        pass

    # Apply system-level environment tweaks (e.g. OpenMM CPU thread count)
    set_openmm_threads(cfg)

    return cfg


def _validate_multi_peptide_config(cfg: Dict[str, Any]) -> None:
    """Validate multi-peptide configuration parameters."""
    mode = cfg.get("mode", "single")
    
    if mode not in ["single", "multi"]:
        raise ValueError(f"mode must be 'single' or 'multi', got '{mode}'")
    
    if mode == "multi":
        # Check required peptides configuration
        if "peptides" not in cfg:
            raise ValueError("'peptides' key is required when mode == 'multi'")
        
        peptides_cfg = cfg["peptides"]
        if not isinstance(peptides_cfg, dict):
            raise ValueError("'peptides' must be a dictionary")
        
        if "train" not in peptides_cfg:
            raise ValueError("'peptides.train' is required when mode == 'multi'")
        
        train_peptides = peptides_cfg["train"]
        if not isinstance(train_peptides, list) or len(train_peptides) == 0:
            raise ValueError("'peptides.train' must be a non-empty list")
        
        # Validate eval peptides (optional, defaults to train)
        eval_peptides = peptides_cfg.get("eval", train_peptides)
        if not isinstance(eval_peptides, list) or len(eval_peptides) == 0:
            raise ValueError("'peptides.eval' must be a non-empty list if specified")
        
        # Store normalized eval peptides
        cfg["peptides"]["eval"] = eval_peptides
        
        # Check architecture compatibility
        model_cfg = cfg.get("model", {})
        architecture = model_cfg.get("architecture", "simple")
        if architecture == "simple":
            raise ValueError("Simple architecture cannot be used in multi-peptide mode")
        
        # Validate multi_mode configuration
        multi_mode_cfg = cfg.get("multi_mode", {})
        batching = multi_mode_cfg.get("batching", "padding")
        if batching not in ["padding", "uniform"]:
            raise ValueError(f"multi_mode.batching must be 'padding' or 'uniform', got '{batching}'")
        
        # Set defaults
        cfg.setdefault("multi_mode", {})["batching"] = batching
        
    elif mode == "single":
        # In single mode, peptides and multi_mode should not be used
        if "peptides" in cfg:
            import warnings
            warnings.warn("'peptides' key ignored in single mode")
        if "multi_mode" in cfg:
            import warnings
            warnings.warn("'multi_mode' key ignored in single mode")


def get_training_peptides(cfg: Dict[str, Any]) -> List[str]:
    """Get list of peptides for training based on mode."""
    mode = cfg.get("mode", "single")
    if mode == "single":
        return [cfg["peptide_code"]]
    elif mode == "multi":
        return cfg["peptides"]["train"]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_eval_peptides(cfg: Dict[str, Any]) -> List[str]:
    """Get list of peptides for evaluation based on mode."""
    mode = cfg.get("mode", "single")
    if mode == "single":
        return [cfg["peptide_code"]]
    elif mode == "multi":
        return cfg["peptides"]["eval"]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def is_multi_peptide_mode(cfg: Dict[str, Any]) -> bool:
    """Check if configuration is in multi-peptide mode."""
    return cfg.get("mode", "single") == "multi"


def get_multi_batching_mode(cfg: Dict[str, Any]) -> str:
    """Get batching mode for multi-peptide training."""
    if not is_multi_peptide_mode(cfg):
        raise ValueError("Not in multi-peptide mode")
    return cfg.get("multi_mode", {}).get("batching", "padding")


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
    """Populate data paths from `cfg['peptide_code']`.
    
    Note: Target selection is now handled directly in main.py based on peptide_code.
    """
    code = cfg["peptide_code"].strip()
    base_dir = f"datasets/pt_dipeptides/{code}"

    data_cfg = cfg.setdefault("data", {})
    data_cfg.setdefault("pt_data_path", f"{base_dir}/pt_{code}.pt")
    data_cfg.setdefault("molecular_data_path", base_dir)