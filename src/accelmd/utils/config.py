"""
Configuration utilities for loading and merging YAML configuration files.
"""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Load a single YAML config file and return the parsed dictionary.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the parsed configuration
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # 1) model_type default + validation
    cfg.setdefault("model_type", "realnvp")
    if cfg["model_type"] not in ("realnvp", "tarflow"):
        raise ValueError(f"Unsupported model_type {cfg['model_type']}")

    # 2) derive output paths
    cfg = derive_output_paths(cfg)

    # 3) default metric template
    cfg["output"].setdefault(
        "metric_template", "swap_rate_flow_${t_low}_to_${t_high}.json"
    )

    # 4) compute metric_json from template
    t_low, t_high = float(cfg["pt"]["temp_low"]), float(cfg["pt"]["temp_high"])
    filename = (
        cfg["output"]["metric_template"]
        .replace("${t_low}",  f"{t_low:.2f}")
        .replace("${t_high}", f"{t_high:.2f}")
    )
    cfg["output"]["metric_json"] = str(
        Path(cfg["output"]["results_dir"]) / filename
    )

    return cfg


def merge_dicts(base: dict, other: dict):
    """
    Recursively merge *other* into *base* (in-place) and return *base*.
    
    Args:
        base: Base dictionary to merge into
        other: Dictionary to merge from
        
    Returns:
        The merged base dictionary
    """
    for key, value in other.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def derive_output_paths(cfg: dict) -> dict:
    """
    Expand ${name}, ${t_low}, ${t_high} tokens and attach
    a complete 'output' block under outputs/<name>/…
    """
    name    = cfg["name"]
    pt_cfg  = cfg["pt"]
    base    = Path(cfg.get("output", {}).get("base_dir", "outputs")) / name

    paths = {
        "base_dir":    str(base),
        "checkpoints": str(base/"checkpoints"),
        "plots_dir":   str(base/"plots"),
        "results_dir": str(base/"results"),
        "logs_dir":    str(base/"logs"),
        "log_file":    str(base/"logs"/"experiment.log"),
        "model_path":  str(base/"model.pt"),
        "config_copy": str(base/"config.yaml"),
    }

    cfg["output"] = {**paths, **cfg.get("output", {})}
    return cfg 