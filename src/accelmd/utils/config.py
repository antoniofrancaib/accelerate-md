"""
Configuration utilities for loading and merging YAML configuration files.
"""

import yaml


def load_config(config_path: str):
    """
    Load a single YAML config file and return the parsed dictionary.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the parsed configuration
    """
    with open(config_path, 'r') as f:
        config_part = yaml.safe_load(f)
    
    # ------------------------------------------------------------------
    #  New: ensure a valid `model_type` key is present for flow selection
    # ------------------------------------------------------------------
    if isinstance(config_part, dict):
        config_part.setdefault("model_type", "realnvp")
        if config_part["model_type"] not in ("realnvp", "tarflow"):
            raise ValueError(f"Unsupported model_type {config_part['model_type']}. Expected 'realnvp' or 'tarflow'.")
    return config_part or {}


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