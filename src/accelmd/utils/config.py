"""
Configuration utilities for loading and merging YAML configuration files.
"""

import yaml
from pathlib import Path
import logging
import re
from typing import Dict, Any, Optional

from src.accelmd.kernels.local.langevin import Langevin
from src.accelmd.kernels.swap.vanilla import VanillaSwap
from src.accelmd.kernels.swap.realnvp import RealNVPSwap
from src.accelmd.kernels.swap.tarflow import TarFlowSwap


def _substitute_variables(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively substitute ${variable} patterns in config values.
    
    Args:
        config: Configuration dictionary to process
        context: Context variables for substitution
        
    Returns:
        Configuration with substituted values
    """
    def substitute_value(value):
        if isinstance(value, str):
            # Find all ${var} or ${var.subkey} or ${var:format} patterns
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            
            for match in matches:
                # Handle format specifications like pt.temp_low:.2f
                if ':' in match:
                    var_path, format_spec = match.split(':', 1)
                else:
                    var_path = match
                    format_spec = None
                
                # Handle nested keys like pt.step_size
                keys = var_path.split('.')
                replacement = context
                try:
                    for key in keys:
                        replacement = replacement[key]
                    
                    # Apply format specification if present
                    if format_spec:
                        if isinstance(replacement, (int, float)):
                            replacement = f"{replacement:{format_spec}}"
                        else:
                            replacement = str(replacement)
                    else:
                        replacement = str(replacement)
                    
                    value = value.replace(f'${{{match}}}', replacement)
                except (KeyError, TypeError, ValueError):
                    # Variable not found or format error, leave as is
                    pass
            return value
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value
    
    return substitute_value(config)


def build_local_kernel(cfg: Dict[str, Any]) -> Optional[Any]:
    """Build a LocalKernel from configuration.
    
    Args:
        cfg: Full configuration dictionary
        
    Returns:
        LocalKernel instance or None if not configured
    """
    kernel_cfg = cfg.get("local_kernel")
    if not kernel_cfg:
        return None
    
    # Re-apply variable substitution with current config values
    # This ensures consistency across temperature pairs
    kernel_cfg = _substitute_variables(kernel_cfg, cfg)
        
    kernel_type = kernel_cfg.get("type", "").lower()
    
    if kernel_type == "langevin":
        # Convert string values to proper types
        step_size = kernel_cfg.get("step_size", 1e-4)
        if isinstance(step_size, str):
            step_size = float(step_size)
            
        return Langevin(
            step_size=step_size,
            mh=bool(kernel_cfg.get("mh", True)),
            device=str(kernel_cfg.get("device", "cpu"))
        )
    else:
        logging.getLogger(__name__).warning(f"Unknown local kernel type: {kernel_type}")
        return None


def build_swap_kernel(cfg: Dict[str, Any]) -> Optional[Any]:
    """Build a SwapKernel from configuration.
    
    Args:
        cfg: Full configuration dictionary
        
    Returns:
        SwapKernel instance or None if not configured
    """
    kernel_cfg = cfg.get("swap_kernel")
    if not kernel_cfg:
        return None
    
    # Re-apply variable substitution with current config values
    # This is crucial for handling multiple temperature pairs correctly
    kernel_cfg = _substitute_variables(kernel_cfg, cfg)
        
    kernel_type = kernel_cfg.get("type", "").lower()
    device = str(kernel_cfg.get("device", "cpu"))
    
    if kernel_type == "vanilla":
        return VanillaSwap()
    
    elif kernel_type == "realnvp":
        flow_checkpoint = kernel_cfg.get("flow_checkpoint")
        if not flow_checkpoint:
            logging.getLogger(__name__).warning("RealNVP swap kernel requires flow_checkpoint")
            if kernel_cfg.get("fallback_to_vanilla", False):
                logging.getLogger(__name__).info("Falling back to vanilla swap kernel")
                return VanillaSwap()
            return None
            
        checkpoint_path = Path(flow_checkpoint)
        if not checkpoint_path.exists():
            if kernel_cfg.get("fallback_to_vanilla", False):
                logging.getLogger(__name__).warning(
                    f"Flow checkpoint not found: {checkpoint_path}. Falling back to vanilla swap."
                )
                return VanillaSwap()
            else:
                logging.getLogger(__name__).warning(f"Flow checkpoint not found: {checkpoint_path}")
                return None
        
        # Extract model config from training configuration for consistency
        model_config = None
        trainer_cfg = cfg.get("trainer", {}).get("realnvp", {}).get("model", {})
        if trainer_cfg:
            # Get target to determine dimension
            target_cfg = cfg.get("target", {})
            target_type = target_cfg.get("type", "")
            
            # Determine dimension based on target type
            if target_type == "aldp":
                # ALDP uses 66-dimensional cartesian coordinates
                dim = 66
            elif target_type == "gmm":
                # GMM uses the configured dimension
                gmm_cfg = cfg.get("gmm", {})
                dim = gmm_cfg.get("dim", 2)
            else:
                # Use default or let the checkpoint inference handle it
                dim = None
            
            if dim is not None:
                model_config = {
                    "dim": dim,
                    "hidden_dim": trainer_cfg.get("hidden_dim", 256),
                    "n_couplings": trainer_cfg.get("n_couplings", 14),
                    "use_permutation": trainer_cfg.get("use_permutation", True),
                }
        
        try:
            return RealNVPSwap(
                flow_checkpoint=checkpoint_path,
                model_config=model_config,
                device=device
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load RealNVP swap kernel: {e}")
            if kernel_cfg.get("fallback_to_vanilla", False):
                logging.getLogger(__name__).info("Falling back to vanilla swap kernel")
                return VanillaSwap()
            return None
    
    elif kernel_type == "tarflow":
        flow_checkpoint = kernel_cfg.get("flow_checkpoint")
        if not flow_checkpoint:
            logging.getLogger(__name__).warning("TarFlow swap kernel requires flow_checkpoint")
            if kernel_cfg.get("fallback_to_vanilla", False):
                logging.getLogger(__name__).info("Falling back to vanilla swap kernel")
                return VanillaSwap()
            return None
            
        checkpoint_path = Path(flow_checkpoint)
        if not checkpoint_path.exists():
            if kernel_cfg.get("fallback_to_vanilla", False):
                logging.getLogger(__name__).warning(
                    f"Flow checkpoint not found: {checkpoint_path}. Falling back to vanilla swap."
                )
                return VanillaSwap()
            else:
                logging.getLogger(__name__).warning(f"Flow checkpoint not found: {checkpoint_path}")
                return None
        
        try:
            return TarFlowSwap(
                flow_checkpoint=checkpoint_path,
                device=device
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load TarFlow swap kernel: {e}")
            if kernel_cfg.get("fallback_to_vanilla", False):
                logging.getLogger(__name__).info("Falling back to vanilla swap kernel")
                return VanillaSwap()
            return None
    
    else:
        logging.getLogger(__name__).warning(f"Unknown swap kernel type: {kernel_type}")
        return None


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

    # 4) Initialize temp_low and temp_high for backward compatibility
    # These will be updated by main.py for each temperature pair
    pt_cfg = cfg.get("pt", {})
    temps = pt_cfg.get("temperatures", [1.0, 10.0])
    if len(temps) < 2:
        raise ValueError("pt.temperatures must contain at least 2 temperature values")
    
    # Set initial values from first and last temperatures for compatibility
    cfg["pt"]["temp_low"] = float(temps[0])
    cfg["pt"]["temp_high"] = float(temps[-1])
    
    # Compute initial metric_json from template
    t_low, t_high = cfg["pt"]["temp_low"], cfg["pt"]["temp_high"]
    filename = (
        cfg["output"]["metric_template"]
        .replace("${t_low}",  f"{t_low:.2f}")
        .replace("${t_high}", f"{t_high:.2f}")
    )
    cfg["output"]["metric_json"] = str(
        Path(cfg["output"]["results_dir"]) / filename
    )

    # 5) Ensure temperatures are sorted and stored as list
    cfg["pt"]["temperatures"] = sorted([float(t) for t in temps])
    
    # 6) Variable substitution for configs
    # First apply substitution to the main config
    cfg = _substitute_variables(cfg, cfg)
    
    # Then apply basic variable substitution to kernel configs
    # (temperature-dependent variables will be handled later in main.py)
    if "local_kernel" in cfg:
        cfg["local_kernel"] = _substitute_variables(cfg["local_kernel"], cfg)
    if "swap_kernel" in cfg:
        cfg["swap_kernel"] = _substitute_variables(cfg["swap_kernel"], cfg)

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
    Expand ${name} tokens and attach a complete 'output' block under outputs/<n>/…
    
    Note: Temperature-specific paths will be set later by main.py for each pair.
    """
    name = cfg["name"]
    base = Path(cfg.get("output", {}).get("base_dir", "outputs")) / name
    
    # Basic paths without temperature-specific information
    paths = {
        "base_dir":    str(base),
        "checkpoints": str(base/"checkpoints"),
        "plots_dir":   str(base/"plots"),
        "results_dir": str(base/"results"),
        "log_file":    str(base/"experiment.log"),
        "config_copy": str(base/"config.yaml"),
    }

    cfg["output"] = {**paths, **cfg.get("output", {})}
    return cfg 