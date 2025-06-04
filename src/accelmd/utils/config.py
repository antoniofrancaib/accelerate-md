"""
Configuration utilities for loading and merging YAML configuration files.
Supports only the unified configuration format.
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


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base and return a new dictionary.
    
    Args:
        base: Base dictionary
        override: Override dictionary to merge
        
    Returns:
        New merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


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
    kernel_cfg = _substitute_variables(kernel_cfg, cfg)
        
    kernel_type = kernel_cfg.get("type", "").lower()
    device = str(kernel_cfg.get("device", "cpu"))
    
    if kernel_type == "vanilla":
        return VanillaSwap()
    
    elif kernel_type == "realnvp":
        flow_checkpoint = kernel_cfg.get("flow_checkpoint")
        
        # If checkpoint path not explicitly set or still contains template variables, 
        # generate it dynamically from current temperature pair
        if not flow_checkpoint or "${" in str(flow_checkpoint):
            name = cfg.get("name", "experiment")
            t_low = cfg.get("pt", {}).get("temp_low", 1.0)
            t_high = cfg.get("pt", {}).get("temp_high", 2.0)
            flow_checkpoint = f"outputs/{name}/models/flow_{t_low:.2f}_{t_high:.2f}.pt"
            logging.getLogger(__name__).info(f"Auto-generated flow checkpoint path: {flow_checkpoint}")
        
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
                gmm_cfg = cfg.get("gmm_params", {})
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
        
        # If checkpoint path not explicitly set or still contains template variables, 
        # generate it dynamically from current temperature pair
        if not flow_checkpoint or "${" in str(flow_checkpoint):
            name = cfg.get("name", "experiment")
            t_low = cfg.get("pt", {}).get("temp_low", 1.0)
            t_high = cfg.get("pt", {}).get("temp_high", 2.0)
            flow_checkpoint = f"outputs/{name}/models/flow_{t_low:.2f}_{t_high:.2f}.pt"
            logging.getLogger(__name__).info(f"Auto-generated flow checkpoint path: {flow_checkpoint}")
        
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
    Load the unified YAML config file and return the processed configuration.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the processed configuration
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Process unified configuration format
    cfg = _process_unified_config(cfg)

    # Validate model_type
    cfg.setdefault("model_type", "realnvp")
    if cfg["model_type"] not in ("realnvp", "tarflow"):
        raise ValueError(f"Unsupported model_type {cfg['model_type']}")

    # Derive output paths
    cfg = derive_output_paths(cfg)

    # Default metric template
    cfg["output"].setdefault(
        "metric_template", "swap_rate_flow_${t_low}_to_${t_high}.json"
    )

    # Initialize temp_low and temp_high for processing
    pt_cfg = cfg.get("pt", {})
    temps = pt_cfg.get("temperatures")
    if not temps or len(temps) < 2:
        raise ValueError("pt.temperatures must be explicitly specified with at least 2 temperature values")
    
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

    # Ensure temperatures are sorted and stored as list
    cfg["pt"]["temperatures"] = sorted([float(t) for t in temps])
    
    # Variable substitution for configs
    cfg = _substitute_variables(cfg, cfg)
    
    # Apply variable substitution to kernel configs
    if "local_kernel" in cfg:
        cfg["local_kernel"] = _substitute_variables(cfg["local_kernel"], cfg)
    if "swap_kernel" in cfg:
        cfg["swap_kernel"] = _substitute_variables(cfg["swap_kernel"], cfg)

    return cfg


def derive_output_paths(cfg: dict) -> dict:
    """
    Expand ${name} tokens and attach a complete 'output' block under outputs/<n>/…
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


def _process_unified_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Process unified configuration format by selecting the appropriate experiment type.
    
    Args:
        cfg: Configuration dictionary in unified format
        
    Returns:
        Processed configuration in the standard format
    """
    # Validate experiment_type is present
    experiment_type = cfg.get("experiment_type")
    if not experiment_type:
        raise ValueError("Missing 'experiment_type' field. Only unified configuration format is supported.")
    
    if experiment_type not in ("aldp", "gmm"):
        raise ValueError(f"Invalid experiment_type: {experiment_type}. Must be 'aldp' or 'gmm'")
    
    # Extract the experiment-specific configuration
    exp_config = cfg.get(experiment_type, {})
    if not exp_config:
        raise ValueError(f"No configuration found for experiment_type: {experiment_type}")
    
    # Start with base configuration (excluding experiment-specific sections)
    processed_cfg = {
        "experiment_type": experiment_type,  # Preserve experiment_type
        "device": cfg.get("device", "cuda"),
        "model_type": cfg.get("model_type", "realnvp"),
        "pt": cfg.get("pt", {}),
        "trainer": cfg.get("trainer", {}),
        "evaluation": cfg.get("evaluation", {}),
        "local_kernel": cfg.get("local_kernel", {}),
        "swap_kernel": cfg.get("swap_kernel", {}),
    }
    
    # Auto-generate experiment name if needed
    name = cfg.get("name", "unified_experiment_auto")
    if name == "unified_experiment_auto":
        # Generate a descriptive name based on experiment type and config
        if experiment_type == "aldp":
            trainer_cfg = processed_cfg.get("trainer", {}).get("realnvp", {})
            model_cfg = trainer_cfg.get("model", {})
            pt_cfg = processed_cfg.get("pt", {})
            
            temps = pt_cfg.get("temperatures", [300, 400, 500])
            n_reps = len(temps)
            n_couplings = model_cfg.get("n_couplings", 150)
            hidden_dim = model_cfg.get("hidden_dim", 512)
            
            # Include temperature range in name to avoid directory collisions
            temp_range = f"{temps[0]:.1f}to{temps[-1]:.1f}"
            name = f"aldp_cart_{n_reps}rep_{temp_range}_{n_couplings}coup_{hidden_dim}hidden"
            
        elif experiment_type == "gmm":
            gmm_cfg = exp_config.get("gmm_params", {})
            trainer_cfg = processed_cfg.get("trainer", {}).get("realnvp", {})
            model_cfg = trainer_cfg.get("model", {})
            pt_cfg = processed_cfg.get("pt", {})
            
            temps = pt_cfg.get("temperatures", [300, 400, 500])
            dim = gmm_cfg.get("dim", 3)
            n_mixes = gmm_cfg.get("n_mixes", 8)
            n_reps = len(temps)
            mode_arrangement = gmm_cfg.get("mode_arrangement", "random")
            
            # Include temperature range in name to avoid directory collisions
            temp_range = f"{temps[0]:.1f}to{temps[-1]:.1f}"
            name = f"gmm_{dim}dim_{n_mixes}mod_{n_reps}rep_{temp_range}_{mode_arrangement}"
    
    processed_cfg["name"] = name
    
    # Merge experiment-specific configuration (overrides)
    processed_cfg = _deep_merge_dicts(processed_cfg, exp_config)
    
    # Automatically set target.type based on experiment_type
    if "target" not in processed_cfg:
        processed_cfg["target"] = {}
    processed_cfg["target"]["type"] = experiment_type
    
    # Handle GMM-specific transformations
    if experiment_type == "gmm":
        # Move gmm_params to gmm for compatibility with existing target code
        if "gmm_params" in processed_cfg:
            processed_cfg["gmm"] = processed_cfg.pop("gmm_params")
    
    # Set device references in kernel configs
    if processed_cfg.get("local_kernel"):
        processed_cfg["local_kernel"]["device"] = processed_cfg["device"]
        # Set step_size from pt config if not explicitly set
        if processed_cfg["local_kernel"].get("step_size") is None:
            pt_cfg = processed_cfg.get("pt", {})
            processed_cfg["local_kernel"]["step_size"] = pt_cfg.get("step_size", 1e-4)
    
    if processed_cfg.get("swap_kernel"):
        processed_cfg["swap_kernel"]["device"] = processed_cfg["device"]
        # Don't auto-generate flow checkpoint path here - let build_swap_kernel handle it dynamically
        # This prevents early template substitution with wrong temperature values
        if processed_cfg["swap_kernel"].get("flow_checkpoint") is None:
            # Set a placeholder that will be handled dynamically
            processed_cfg["swap_kernel"]["flow_checkpoint"] = None
    
    logging.getLogger(__name__).info(f"Processed unified config for experiment_type: {experiment_type}")
    logging.getLogger(__name__).info(f"Generated experiment name: {name}")
    
    return processed_cfg 