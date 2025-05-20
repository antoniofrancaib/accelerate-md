from typing import Dict, Any, Callable
import torch

# Local imports – avoid heavy optional deps at import time
from .gmm import GMM  # lightweight
from .aldp.boltzmann import get_aldp_target  # heavy but only used when chosen

# ---------------------------------------------------------------------------
# Registry of factory callables
# ---------------------------------------------------------------------------

TARGET_REGISTRY: Dict[str, Callable[[Dict[str, Any], torch.device], Any]] = {
    "gmm": lambda cfg, device: GMM(
        dim=cfg["gmm"]["dim"],
        n_mixes=cfg["gmm"]["n_mixes"],
        loc_scaling=cfg["gmm"].get("loc_scaling", 3.0),
        device=device,
    ),
    "aldp": lambda cfg, device: get_aldp_target(cfg["target"], device),
}


def build_target(cfg: Dict[str, Any], device: torch.device):
    """Instantiate and return the target distribution specified in *cfg*."""
    ttype = cfg["target"]["type"].lower()
    if ttype not in TARGET_REGISTRY:
        raise KeyError(f"Unknown target type: {ttype}. Available: {list(TARGET_REGISTRY)}")
    return TARGET_REGISTRY[ttype](cfg, device)
