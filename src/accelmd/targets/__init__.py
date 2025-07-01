from typing import Callable, Dict

__all__ = ["TARGET_REGISTRY", "register_target", "build_target"]

TARGET_REGISTRY: Dict[str, Callable] = {}

def register_target(name: str):
    """Decorator to register a target distribution factory by name."""

    def decorator(fn: Callable):
        TARGET_REGISTRY[name] = fn
        return fn

    return decorator

def build_target(name: str, *args, **kwargs):
    if name not in TARGET_REGISTRY:
        raise KeyError(f"Unknown target: {name}")
    return TARGET_REGISTRY[name](*args, **kwargs)

# Ensure key targets are registered on import
from importlib import import_module

for _mod in [".aldp_boltzmann", ".dipeptide_potential"]:
    try:
        import_module(_mod, package=__name__)
    except ModuleNotFoundError:
        pass 