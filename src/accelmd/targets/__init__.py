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

# -----------------------------------------------------------------------------
# Reusable Boltzmann target cache
# -----------------------------------------------------------------------------

from functools import lru_cache


def _freeze(obj):
    """Convert lists/dicts to tuples so they become hashable."""
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    return obj


@lru_cache(maxsize=128)
def _cached_target(name: str, args_key: tuple, kwargs_key: tuple):
    """Build or reuse a Boltzmann target."""
    return TARGET_REGISTRY[name](*args_key, **dict(kwargs_key))


# Replace the original build_target with a cached version --------------------------------

def build_target(name: str, *args, **kwargs):  # type: ignore[override]
    """Return (and cache) a Boltzmann target for the given arguments."""
    if name not in TARGET_REGISTRY:
        raise KeyError(f"Unknown target: {name}")

    args_key = _freeze(args)
    kwargs_key = _freeze(kwargs)
    return _cached_target(name, args_key, kwargs_key)

# Re-export for `__all__`
globals()["build_target"] = build_target 