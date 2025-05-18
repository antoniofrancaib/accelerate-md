"""Metric sub-modules used by src.accelmd.evaluators.evaluator.

The individual metric modules each expose a public ``run(cfg: dict) -> None``
function which executes the metric end-to-end using the experiment *cfg*.
This init file merely makes the sub-modules importable as attributes so that
client code can do::

    from src.accelmd.evaluators import metrics
    metrics.summary_rates.run(cfg)

Nothing is executed at import-time beyond the standard module loading.
"""

from importlib import import_module

# Eagerly import the metric modules so they are available as attributes.
# This is purely for convenience and has no functional side-effects.
_modules = [
    "moving_average_acceptance",
    "acceptance_autocorrelation",
]
for _m in _modules:
    try:
        globals()[_m] = import_module(f"{__name__}.{_m}")
    except Exception as exc:  # pragma: no cover — robust to missing deps
        # Defer the ImportError until attribute access to keep package importable
        import types, warnings
        err = exc
        def _lazy_fail(*_args, **_kwargs):  # type: ignore[override]
            raise RuntimeError(
                f"Failed to import metric module '{_m}': {err}") from err

        mod = types.ModuleType(f"{__name__}.{_m}")
        mod.run = _lazy_fail  # type: ignore[attr-defined]
        globals()[_m] = mod

__all__ = _modules
