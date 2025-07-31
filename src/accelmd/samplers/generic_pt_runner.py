from __future__ import annotations

"""Generic Parallel Tempering (PT) runner that works for any TargetDistribution.

This is an **adapted** subset of the legacy `run_parallel_tempering.py` script.
The heavy Monte-Carlo logic lives in :pyclass:`src.accelmd.samplers.pt.parallel_tempering.ParallelTempering`.
Here we only provide convenience wrappers to
    • build the target distribution via ``build_target``,
    • set up the temperature ladder,
    • create initial replica configurations, and
    • run the PT loop with optional dynamic step-size adaptation.

The module is intentionally lightweight so that unit-tests can import the
helper functions without side-effects (no target is instantiated until one
calls :func:`run_parallel_tempering`).
"""

from pathlib import Path
from typing import Dict, Tuple, List

import torch
import numpy as np
from tqdm import tqdm

from src.accelmd.targets import build_target

# We *assume* that the PT machinery will be ported to ``src.accelmd.samplers.pt``
# exactly as used by the legacy runner.  To avoid circular deps we import lazily
# inside ``run_parallel_tempering``.

__all__ = [
    "setup_temperature_ladder",
    "initialise_configurations",
    "run_parallel_tempering",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def setup_temperature_ladder(cfg: Dict) -> torch.Tensor:
    """Return 1-D tensor of replica temperatures (K)."""
    # Priority 1: explicit list passed in config
    if "temperature_values" in cfg:
        temps = np.array(cfg["temperature_values"], dtype=np.float64)
        return torch.from_numpy(temps).float()

    sched = cfg.get("temp_schedule", "geom").lower()
    t_low = float(cfg["temp_low"])
    t_high = float(cfg["temp_high"])
    n_temp = int(cfg["total_n_temp"])

    if sched == "geom":
        temps = np.geomspace(t_low, t_high, n_temp, dtype=np.float64)
    elif sched == "linear":
        temps = np.linspace(t_low, t_high, n_temp, dtype=np.float64)
    else:
        raise ValueError(f"Unknown temp_schedule '{sched}'.")
    return torch.from_numpy(temps).float()


# -----------------------------------------------------------------------------


def initialise_configurations(target, cfg: Dict, device: torch.device) -> torch.Tensor:
    """Return initial coordinate tensor shaped [n_temp, n_chains, dim]."""
    n_chains = int(cfg["num_chains"])
    n_temp = int(cfg["total_n_temp"])

    # Try to use the minimised conformation that the target already has in its
    # Context; otherwise just sample a small random offset.
    try:
        pos_nm = target.context.getState(getPositions=True).getPositions(asNumpy=True)
        init = torch.tensor(pos_nm.value_in_unit(torch.tensor(1.0).unit.nanometer))
        init = init.view(-1)  # flatten dim
    except Exception:  # pragma: no cover – fallback path
        init = torch.zeros(target.dim)

    init = init.to(device)
    # replicate across replicas and chains
    x0 = init.unsqueeze(0).unsqueeze(0).repeat(n_temp, n_chains, 1)
    # tiny random perturbation so replicas are not identical
    x0 += torch.randn_like(x0) * 0.01
    return x0


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def run_parallel_tempering(cfg: Dict, *, test_run: bool = False, device: str | torch.device | None = None):
    """Run a PT simulation according to *cfg* and return a dict with results.

    ``cfg`` must contain – at minimum – the following keys (same as legacy
    script but names slightly cleaned):
        target: { name: "aldp"|"dipeptide",  ...target-specific kwargs }
        num_steps, swap_interval, step_size,
        temp_low, temp_high, total_n_temp, temp_schedule,
        num_chains,
        output_dir
    """
    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # ------------------------------------------------------------------
    # Build target distribution
    # ------------------------------------------------------------------
    tgt_cfg = cfg["target"]
    target = build_target(tgt_cfg.pop("name"), **tgt_cfg, device="cpu")

    # ------------------------------------------------------------------
    # Temperature ladder & initial positions
    # ------------------------------------------------------------------
    temps = setup_temperature_ladder(cfg).to(device)
    x0 = initialise_configurations(target, cfg, device)

    # ------------------------------------------------------------------
    # Step sizes tensor
    # ------------------------------------------------------------------
    step_size_scalar = float(cfg["step_size"])

    # ------------------------------------------------------------------
    # Import PT samplers lazily (they may not exist at import time of this file)
    # ------------------------------------------------------------------
    from src.accelmd.samplers.pt.sampler import ParallelTempering  # type: ignore
    from src.accelmd.samplers.pt.dyn_wrapper import DynSamplerWrapper  # type: ignore

    pt = ParallelTempering(
        x=x0,
        energy_func=lambda y: -target.log_prob(y),
        step_size=step_size_scalar,
        swap_interval=cfg["swap_interval"],
        temperatures=temps,
        mh=True,
        device=device,
        point_estimator=False,
    )

    pt = DynSamplerWrapper(
        pt,
        per_temp=False,
        total_n_temp=len(temps),
        target_acceptance_rate=0.6,
        alpha=0.25,
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    num_steps = 1000 if test_run else int(cfg["num_steps"])

    traj: List[torch.Tensor] = []
    acc_hist: List[torch.Tensor] = []
    swap_hist: List[np.ndarray] = []

    with tqdm(range(num_steps), desc="PT") as bar:
        for step in bar:
            samples, acc_rates, _ = pt.sample()
            traj.append(samples.clone().cpu().float())  # detach to save RAM
            acc_hist.append(acc_rates.clone().cpu().float())
            if hasattr(pt.sampler, "swap_rates"):
                swap_hist.append(pt.sampler.swap_rates.copy())

            # Update tqdm bar title
            bar.set_postfix(acc=acc_rates.mean().item())

            # ------------------------------------------------------------------
            # Slurm logging helper – emit a *newline‐terminated* progress message
            # every `print_interval` steps so that stdout/stderr is flushed to
            # file and users can monitor real-time progress via `tail -f`.
            # ------------------------------------------------------------------
            print_interval = 50
            if step % print_interval == 0:
                mean_acc = acc_rates.mean().item()
                swap_rate = getattr(pt.sampler, "swap_rate", 0.0)
                print(
                    f"[PT] step {step:>6}/{num_steps}  acc={mean_acc: .3f}  swap={swap_rate: .3f}",
                    flush=True,
                )

    trajectory = torch.stack(traj, dim=2)  # [temp, chain, step, dim]

    results = {
        "trajectory": trajectory,
        "acceptance_rates": acc_hist,
        "swap_rates": swap_hist,
        "temperatures": temps.cpu(),
        "config": cfg,
    }

    # Persist if requested
    out_dir = Path(cfg.get("output_dir", "outputs/pt_runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.get("run_name", target.__class__.__name__)
    out_path = out_dir / f"pt_{tag}.pt"
    torch.save(results, out_path)

    return results 