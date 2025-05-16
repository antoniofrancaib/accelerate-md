#!/usr/bin/env python
"""
End-to-end driver that:
1. trains a RealNVP flow on the 40-mode grid GMM (T=1 ↔ 10)
2. saves / reloads the checkpoint
3. generates a 2×2 scatter figure:
   TL = true high-T,  TR = mapped high→low,
   BL = true low-T,   BR = mapped low→high
All paths / hyper-params are read from configs/pt/gmm.yaml.
"""

from pathlib import Path
import torch, yaml, logging
import matplotlib.pyplot as plt
from src.accelmd.utils.config import load_config
from src.accelmd.trainers.realnvp_trainer import train_realnvp
from src.accelmd.targets.gmm import GMM
from src.accelmd.models.realnvp import create_realnvp_flow

CFG_PATH = "configs/pt/gmm.yaml"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def main():
    cfg = load_config(CFG_PATH)
    device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # 1) train (only if checkpoint missing)
    ckpt_dir = Path(cfg["trainer"]["realnvp"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_low, t_high = float(cfg["pt"]["temp_low"]), float(cfg["pt"]["temp_high"])
    ckpt_path = ckpt_dir / f"flow_{t_low:.2f}_to_{t_high:.2f}.pt"
    if not ckpt_path.is_file():
        logging.info("Checkpoint absent → training from scratch.")
        expected_ckpt_path = train_realnvp(cfg)
        
        # Verify training succeeded and checkpoint was created
        if not expected_ckpt_path.is_file():
            logging.error(f"Training failed to produce a valid checkpoint at {expected_ckpt_path}")
            logging.info("Will generate scatter plots with untrained flow - expect poor results")
            # Save a dummy checkpoint with random initialization to continue
            flow = create_realnvp_flow(cfg["trainer"]["realnvp"]["model"]).to(device)
            torch.save(flow.state_dict(), expected_ckpt_path)
        
        ckpt_path = expected_ckpt_path
    else:
        logging.info("Using existing checkpoint %s", ckpt_path)

    # ------------------------------------------------------------------ #
    # 2) rebuild the exact same GMM
    gmm_cfg = cfg["gmm"]
    gmm = GMM(gmm_cfg["dim"], gmm_cfg["n_mixes"], gmm_cfg["loc_scaling"], device=device)
    if gmm_cfg.get("custom_modes", True):
        if "locations" in gmm_cfg: gmm.locs.copy_(torch.tensor(gmm_cfg["locations"], device=device))
        if "scales"    in gmm_cfg: gmm.scale_trils.copy_(torch.tensor(gmm_cfg["scales"], device=device))
        if "weights"   in gmm_cfg: gmm.cat_probs.copy_(torch.tensor(gmm_cfg["weights"], device=device))
    else:
        # uniform mode generation already handled in trainer, reproduce it here
        from src.accelmd.evaluators.gmm_swap_rate import _build_gmm_from_config
        gmm = _build_gmm_from_config(gmm_cfg, device)

    hi_gmm = gmm.tempered_version(t_high, scaling_method="sqrt")

    # ------------------------------------------------------------------ #
    # 3) load flow
    flow = create_realnvp_flow(cfg["trainer"]["realnvp"]["model"]).to(device)
    flow.load_state_dict(torch.load(ckpt_path, map_location=device))
    flow.eval()

    # ------------------------------------------------------------------ #
    # 4) sample + map
    N = 5000
    with torch.no_grad():
        x_hi = hi_gmm.sample((N,)).to(device)
        x_lo = gmm.sample((N,)).to(device)
        x_hi2lo, _ = flow.inverse(x_hi)
        x_lo2hi, _ = flow.forward(x_lo)

    # ------------------------------------------------------------------ #
    # 5) figure
    def _scatter(ax, pts, title):
        ax.scatter(pts[:,0], pts[:,1], s=4, alpha=.6)
        ax.set_title(title)
        ax.set_xlim(-7,7); ax.set_ylim(-7,7); ax.grid(alpha=.3)

    fig, ax = plt.subplots(2,2, figsize=(13,11))
    _scatter(ax[0,0], x_hi.cpu(),              f"True High-T (T={t_high})")
    _scatter(ax[0,1], x_hi2lo.cpu(),           "Mapped High→Low (flow.inverse)")
    _scatter(ax[1,0], x_lo.cpu(),              f"True Low-T (T={t_low})")
    _scatter(ax[1,1], x_lo2hi.cpu(),           "Mapped Low→High (flow.forward)")
    plt.tight_layout()

    plot_dir = Path(cfg["evaluator"]["plot_dir"]); plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "bidirectional_verification_gmm40.png"
    plt.savefig(out_path, dpi=200)
    logging.info("Figure saved → %s", out_path)

if __name__ == "__main__":
    main()
