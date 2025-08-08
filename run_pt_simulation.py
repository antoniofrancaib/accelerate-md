import yaml
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from src.accelmd.samplers.pt.sampler import ParallelTempering
from src.accelmd.samplers.pt.dyn_wrapper import DynSamplerWrapper
from src.accelmd.targets.dipeptide_potential import DipeptidePotentialCart
from src.accelmd.data.molecular_data import torch_to_mdtraj
from src.accelmd.utils.plot_utils import plot_Ramachandran
import mdtraj as md


"""Usage:
Edit run_pt_simulation.py line 127 to set your desired peptide (e.g., "name": "AK")
Submit with: conda activate accelmd && python run_pt_simulation.py or sbatch run_pt_simulation.sh
The script automatically uses the correct paths and names"""

def main(config):
    """
    Set up WandB
    """
    if config["use_wandb"]:
        with open('configs/logger/wandb.yaml', 'r') as file:
            wandb_config = yaml.safe_load(file)

        wandb_config['tags'] = ['PT', config["name"]]
        wandb_config['group'] = config["name"]
        wandb.init(
            entity=wandb_config['entity'],
            project=wandb_config['project'],
            mode="online" if not wandb_config['offline'] else "offline",
            group=wandb_config['group'],
            tags=wandb_config['tags'],
            config=config
        )
    """
    Set up temperatures
    """
    if config["temp_schedule"] == 'geom':
        all_temps = torch.from_numpy(np.geomspace(config["temp_low"], config["temp_high"], config["total_n_temp"])).float().to(config["device"])
    elif config["temp_schedule"] == 'linear':
        all_temps = torch.linspace(config["temp_low"], config["temp_high"], config["total_n_temp"]).float().to(config["device"])
    """
    Set up plot-saving paths for each temperature
    """
    base_plot_path = f"{config['plot_fold']}/{config['name']}"
    os.makedirs(config["plot_fold"], exist_ok=True)
    os.makedirs(base_plot_path, exist_ok=True)
    
    # Create temperature-specific directories
    temp_plot_paths = {}
    for i, temp in enumerate(all_temps):
        temp_dir = f"{base_plot_path}/{temp.item():.2f}"
        os.makedirs(temp_dir, exist_ok=True)
        temp_plot_paths[i] = temp_dir

    target = DipeptidePotentialCart(
        pdb_path=f"datasets/pt_dipeptides/{config['name']}/ref.pdb",
        n_threads=1,  # TODO: fix n_threads=64
        device=config["device"]
    )

    state = target.context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    pos_array = np.array([[v.x, v.y, v.z] for v in minimized_positions])

    x_init = torch.tensor(pos_array, device=config["device"]).view(1, -1)
    x_init = x_init.unsqueeze(0).repeat(config["total_n_temp"], config["num_chains"], 1)

    dim = x_init.shape[-1]

    def eval_fn(coords, plot_path):
        traj = md.Trajectory(
            coords.view(-1, target.n_atoms, 3).detach().cpu().numpy(), 
            topology=md.Topology.from_openmm(target.topology)
        )
        phi = md.compute_phi(traj)[1].flatten()
        psi = md.compute_psi(traj)[1].flatten()
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_Ramachandran(ax, phi, psi)
        fig.savefig(plot_path)
        return fig

    pt = ParallelTempering(x=x_init,
                       energy_func=lambda x: -target.log_prob(x),
                       step_size=torch.tensor([config["step_size"]] * (config["total_n_temp"] * config["num_chains"]), device=config["device"]).unsqueeze(-1),
                       swap_interval=config["swap_interval"],
                       temperatures=all_temps,
                       mh=True,
                       device=config["device"])
    pt = DynSamplerWrapper(pt, per_temp=True, total_n_temp=config["total_n_temp"], target_acceptance_rate=0.6, alpha=0.25)
    progress_bar = tqdm(range(config["num_steps"]), desc="Parallel Tempering")
    swap_rates = []
    traj = []
    for i in progress_bar:
        """
        The following two lines are almost all you need for sampling from PT
        """
        new_samples, acc, *_ = pt.sample()
        traj.append(new_samples.clone().detach().cpu().float())

        """
        The following codes are logging useful information
        """
        if pt.sampler.swap_rates:
            swap_rates.append(pt.sampler.swap_rates)
            if wandb.run is not None:
                for j in range(len(all_temps) - 1):
                    wandb.log({f"swap_rates/{all_temps[j].item():.2f}~{all_temps[j + 1].item():.2f}": pt.sampler.swap_rates[j]}, step=i)
        if i == 0 or (i + 1) % config["check_interval"] == 0:
            os.makedirs(config["save_fold"], exist_ok=True)
            # Save full trajectory structure [temp, chain, step, coord] for checkpoints too
            torch.save(torch.stack(traj, dim=2).detach().cpu().float(), f"{config['save_fold']}/pt_{config['name']}.pt")
            
            # Generate plots for all temperatures
            recent_traj = torch.stack(traj[-config["check_interval"]:], dim=2)
            for temp_idx in range(config["total_n_temp"]):
                fig = eval_fn(
                    recent_traj[temp_idx].reshape(-1, dim).to(config["device"]),
                    f'{temp_plot_paths[temp_idx]}/{i + 1}.png'
                )
                if wandb.run is not None:
                    wandb.log({f"Rama-plots/T_{all_temps[temp_idx].item():.2f}": wandb.Image(fig)}, step=i)
                plt.close(fig)  # Close figure to prevent memory leak
        if wandb.run is not None:
            for j in range(len(all_temps)):
                wandb.log({f"acc_rates/{all_temps[j].item():.2f}": acc[j].item()}, step=i)
        progress_bar.set_postfix_str(f"acc rate: {acc.mean().item()}")
    traj = torch.stack(traj, dim=2)
    os.makedirs(config["save_fold"], exist_ok=True)
    torch.save(traj.detach().cpu().float(), f"{config['save_fold']}/pt_{config['name']}.pt")

if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "name": "SS",  # gmm or aldp_inter or aldp_cart
        "temp_schedule": "geom",  # geom/linear, we usually use the geometrical schedule, but you could play
        "temp_low": 1.0,
        "temp_high": 1.0,
        "total_n_temp": 1,
        "num_chains": 10,
        "num_steps": 500000, # number of steps for PT simulation
        "step_size": 0.0001, # step size of MCMC
        "swap_interval": 100, # how often to swap samples between different temperatures
        "check_interval": 10000,  # how many steps to evaluate the samples
        "plot_fold": "plots",  # set where you save the plots
        "save_fold": "results",  # set where you save the results
        "use_wandb": False,
        "device": device
    }
    main(config)