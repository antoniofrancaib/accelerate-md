# runner.py: needs to be fixed*
import argparse
import numpy as np

def run_mcmc(config_path):
    from main.targets.aldp import get_aldp_potential
    from main.sampler.mcmc_simulator import sample_mcmc_chain
    from main.sampler.sampler import LangevinDynamics
    import torch

    device = "cpu"
    target = get_aldp_potential(config_path, device)
    init_samples = target.sample(100).to(device)

    sampler = LangevinDynamics(
        x=init_samples,
        energy_func=lambda x: -target.log_prob(x),
        step_size=1e-4,
        device=device,
    )

    samples = sample_mcmc_chain(
        sampler=sampler,
        num_chains=100,
        num_steps=1000,
        num_interval=10,
        init_pos=init_samples
    )
    print("✅ Done. Sample shape:", samples.shape)


def run_bilevel(config_path):
    print("⚙️ Not implemented yet — add logic here for bilevel sampling.")
    # TODO: implement using bilevel_sampling.py

def run_pt(config_path):
    from main.targets.gmm import GMM
    from main.sampler.sampler import ParallelTempering
    from main.sampler.dyn_mcmc_warp import DynSamplerWrapper
    import torch
    import yaml
    from tqdm import tqdm

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config["device"] == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA not available, switching to CPU.")
            config["device"] = "cpu"

    device = config["device"]
    dim = 2  # GMM is defined for 2D in this repo
    target = GMM(dim=dim, n_mixes=5, loc_scaling=2.0, device=device)

    temperatures = torch.logspace(
        start=torch.log10(torch.tensor(config["temp_low"])),
        end=torch.log10(torch.tensor(config["temp_high"])),
        steps=config["total_n_temp"],
        device=device
    )

    init_samples = target.sample((config["total_n_temp"], config["num_chains"])).to(device)
    step_size = torch.tensor(config["step_size"], device=device)

    pt = ParallelTempering(
        x=init_samples,
        energy_func=lambda x: -target.log_prob(x),
        step_size=step_size.repeat_interleave(config["num_chains"]).unsqueeze(-1),
        swap_interval=config["swap_interval"],
        temperatures=temperatures,
        device=device
    )
    pt = DynSamplerWrapper(pt, per_temp=True, total_n_temp=config["total_n_temp"])

    for i in tqdm(range(config["num_steps"])):
        _, acc, _ = pt.sample()
        if i % config["check_interval"] == 0:
            print(f"[{i}] acc = {acc}, swap_rate = {pt.sampler.swap_rate:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run molecular dynamics samplers.")
    subparsers = parser.add_subparsers(dest="command")

    parser_mcmc = subparsers.add_parser("mcmc", help="Run MCMC sampler")
    parser_mcmc.add_argument("--config", type=str, required=True)

    parser_bilevel = subparsers.add_parser("bilevel", help="Run bilevel sampler")
    parser_bilevel.add_argument("--config", type=str, required=True)

    parser_pt = subparsers.add_parser("pt", help="Run Parallel Tempering sampler on GMM")
    parser_pt.add_argument("--config", type=str, required=True)
    
    args = parser.parse_args()

    if args.command == "mcmc":
        run_mcmc(args.config)
    elif args.command == "bilevel":
        run_bilevel(args.config)
    elif args.command == "pt":
        run_pt(args.config)
    else:
        parser.print_help()
