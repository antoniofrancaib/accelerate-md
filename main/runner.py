# runner.py
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run molecular dynamics samplers.")
    subparsers = parser.add_subparsers(dest="command")

    parser_mcmc = subparsers.add_parser("mcmc", help="Run MCMC sampler")
    parser_mcmc.add_argument("--config", type=str, required=True)

    parser_bilevel = subparsers.add_parser("bilevel", help="Run bilevel sampler")
    parser_bilevel.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    if args.command == "mcmc":
        run_mcmc(args.config)
    elif args.command == "bilevel":
        run_bilevel(args.config)
    else:
        parser.print_help()
