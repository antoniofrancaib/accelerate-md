This is the repository for the MLMI MPhil project: Accelerating Molecular Dynamics via Neural Networks. (work still in progress)

Great docs of this repo: https://deepwiki.com/antoniofrancaib/accelerate-md

Toy [notebook](https://github.com/antoniofrancaib/accelerate-md/blob/main/notebooks/toy.ipynb) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniofrancaib/accelerate-md/blob/main/notebooks/toy.ipynb)

🧬 Tutorial notebook to get started: (currently is quite toy, just put it here to remind me in the future to do when all the job is done!) [Introductory notebook](https://github.com/antoniofrancaib/accelerate-md/blob/main/notebooks/introduction.ipynb) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniofrancaib/accelerate-md/blob/main/notebooks/introduction.ipynb)

✍️ New to molecular simulations? Here's a curated reading + watch list for beginners: (currently is quite toy, just put it here to remind me in the future to do when all the job is done!) [RESOURCES](notebooks/RESOURCES.md)

## Problem: 
Transferable normalizing-flow swap kernels boost replica-exchange acceptance to sample Boltzmann distributions of *unseen* peptides faster than vanilla Parallel Tempering.  

## Quick start  
The pipeline I wrote is *super* user-friendly structured.

**Clone the Repository**
```bash
git clone https://github.com/your-username/accelerate-md.git
cd accelerate-md
```

**Set Up the Conda Environment**
```bash
conda env create -f environment.yml
conda activate accelmd
```

**Choose Your Experiment** 

Edit `configs/experiment.yaml` to choose your experiment type:
```yaml
experiment_type: "aldp"  # For ALDP cartesian experiments
# OR
experiment_type: "gmm"   # For Gaussian Mixture Model experiments
```

**Run the Experiment**
```bash
# Uses configs/experiment.yaml by default
sbatch run_experiment.sh

# Or run directly
python main.py --run-all --config configs/experiment.yaml
```

**Expected output structure**

For n temperatures (n = len(temperatures)), the  is:

```bash
outputs/<experiment_name>/
├── config.yaml
├── experiment.log
├── models/
│   ├── flow_<T0>_<T1>.pt
│   ├── flow_<T1>_<T2>.pt
│   └── … (n−1 files)
├── metrics/
│   ├── swap_rate_flow_<T0>_<T1>.json
│   ├── swap_rate_flow_<T1>_<T2>.json
│   └── … (n−1 files)
└── plots/
    ├── bidirectional_verification_<T0>_<T1>.png
    ├── bidirectional_verification_<T1>_<T2>.png
    ├── ...
    ├── acceptance_autocorrelation_<T0>_<T1>.png
    ├── acceptance_autocorrelation_<T1>_<T2>.png
    ├── ...
    ├── moving_average_acceptance_<T0>_<T1>.png
    ├── moving_average_acceptance_<T1>_<T2>.png
    ├── ...
    └── … num_metrics*(n−1 files)
```

## Unified Configuration System

AccelMD uses a single configuration file for all experiments:

- **Switch experiment types** by changing `experiment_type: "aldp"` or `"gmm"`
- **Auto-generated names** like `aldp_cart_3rep_150coup_512hidden`
- **Smart defaults with overrides** - shared settings use ALDP as base, experiment-specific sections override what's different
- **Deep merging** preserves most settings while changing specifics

See `configs/README.md` for detailed configuration documentation.

**Pipeline explained:** (this is to be improved in the future )
```bash
┌──────── main.py ────────┐
│  loads cfg              │
│  builds target          │──►  src/accelmd/targets/*
│  builds PT sampler      │
│  trains flow (trainer)  │──►  src/accelmd/trainers/*
│  runs evaluators        │
└─────────────────────────┘
```
