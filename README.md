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

**Expected output structure (v0)**

Each *temperature pair* gets its own sub-directory so that checkpoints and
metrics stay neatly isolated.  For a ladder with *n* temperatures (n−1 pairs)
you will see:

```bash
outputs/<experiment_name>/
├── used_config.yaml            # copy of the YAML used for the run (provenance)
├── pair_0_1/                   # adjacent-pair index, not absolute Kelvin
│   ├── models/
│   │   ├── best_model_epoch12.pt
│   │   ├── best_model_epoch26.pt
│   │   └── … (one file per new best checkpoint)
│   ├── metrics/
│   │   └── swap_acceptance.json        # naïve vs flow acceptance numbers
│   └── plots/
│       ├── loss_curve.png              # train/val loss over epochs
│       └── clipping_fraction.png       # fraction of batches hitting sentinel loss
├── pair_1_2/
│   └── … (same structure as above)
├── pair_2_3/
│   └── …
└── … (n−1 directories total)
```

If you train non-adjacent pairs or use a universal flow, the directory names
will follow the same `pair_i_j` pattern but with your custom indices.

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
