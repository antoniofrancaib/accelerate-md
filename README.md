This is the repository for the MLMI MPhil project: Accelerating Molecular Dynamics via Neural Networks. (work still in progress)

conda activate accelmd && \
sbatch --export=TRAIN_MODE=single,TEMP_PAIR="0 2" \
       run_pt_swap_flows.sh

conda activate accelmd && python -u main.py --config configs/multi_transformer_mcmc.yaml --evaluate --temp-pair 0 1 --checkpoint outputs/multi_transformer_mcmc/pair_0_1/models/best_model_epoch47.pt 


python experiments/SAR/evaluate_sar.py --architecture simple --pair 0 1 --peptide AA --num-seeds 10

conda activate accelmd && python main.py --config configs/AA_simple_mcmc.yaml --temp-pair 2 3 --epochs 3000

git fetch origin && git reset --hard origin/main

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

## Multi-Peptide Training

AccelMD supports training flows on multiple peptides simultaneously, enabling better generalization across different molecular sizes and types.

### Configuration

Enable multi-peptide mode by setting `mode: multi` and specifying peptide lists:

```yaml
mode: multi          # Enable multi-peptide training

peptides:
  train: [AA, AK, AS]  # Peptides for training (required)
  eval:  [GA, KW]      # Peptides for evaluation (optional, defaults to train)

multi_mode:
  batching: padding    # "padding" (default) or "uniform"

model:
  architecture: transformer  # MUST be "graph" or "transformer"
```

### Key Features

- **Mixed Training**: Train one flow on multiple peptide types simultaneously
- **Separate Evaluation**: Evaluate each peptide individually with detailed metrics
- **Dimension Agnostic**: Flows handle variable molecule sizes automatically
- **Two Batching Modes**:
  - `padding`: Pad smaller molecules to batch max size (default)
  - `uniform`: Round-robin between peptide-specific batches

### Architecture Requirements

Multi-peptide mode **requires** `architecture: "graph"` or `"transformer"`. The simple architecture cannot handle variable molecule sizes.

### Output Structure

In multi-peptide mode, evaluation results are organized by peptide:

```bash
outputs/<experiment_name>/
├── pair_0_1/
│   ├── AA/                     # Per-peptide evaluation
│   │   ├── metrics/
│   │   │   └── swap_acceptance.json
│   │   └── plots/
│   │       └── rama_grid.png
│   ├── AK/
│   │   └── ... (same structure)
│   └── models/                 # Shared model checkpoints
│       └── best_model_epoch*.pt
```

### Example Usage

```bash
# Train on multiple peptides
conda activate accelmd && python main.py --config configs/multi_peptide_example.yaml --temp-pair 0 1

# Results saved per-peptide for detailed analysis
```

See `configs/multi_peptide_example.yaml` for a complete configuration example.

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

## Replica Exchange Kinetics Workflow

### Long Sanity Runs
Run comprehensive 50k-step PT simulations for validation:

```bash
# 1. Long sanity run
python scripts/run_long_pt.py
```

This launches two simulations:
- **Vanilla PT**: Standard replica exchange (no flows)
- **Transformer Flow PT**: Enhanced with learned swap kernels

Results saved to `experiments/analysis/results/longtest/` with detailed metrics.

### Kinetics Dashboard
Generate a comprehensive 4-panel performance comparison:

```bash
# 2. Build the dashboard  
python -m experiments.analysis.kinetics_dashboard
open experiments/analysis/results/kinetics/replica_exchange_dashboard.png
```

The dashboard provides:
- **Panel A**: Mean swap acceptance rates
- **Panel B**: Round-trip times (cold→hot→cold)
- **Panel C**: φ dihedral autocorrelation times
- **Panel D**: Computational cost breakdown

### Individual PT Simulations
Run custom PT simulations with the new `src/run_pt.py` tool:

```bash
# Vanilla PT simulation
python -m src.run_pt --cfg configs/AA_simple.yaml \
                     --out_dir results/my_vanilla \
                     --n_steps 20000 --no_flow

# Flow-enhanced PT simulation  
python -m src.run_pt --cfg configs/multi_transformer.yaml \
                     --checkpoint_dir checkpoints/multi_transformer \
                     --out_dir results/my_flow \
                     --n_steps 20000

# Quick test with synthetic data
python -m src.run_pt --cfg configs/AA_simple.yaml \
                     --out_dir results/test \
                     --n_steps 1000 --dry_run
```

### Testing
Run the CI test suite:

```bash
# Run all tests
conda activate accelmd && pytest -q tests/

# Run specific PT tests
conda activate accelmd && pytest -q tests/test_run_pt.py
```
