This is the repository for the MLMI MPhil project: Accelerating Molecular Dynamics via Neural Networks. (work still in progress)

🧬 Tutorial notebook to get started: (currently is quite toy, just put it here to remind me in the future to do when all the job is done!) [Introductory notebook](https://github.com/antoniofrancaib/accelerate-md/blob/main/notebooks/introduction.ipynb) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniofrancaib/accelerate-md/blob/main/notebooks/introduction.ipynb)

✍️ New to molecular simulations? Here’s a curated reading + watch list for beginners: (currently is quite toy, just put it here to remind me in the future to do when all the job is done!) [RESOURCES](notebooks/RESOURCES.md)

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

**Tweak ``configs/*.yaml`` at your convenience** 
From this file you choose ``<experiment_name>``, which metrics you want to see (all of them by default), the different ``temperatures`` levels at which you want to compute the simulation, the type of molecule you want to sample from, 

**Run**
```bash
python main.py --run-all --config configs/{exp-config}.yaml 
```
or on CSD3:
```bash
sbatch run_experiment.sh 
```
using same run_experiment.sh I provided (as an example) 

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