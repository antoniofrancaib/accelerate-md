#!/bin/bash

####### SBATCH directives begin here ###############################
#SBATCH -J unified_experiment          # Job name
#SBATCH -A MLMI-jaf98-SL2-GPU          # GPU job account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # Number of CPU cores
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --time=6:00:00                 # Walltime - 6 hours (training + evaluation)
#SBATCH --mail-type=NONE
#SBATCH --output=logs/unified_experiment_%j.out
#SBATCH --error=logs/unified_experiment_%j.err
#SBATCH -p ampere                      # GPU partition name on CSD3
####### SBATCH directives end here ###############################

# ========== environment setup ==========
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

# activate your Python env (adjust name/path as needed)
source /home/$USER/.bashrc
conda activate accelmd

# ========== run experiment ================
WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"
echo "JobID: $SLURM_JOB_ID"
echo "Running on $(hostname)"
echo "Directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration file - now defaults to unified config
# Can still override with: CONFIG=configs/aldp.yaml sbatch run_experiment.sh
CONFIG="${CONFIG:-configs/experiment.yaml}"

# Display GPU info
nvidia-smi

# Run the full pipeline (train + evaluate)
CMD="stdbuf -oL python -u main.py --run-all --config $CONFIG"

echo "============================================="
echo "Starting AccelMD Unified Experiment"
echo "Config: $CONFIG"
echo "============================================="
echo "Executing:"
echo "  $CMD"
echo "============================================="

eval $CMD

# Determine output directory from config
OUTPUT_DIR=$(python - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))

# Handle unified config format
if "experiment_type" in cfg:
    # Unified config - determine name based on experiment type
    exp_type = cfg["experiment_type"]
    name = cfg.get("name", "unified_experiment_auto")
    if name == "unified_experiment_auto":
        # Auto-generate name (simplified version for output detection)
        if exp_type == "aldp":
            name = "aldp_cart_experiment"  
        elif exp_type == "gmm":
            name = "gmm_experiment"
        else:
            name = f"{exp_type}_experiment"
else:
    # Legacy config format
    name = cfg.get("name", "default")

base = cfg.get("output",{}).get("base_dir","outputs")
print(f"{base}/{name}")
PY
)

echo "============================================="
echo "Experiment completed"
echo "Results saved to: $OUTPUT_DIR/"
echo "============================================="
