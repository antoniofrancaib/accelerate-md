#!/bin/bash

####### SBATCH directives begin here ###############################
#SBATCH -J gmm_experiment              # Job name
#SBATCH -A MLMI-jaf98-SL2-GPU          # GPU job account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # Number of CPU cores
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --time=6:00:00                 # Walltime - 6 hours (training + evaluation)
#SBATCH --mail-type=NONE
#SBATCH --output=logs/gmm_experiment_%j.out
#SBATCH --error=logs/gmm_experiment_%j.err
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

# Configuration file (point to your YAML in configs/pt)
CONFIG="${CONFIG:-configs/aldp.yaml}"

# Display GPU info
nvidia-smi

# Run the full pipeline (train + evaluate)
# ———> note we drop --experiment-name since main.py doesn't accept it
# ———> model will be saved to outputs/${name}/model.pt and used by the evaluator
CMD="stdbuf -oL python -u main.py --run-all --config $CONFIG"

echo "============================================="
echo "Starting AccelMD GMM experiment"
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
name = cfg.get("name", "default")
base = cfg.get("output",{}).get("base_dir","outputs")
print(f"{base}/{name}")
PY
)

echo "============================================="
echo "Experiment completed"
echo "Results saved to: $OUTPUT_DIR/"
echo "============================================="
