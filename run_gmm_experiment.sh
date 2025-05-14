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

# Create all necessary output directories
mkdir -p checkpoints/realnvp_gmm
mkdir -p plots/pt
mkdir -p results
mkdir -p logs

# Add current directory to PYTHONPATH to ensure modules can be found
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration file path
CONFIG="configs/pt/gmm.yaml"

# Display GPU information
nvidia-smi

# Command to run the full pipeline (train flows + evaluate swap)
CMD="python -u runner.py run-all --config $CONFIG --wandb"

echo "============================================="
echo "Starting GMM experiment with RealNVP flows"
echo "Config: $CONFIG"
echo "============================================="
echo "Executing:"
echo "  $CMD"
echo "============================================="

eval $CMD

echo "============================================="
echo "Experiment completed"
echo "Results saved to plots/pt/ and results/"
echo "=============================================" 