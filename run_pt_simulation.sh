#!/bin/bash

########## SBATCH directives begin ############################
#SBATCH -J pt_simulation            # Job name
#SBATCH -A MLMI-jaf98-SL2-GPU       # Account / project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=32:00:00             # Wall-time
#SBATCH --output=logs/pt_simulation_%j.out
#SBATCH --error=logs/pt_simulation_%j.err
#SBATCH -p ampere                   # CSD3 GPU partition
########## SBATCH directives end ##############################

# ==== Environment setup ====
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/$USER/.bashrc
conda activate accelmd

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

echo "JobID       : $SLURM_JOB_ID"
echo "Hostname    : $(hostname)"
echo "Working dir : $(pwd)"

mkdir -p logs

# Add project root to PYTHONPATH so imports resolve
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --------- User-configurable section ----------
NAME="${NAME:-AK}"          # Peptide name (e.g., AK, GA, KW)
TOTAL_STEPS="${STEPS:-1000}"  # Override number of PT steps
# ---------------------------------------------

# Display GPU info (optional)
nvidia-smi || true

echo "===================================================="
echo "Launching PT simulation"
echo "Peptide        : $NAME"
echo "Total steps    : $TOTAL_STEPS"
echo "===================================================="

# Run the simulation
stdbuf -oL python -u run_pt_simulation_AA.py <<PY
import run_pt_simulation_AA as rps
cfg = rps.config.copy()
cfg["name"] = "$NAME"
cfg["num_steps"] = int("$TOTAL_STEPS")
rps.main(cfg)
PY

echo "===================================================="
echo "Simulation completed.  Check logs/ and results/ for artefacts."
echo "====================================================" 