#!/bin/bash

########## SBATCH directives begin ############################
#SBATCH -J pt_swap_flows            # Job name
#SBATCH -A MLMI-jaf98-SL2-GPU       # Account / project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=36:00:00             # 12-hour wall-time
#SBATCH --output=logs/pt_swap_flows_%j.out
#SBATCH --error=logs/pt_swap_flows_%j.err
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

echo "JobID: $SLURM_JOB_ID"
echo "Running on $(hostname)"
echo "Working directory: $(pwd)"

mkdir -p logs

# Add project root to PYTHONPATH so `main.py` finds src/accelmd
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --------- User-configurable section ----------
CONFIG="${CONFIG:-configs/multi_graph_mcmc.yaml}"
TRAIN_MODE="${TRAIN_MODE:-all}"   # all | single
TEMP_PAIR="${TEMP_PAIR:-0 1}"      # used only if TRAIN_MODE==single
EPOCHS_OVERRIDE="${EPOCHS:-}"      # optionally override epochs
# ---------------------------------------------

# Display GPU info
nvidia-smi || true

# Build command
if [ "$TRAIN_MODE" = "single" ]; then
    CMD=(python -u main.py --config "$CONFIG" --temp-pair $TEMP_PAIR)
else
    CMD=(python -u main.py --config "$CONFIG")
fi

# Optional epochs override
if [ -n "$EPOCHS_OVERRIDE" ]; then
    CMD+=( --epochs "$EPOCHS_OVERRIDE" )
fi

# Echo and run
echo "===================================================="
echo "Launching PT-swap training"
echo "Config       : $CONFIG"
if [ "$TRAIN_MODE" = "single" ]; then
  echo "Temperature pair : $TEMP_PAIR"
else
  echo "Training all adjacent pairs"
fi
if [ -n "$EPOCHS_OVERRIDE" ]; then
  echo "Epochs override  : $EPOCHS_OVERRIDE"
fi
echo "Command: ${CMD[@]}"
echo "===================================================="

# Use stdbuf to force line-buffered output for live log streaming
stdbuf -oL "${CMD[@]}"

# Summarise outputs path
OUTPUT_DIR=$(python - <<PY
import yaml, os, sys; cfg=yaml.safe_load(open("$CONFIG")); base=cfg["output"]["base_dir"]; exp=cfg["experiment_name"]; print(os.path.join(base, exp))
PY
)

echo "===================================================="
echo "Training completed.  Artefacts can be found under: $OUTPUT_DIR"
echo "===================================================="

# ----- Post-processing: move Slurm logs + copy config -------
mkdir -p "$OUTPUT_DIR/logs"
cp "logs/pt_swap_flows_${SLURM_JOB_ID}.out" "$OUTPUT_DIR/logs/" 2>/dev/null || true
cp "logs/pt_swap_flows_${SLURM_JOB_ID}.err" "$OUTPUT_DIR/logs/" 2>/dev/null || true
# Save exact config used for this run (after possible env overrides)
cp "$CONFIG" "$OUTPUT_DIR/used_config.yaml" 2>/dev/null || true 