#!/bin/bash

########## SBATCH directives begin ############################
#SBATCH -J pt_dataset_batch5            # Job name
#SBATCH -A MLMI-jaf98-SL2-GPU           # Project / account (GPU hours)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # CPU cores
#SBATCH --mem=12G                       # Memory
#SBATCH --time=24:00:00                 # Wall-time (give extra time for 5 builds)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --output=logs/pt_dataset_batch_%j.out
#SBATCH --error=logs/pt_dataset_batch_%j.err
#SBATCH -p ampere                       # CSD3 GPU partition
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

echo "===================================================="
echo " Multi-dipedtide PT dataset generation (5×) "
echo " Config  : configs/pt_dataset.yaml"
echo " PDB root: data/timewarp/2AA-1-big/train"
echo "===================================================="

# Add project root to PYTHONPATH so imports resolve
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p logs

# Dipeptides to process (must correspond to <XY>-traj-state0.pdb under $PDB_ROOT)
PEPTIDES=(YF YK VE WK)

PDB_ROOT="data/timewarp/2AA-1-big/train"

TOTAL_START=$(date +%s)

for PEP in "${PEPTIDES[@]}"; do
  echo "----------------------------------------------------"
  echo "Building PT dataset for $PEP"
  echo "----------------------------------------------------"

  START=$(date +%s)

  stdbuf -oL python -u scripts/build_pt_dataset.py \
    --pdb-dir "$PDB_ROOT" \
    --peptides "$PEP" \
    --config configs/pt_dataset.yaml \
    --out-root data/pt_dipeptides \
    --device cuda

  END=$(date +%s)
  DUR=$((END-START))
  printf -v DUR_HMS '%02d:%02d:%02d' $((DUR/3600)) $(((DUR%3600)/60)) $((DUR%60))
  echo "Dataset for $PEP completed in $DUR_HMS (hh:mm:ss)"
  echo "Output saved to data/pt_dipeptides/$PEP"
  echo "----------------------------------------------------"

done

TOTAL_END=$(date +%s)
TOTAL_DUR=$((TOTAL_END-TOTAL_START))
printf -v TOTAL_HMS '%02d:%02d:%02d' $((TOTAL_DUR/3600)) $(((TOTAL_DUR%3600)/60)) $((TOTAL_DUR%60))

echo "===================================================="
echo "All 5 PT datasets generated. Total runtime: $TOTAL_HMS (hh:mm:ss)"
echo "====================================================" 