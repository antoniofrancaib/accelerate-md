#!/bin/bash

########## SBATCH directives begin ############################
#SBATCH -J pt_dataset_AF                 # Job name
#SBATCH -A MLMI-jaf98-SL2-GPU            # Project / account (GPU hours)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=12G                        # Memory
#SBATCH --time=24:00:00                  # Wall-time
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --output=logs/pt_dataset_AF_%j.out
#SBATCH --error=logs/pt_dataset_AF_%j.err
#SBATCH -p ampere                        # CSD3 GPU partition
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

# Add project root to PYTHONPATH so imports resolve
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "===================================================="
echo "Launching PT dataset builder for AF"
echo "Config : configs/pt_dataset.yaml"
echo "===================================================="

# Record start time
START_TS=$(date +%s)

# All dipeptide PDBs reside in the 2AA-1-big dataset directory.
PDB_ROOT="data/timewarp/2AA-1-big/train"

# Use stdbuf for line-buffered output
stdbuf -oL python -u scripts/build_pt_dataset.py \
  --pdb-dir "$PDB_ROOT" \
  --peptides AF \
  --config configs/pt_dataset.yaml \
  --out-root data/pt_dipeptides \
  --device cuda

# Record end time and compute elapsed
END_TS=$(date +%s)
DUR=$((END_TS-START_TS))
printf -v DUR_HMS '%02d:%02d:%02d' $((DUR/3600)) $(((DUR%3600)/60)) $((DUR%60))

echo "===================================================="
echo "PT dataset generation completed (AF)."
echo "Artefacts saved under data/pt_dipeptides/AF"
echo "Total runtime : $DUR_HMS (hh:mm:ss)"
echo "====================================================" 