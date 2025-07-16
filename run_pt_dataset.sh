#!/bin/bash
#SBATCH -J pt_dataset_AK
#SBATCH -A MLMI-jaf98-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pt_dataset_AK_%j.out
#SBATCH --error=logs/pt_dataset_AK_%j.err
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
source /home/$USER/.bashrc
conda activate accelmd

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Launching PT dataset builder for AK"
PDB_ROOT="data/timewarp/2AA-1-big/train"

stdbuf -oL python -u scripts/build_pt_dataset.py \
  --pdb-dir "$PDB_ROOT" \
  --peptides AK \
  --config configs/pt_dataset.yaml \
  --out-root data/pt_dipeptides \
  --device cuda 