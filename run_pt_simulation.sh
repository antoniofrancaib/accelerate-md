#!/bin/bash

# Extract peptide name from run_pt_simulation.py
PEPTIDE_NAME=$(python -c "
import sys; sys.path.append('src')
exec(open('run_pt_simulation.py').read())
print(config['name'])
")

#SBATCH -J pt_simulation_${PEPTIDE_NAME}
#SBATCH -A MLMI-jaf98-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pt_simulation_${PEPTIDE_NAME}_%j.out
#SBATCH --error=logs/pt_simulation_${PEPTIDE_NAME}_%j.err
#SBATCH -p ampere

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
mkdir -p results
mkdir -p plots/${PEPTIDE_NAME}

# Add project root to PYTHONPATH so run_pt_simulation.py finds accelmd
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Display GPU info and verify CUDA availability
nvidia-smi || true
echo ""
echo "CUDA device check:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

echo "===================================================="
echo "Launching PT simulation for ${PEPTIDE_NAME} peptide"
echo "Using device: $(python -c 'import torch; print(\"cuda\" if torch.cuda.is_available() else \"cpu\")')"
echo "Output will be saved to:"
echo "  - PT trajectory: results/pt_${PEPTIDE_NAME}.pt"
echo "  - Ramachandran plots: plots/${PEPTIDE_NAME}/"
echo "===================================================="

# Run PT simulation - this will create results/pt_AK.pt and plots/AK/ directory
stdbuf -oL python -u run_pt_simulation.py

# Verify outputs were created
if [ -f "results/pt_${PEPTIDE_NAME}.pt" ]; then
    echo "✅ PT trajectory saved to results/pt_${PEPTIDE_NAME}.pt"
    ls -lh results/pt_${PEPTIDE_NAME}.pt
else
    echo "❌ Expected output results/pt_${PEPTIDE_NAME}.pt not found"
fi

if [ -d "plots/${PEPTIDE_NAME}" ]; then
    echo "✅ Ramachandran plots directory created at plots/${PEPTIDE_NAME}/"
    ls -la plots/${PEPTIDE_NAME}/
else
    echo "❌ Expected plots directory plots/${PEPTIDE_NAME}/ not found"
fi

echo "===================================================="
echo "PT simulation completed."
echo "====================================================" 