#!/bin/bash
#SBATCH --job-name=fas_dimacs_48
#SBATCH --output=fas_dimacs_48_%j.out
#SBATCH --error=fas_dimacs_48_%j.err
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=48
#SBATCH --nodes=1

# --- 1. Load System Python ---
module load python

# --- 2. Activate Conda Environment ---
# Initialize conda for this shell session
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the specific environment
conda activate feedback-weighted-maximization

# --- 3. Debug Info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"

# --- 4. Run the Python Script ---
# Make sure the file name matches exactly what you asked for:
python feedback_two_heu_wulver-mm9a-48cpu.py

echo "End Time: $(date)"
