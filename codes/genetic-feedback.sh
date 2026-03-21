#!/bin/bash
#SBATCH --job-name=genetic_fas
#SBATCH --output=genetic_fas_%j.out
#SBATCH --error=genetic_fas_%j.err
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=48
#SBATCH --nodes=1

# --- 1. Load System Python ---
module load python

# --- 2. Activate Your Conda Environment ---
# This line initializes conda for the script shell
source $(conda info --base)/etc/profile.d/conda.sh

# This line activates the environment where you installed pandas
conda activate feedback-weighted-maximization

# --- 3. Debug Info ---
echo "Running on host: $(hostname)"
echo "Using $SLURM_CPUS_PER_TASK CPUs"
echo "Start: $(date)"

# --- 4. Run the Python Script ---
# Ensure the .py file exists in this folder before running!
python Genetic-feedback-arc-set.py

echo "End: $(date)"
