#!/bin/bash
#SBATCH --job-name=experiment_array
#SBATCH --output=experiment_%A_%a.out
#SBATCH --error=experiment_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3  # Adjust this based on the number of experiments

# Load necessary modules
module load anaconda/3

# Activate your environment
source /path/to/your/venv/bin/activate

# Array of experiments
EXPERIMENTS=(
    "python main.py --dataset dataset1 --other-parameters"
    "python main.py --dataset dataset2 --other-parameters"
    "python main.py --dataset dataset1 --downsampling 8 --other-parameters"
    "python main.py --dataset dataset1 --downsampling 16 --other-parameters"
)

# Run the experiment corresponding to the array task ID
${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
