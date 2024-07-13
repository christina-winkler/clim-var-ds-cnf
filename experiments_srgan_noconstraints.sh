#!/bin/bash
#SBATCH --job-name=tcw_ds_srgan
#SBATCH --output=experiment_%A_%a.out
#SBATCH --error=experiment_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19  # Adjust this based on the number of experiments

# Load necessary modules
module load anaconda/3

# Activate your environment
conda activate sr

# Array of experiments with GPU requests
EXPERIMENTS=(
    "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srgan --s 2 --log_interval 200 --nb 16 --constraint None"
    "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srgan --s 4 --log_interval 200 --nb 16 --constraint None"
    "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srgan --s 8 --log_interval 200 --nb 16 --constraint None"
    "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srgan --s 16 --log_interval 200 --nb 16 --constraint None"                                    
)

# Run the experiment corresponding to the array task ID
${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
