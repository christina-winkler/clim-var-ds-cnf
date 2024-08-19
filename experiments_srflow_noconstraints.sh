#!/bin/bash
#SBATCH --job-name=tcw_ds_flow
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
    #"srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 2 --log_interval 200 --constraint None"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 4 --log_interval 200 --constraint None"
    "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 8 --log_interval 200 --constraint None"
    "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 16 --log_interval 200 --constraint None"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 2 --log_interval 200 --constraint add"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 4 --log_interval 200 --constraint add"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 8 --log_interval 200 --constraint add"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 16 --log_interval 200 --constraint add"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 2 --log_interval 200 --constraint scadd"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 4 --log_interval 200 --constraint scadd"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 8 --log_interval 200 --constraint scadd"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 16 --log_interval 200 --constraint scadd"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 2 --log_interval 200 --constraint softmax"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 4 --log_interval 200 --constraint softmax"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 8 --log_interval 200 --constraint softmax"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 16 --log_interval 200 --constraint softmax"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 2 --log_interval 200 --constraint mult"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 4 --log_interval 200 --constraint mult"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 8 --log_interval 200 --constraint mult"
    # "srun --gres=gpu:1 python main.py --trainset era5-TCW --train --modeltype srflow --s 16 --log_interval 200 --constraint mult"                                       
)

# Run the experiment corresponding to the array task ID
${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}