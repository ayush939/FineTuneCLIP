#!/bin/bash
#SBATCH --job-name=ft-clip
#SBATCH --output=logs/output/ft-clip_output_%j.txt
#SBATCH --error=logs/output/ft-clip_error_%j.txt

#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=8 # Number of CPUs
#SBATCH --gres=gpu:7g.79gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=600G  # Specify the total amount of memory
#SBATCH --time=75:00:00  # Set the time limit to 1 minute
#SBATCH --partition=ultimate 
#SBATCH --qos=ultimate
#SBATCH --account=ultimate

# tensorboard --logdir=./logs --port=6006 --bind_all &
# Run the Python script
python -u -m clip_imagenet.py 2>&1