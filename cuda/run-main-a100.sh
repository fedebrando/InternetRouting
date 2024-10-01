#!/bin/sh
#< 1 node with 1 GPU
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=4G
#SBATCH --job-name="IR-CUDA"
#SBATCH --output=%x.o%j
#< Charge resources to account
#SBATCH --account=<insert your account>
module load cuda
cd "$SLURM_SUBMIT_DIR"
build/bin/main
