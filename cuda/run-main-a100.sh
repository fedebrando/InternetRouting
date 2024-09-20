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
## Reservation
##SBATCH --reservation=t_2023_hpc_lmi-20240419
#< Charge resources to account
#SBATCH --account=t_2023_hpc_lmi
module load cuda
cd "$SLURM_SUBMIT_DIR"
build/bin/main
