#!/bin/bash

#SBATCH --job-name=job
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --account=marlowe-m000115
#SBATCH --gpus=1
#SBATCH --mem=5G
#SBATCH --time=00:20:00

unset LD_LIBRARY_PATH
cd /users/bendodge/hugegp
source .venv/bin/activate

python benchmark_script.py