#!/bin/bash 

#SBATCH --job-name=A_1

#SBATCH --time=4:00:00 

#SBATCH --account=bcga-delta-gpu 

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8 

#SBATCH --partition=gpuA100x4 

#SBATCH --mem=32G 

#SBATCH --mail-user=jli416@uky.edu

#SBATCH --mail-type=END 

#SBATCH --mail-type=FAIL 

#SBATCH -o /work/hdd/bcga/priv-bench/_out/%x-%j.output

#SBATCH -e /work/hdd/bcga/priv-bench/_out/%x-%j.err 

source /u/jliu80/.bashrc 

module purge 

module reset # load the default Delta modules 

module load miniforge3-python

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /work/hdd/bcga/jliu80/conda/env

# Prevent ~/.local site-packages from shadowing the conda environment.
export PYTHONNOUSERSITE=1

PYTHON_BIN="/work/hdd/bcga/jliu80/conda/env/bin/python"

echo "Python executable: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('sys.executable=', sys.executable)"

#export PYTHONPATH=": /scratch/bcga/dchen4/open_clip/src" # (this line is optional) 

echo "Running Example Script!" 

# Your training and testing commands here 

cd /work/hdd/bcga/priv-bench/

srun --cpu_bind=v --accel-bind=gn --export=ALL,PYTHONNOUSERSITE=1 "$PYTHON_BIN" -u /work/hdd/bcga/priv-bench/a_1.py \
  --mode sample \
  --network-pkl /work/hdd/bcga/priv-bench/checkpoints/00001-stylegan3-t-celeba_official_train_cond_256-gpus4-batch32-gamma2/network-snapshot-007600.pkl \
  --image-size 256 \
  --sample-size 10000