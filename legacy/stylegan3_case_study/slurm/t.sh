#!/bin/bash 

#SBATCH --job-name=T

#SBATCH --time=3:00:00 

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

srun --cpu_bind=v --accel-bind=gn --export=ALL,PYTHONNOUSERSITE=1 "$PYTHON_BIN" -u /work/hdd/bcga/priv-bench/t.py \
  --epochs 5 \
  --batch-size 8 \
  --max-grad-elements 8192