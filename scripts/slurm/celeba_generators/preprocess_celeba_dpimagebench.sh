#!/bin/bash
#SBATCH --job-name=CELEBA_PREP
#SBATCH --time=12:00:00
#SBATCH --account=bcga-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --mem=64G
#SBATCH --mail-user=jli416@uky.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -o /work/hdd/bcga/priv-bench/_out/%x-%j.output
#SBATCH -e /work/hdd/bcga/priv-bench/_out/%x-%j.err

source /u/jliu80/.bashrc
module purge
module reset
module load miniforge3-python

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/hdd/bcga/jliu80/conda/env

export PYTHONNOUSERSITE=1
export PYTHONPATH=/work/hdd/bcga/priv-bench/external/DPImageBench:$PYTHONPATH

PYTHON_BIN="/work/hdd/bcga/jliu80/conda/env/bin/python"

echo "Python executable: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys, torch
print("sys.executable=", sys.executable)
print("torch=", torch.__version__)
print("cuda=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu=", torch.cuda.get_device_name(0))
PY

cd /work/hdd/bcga/priv-bench/external/DPImageBench

"$PYTHON_BIN" -u data/preprocess_dataset.py \
  --data_name celeba \
  --data_dir /work/hdd/bcga/priv-bench/datasets/dpimagebench \
  --celeba_attr Male \
  --fid_batch_size 500