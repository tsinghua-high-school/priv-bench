#!/bin/bash 

#SBATCH --job-name=Example!

#SBATCH --time=12:00:00 

#SBATCH --account=bcga-delta-gpu 

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8 

#SBATCH --partition=gpuA100x4 

#SBATCH --mem=32G 

#SBATCH --mail-user=user.name@email.com

#SBATCH --mail-type=END 

#SBATCH --mail-type=FAIL 

#SBATCH -o /work/hdd/bcga/priv-bench/_out/%x.output

#SBATCH -e /work/hdd/bcga/priv-bench/_out/%x.err 

source /u/jliu80/.bashrc 

conda deactivate 

conda deactivate # just making sure 

module purge 

module reset # load the default Delta modules 

module load anaconda_gpu 

conda activate /work/hdd/bcga/jliu80/conda

#export PYTHONPATH=": /scratch/bcga/dchen4/open_clip/src" # (this line is optional) 

echo "Running Example Script!" 

# Your training and testing commands here 

cd /work/hdd/bcga/priv-bench/

srun --cpu_bind=v --accel-bind=gn python -u example.py 