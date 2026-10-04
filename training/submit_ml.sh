#!/bin/bash
#SBATCH --job-name=lettuce_ml
#SBATCH --output=logs/ml_%j.log
#SBATCH --error=logs/ml_err_%j.log
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

cd "${SLURM_SUBMIT_DIR:-.}"
source "$HOME"/miniconda3/etc/profile.d/conda.sh
conda activate lettuce
module load cuda/12.1
mkdir -p logs results
python3 -m training.ml_baselines
echo "ML baselines done: $(date)"
