#!/bin/bash
#SBATCH --job-name=lettuce_clf_all
#SBATCH --output=logs/clf_all_%j.log
#SBATCH --error=logs/clf_all_err_%j.log
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

cd "${SLURM_SUBMIT_DIR:-.}"
source "$HOME"/miniconda3/etc/profile.d/conda.sh
conda activate lettuce
module load cuda/12.1
mkdir -p logs results
python3 -m training.run_classification_all
echo "Classification (all architectures) done: $(date)"
