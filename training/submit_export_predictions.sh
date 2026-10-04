#!/bin/bash
#SBATCH --job-name=lettuce_export_preds
#SBATCH --output=logs/export_preds_%j.log
#SBATCH --error=logs/export_preds_err_%j.log
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

cd "${SLURM_SUBMIT_DIR:-.}"
source "$HOME"/miniconda3/etc/profile.d/conda.sh
conda activate lettuce
module load cuda/12.1
mkdir -p logs results/predictions
python3 -m training.export_predictions
echo "Export predictions done: $(date)"
