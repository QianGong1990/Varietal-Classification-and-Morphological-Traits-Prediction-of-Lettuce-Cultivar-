#!/bin/bash
#SBATCH --job-name=lettuce_gradcam
#SBATCH --output=logs/gradcam_%j.log
#SBATCH --error=logs/gradcam_err_%j.log
#SBATCH --time=00:30:00
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
mkdir -p logs results/gradcam
python3 -m training.run_gradcam
echo "Grad-CAM done: $(date)"
