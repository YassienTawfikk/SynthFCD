#!/bin/bash
#SBATCH --job-name=synthseg_nonparametric_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

# Load modules and activate virtualenv
module load python

# Navigate to project directory
cd ..

# Run training #V3
python scripts/train_non_parametric_synthseg.py fit \
  --data.ndim=3 \
  --model.ndim=3 \
  --model.nb_classes=2 \
  --data.batch_size=1 \
  --data.num_workers=12 \
  --trainer.max_epochs=1000 \
  --trainer.accelerator=gpu
  # --ckpt_path=lightning_logs/version_2/checkpoints/last.ckpt