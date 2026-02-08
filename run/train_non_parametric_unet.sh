#!/bin/bash
#SBATCH --job-name=unet_nonparametric_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

# Load modules and activate virtualenv
module load python
source ~/virtual_env/synth_hipp_env/bin/activate

# Navigate to project directory
cd /cifs/khan_new/trainees/msalma29/synth_hipp/Learn2Synth

# Run training #V2
python scripts/train_non_parametric_unet.py fit \
  --data.ndim=3 \
  --model.ndim=3 \
  --model.nb_classes=9 \
  --data.batch_size=1 \
  --data.num_workers=12 \
  --trainer.max_epochs=1000 \
  --trainer.accelerator=gpu 