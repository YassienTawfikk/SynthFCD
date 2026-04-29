#!/bin/bash
set -euo pipefail

: "${RUN_NAME:?RUN_NAME is not set}"
: "${TRAINING_TIME_MINUTES:?TRAINING_TIME_MINUTES is not set}"

if [ -n "${CKPT_PATH:-}" ]; then
  ckpt_arg=(--ckpt_path "$CKPT_PATH")
else
  ckpt_arg=()
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
L2S_RUN_NAME="$RUN_NAME" \
L2S_TIME_LIMIT_MINUTES="$TRAINING_TIME_MINUTES" \
python scripts/synthFCD/train_non_parametric_synthFCD.py fit \
  "${ckpt_arg[@]}" \
  --data.ndim 3 \
  --model.ndim 3 \
  --model.nb_classes 7 \
  --data.batch_size 1 \
  --data.num_workers 4 \
  --data.eval 0.2 \
  --trainer.max_epochs 2000 \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 16-mixed \
  --trainer.enable_progress_bar false \
  --trainer.log_every_n_steps 5 \
  --checkpoint.save_top_k 1 \
  --checkpoint.monitor eval_loss \
  --checkpoint.mode min \
  --checkpoint.save_last true