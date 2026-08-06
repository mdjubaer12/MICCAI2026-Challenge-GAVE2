#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATASET_ROOT TRAINING_OUTPUT_ROOT" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(realpath "$1")"
RUN_ROOT="$2"
PYTHON="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$RUN_ROOT" && -n "$(find "$RUN_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Training output root must be absent or empty: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT/logs"

"$PYTHON" -u "$ROOT/scripts/evaluate_hrvrl_task1.py" \
  --dataset-root "$DATASET_ROOT" \
  --hrvrl-root "$ROOT/external/HRVRL" \
  --checkpoint "$ROOT/external/HRVRL/weights/G_pretrain.pkl" \
  --output-dir "$RUN_ROOT/zero_shot_full" \
  --patch-size 512 \
  --stride 150 \
  --batch-size 2 \
  --threshold 0.5 \
  --topology-paths 0 \
  --device cuda \
  2>&1 | tee "$RUN_ROOT/logs/zero_shot_teacher.log"

for fold in 0 1 2 3 4; do
  "$PYTHON" -u "$ROOT/scripts/train_hrvrl_task1.py" \
    --dataset-root "$DATASET_ROOT" \
    --manifest "$ROOT/manifests/training_folds.csv" \
    --hrvrl-root "$ROOT/external/HRVRL" \
    --checkpoint "$ROOT/external/HRVRL/weights/G_pretrain.pkl" \
    --teacher-root "$RUN_ROOT/zero_shot_full/predictions" \
    --fold "$fold" \
    --output-dir "$RUN_ROOT/fold_${fold}" \
    --epochs 24 \
    --patch-size 512 \
    --context-scale 1.5 \
    --patches-per-case 8 \
    --vessel-sampling-probability 0.8 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --workers 2 \
    --freeze-encoder-epochs 5 \
    --encoder-unfreeze-blocks 2 \
    --learning-rate 1e-4 \
    --encoder-learning-rate 1e-5 \
    --weight-decay 1e-4 \
    --dice-loss-weight 0.55 \
    --focal-tversky-weight 0.15 \
    --teacher-weight 0.10 \
    --teacher-channel-weights 0.25 1.00 1.00 \
    --hierarchy-weight 0.02 \
    --validation-every 2 \
    --validation-stride 256 \
    --validation-batch-size 2 \
    --validation-threshold 0.45 \
    --selection-weights 0.50 0.15 0.35 \
    --seed 20260723 \
    --log-every 25 \
    --device cuda \
    2>&1 | tee "$RUN_ROOT/logs/train_fold_${fold}.log"
done

echo "All five Task 1 folds completed under $RUN_ROOT"
