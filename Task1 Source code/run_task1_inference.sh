#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DATASET_ROOT [OUTPUT_ROOT]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(realpath "$1")"
OUTPUT_ROOT="${2:-$ROOT/output/task1_8_16666}"
PYTHON="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d "$DATASET_ROOT/validation/images" || ! -d "$DATASET_ROOT/validation/masks" ]]; then
  echo "Invalid dataset root: expected validation/images and validation/masks" >&2
  exit 1
fi
if [[ -e "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output root must be absent or empty: $OUTPUT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"
"$PYTHON" "$ROOT/verify_source_package.py" --allow-weights

required_weights=(
  "$ROOT/external/HRVRL/weights/G_pretrain.pkl"
  "$ROOT/external/RRWNet/weights/rrwnet_RITE_refinement.pth"
  "$ROOT/models/hrvrl/fold_0_best.pt"
  "$ROOT/models/hrvrl/fold_1_best.pt"
  "$ROOT/models/hrvrl/fold_2_best.pt"
  "$ROOT/models/hrvrl/fold_3_best.pt"
  "$ROOT/models/hrvrl/fold_4_best.pt"
)
for required in "${required_weights[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required model binary: $required" >&2
    echo "See WEIGHTS.md for filenames and SHA-256 values." >&2
    exit 1
  fi
done

for fold in 0 1 2 3 4; do
  echo "Running HRVRL fold $fold"
  "$PYTHON" -u "$ROOT/scripts/export_hrvrl_task1_validation.py" \
    --dataset-root "$DATASET_ROOT" \
    --hrvrl-root "$ROOT/external/HRVRL" \
    --pretrained-checkpoint "$ROOT/external/HRVRL/weights/G_pretrain.pkl" \
    --fold "$fold" \
    --checkpoint "$ROOT/models/hrvrl/fold_${fold}_best.pt" \
    --output-dir "$OUTPUT_ROOT/hrvrl/fold_${fold}" \
    --patch-size 512 \
    --stride 150 \
    --batch-size 4 \
    --device cuda \
    2>&1 | tee "$OUTPUT_ROOT/logs/hrvrl_fold_${fold}.log"

  echo "Running RRWNet stage 01 for fold $fold"
  "$PYTHON" -u "$ROOT/scripts/refine_task1_rrwnet.py" \
    --input-dir "$OUTPUT_ROOT/hrvrl/fold_${fold}/predictions" \
    --output-dir "$OUTPUT_ROOT/rrwnet/fold_${fold}" \
    --weights "$ROOT/external/RRWNet/weights/rrwnet_RITE_refinement.pth" \
    --rrwnet-root "$ROOT/external/RRWNet" \
    --dataset-root "$DATASET_ROOT" \
    --split validation \
    --recursions 0 \
    --device cuda \
    --amp \
    2>&1 | tee "$OUTPUT_ROOT/logs/rrwnet_fold_${fold}.log"
done

"$PYTHON" -u "$ROOT/scripts/assemble_task1_validation_ensemble.py" \
  --dataset-root "$DATASET_ROOT" \
  --prediction-pattern "$OUTPUT_ROOT/hrvrl/fold_{fold}/predictions" \
  --prediction-pattern "$OUTPUT_ROOT/rrwnet/fold_{fold}/stage_01/predictions" \
  --family-weights 0.90 0.10 \
  --folds 0 1 2 3 4 \
  --fold-weights 1 1 1 1 1 \
  --output-dir "$OUTPUT_ROOT/blend_raw" \
  2>&1 | tee "$OUTPUT_ROOT/logs/ensemble.log"

"$PYTHON" -u "$ROOT/scripts/apply_task1_postprocess.py" \
  --dataset-root "$DATASET_ROOT" \
  --split validation \
  --input-dir "$OUTPUT_ROOT/blend_raw" \
  --output-dir "$OUTPUT_ROOT/rr10_c3" \
  --low-threshold 0.002 \
  --seed-threshold 0.80 \
  --closing-radius 3 \
  2>&1 | tee "$OUTPUT_ROOT/logs/postprocess.log"

"$PYTHON" -u "$ROOT/scripts/enhance_task1_only.py" \
  --dataset-root "$DATASET_ROOT" \
  --input-dir "$OUTPUT_ROOT/rr10_c3" \
  --output-dir "$OUTPUT_ROOT/final" \
  2>&1 | tee "$OUTPUT_ROOT/logs/topology_enhancement.log"

"$PYTHON" "$ROOT/compare_reference.py" "$OUTPUT_ROOT/final"

"$PYTHON" "$ROOT/package_task1_predictions.py" \
  --dataset-root "$DATASET_ROOT" \
  --prediction-dir "$OUTPUT_ROOT/final" \
  --output-zip "$OUTPUT_ROOT/PixelPulse_Task1_predictions.zip"

echo "Task 1 reproduction completed."
echo "Predictions: $OUTPUT_ROOT/final"
echo "ZIP: $OUTPUT_ROOT/PixelPulse_Task1_predictions.zip"
