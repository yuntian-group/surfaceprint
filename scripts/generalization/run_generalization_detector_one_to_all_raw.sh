#!/usr/bin/env bash
set -euo pipefail

# For each instruct model, train a detector on that model only and evaluate on
# all instruct models (including the train model, for in-domain + OOD metrics).
# Run from the repository root:
#   bash scripts/generalization/run_generalization_detector_one_to_all.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DATASET_PATH="${DATASET_PATH:-dataset/merged_dataset}"
PLANS_JSON="${PLANS_JSON:-scripts/generalization/one_to_all_other.json}"
OUTPUT_DIR="${OUTPUT_DIR:-models/generalization_one_to_all_other}"
if [[ -z "${PLAN_WORKERS:-}" ]]; then
  PLAN_WORKERS="$(python -c "import json; print(len(json.load(open('${PLANS_JSON}'))))")"
fi
METRICS_JSON="${METRICS_JSON:-${OUTPUT_DIR}/one_to_all_other_raw.metrics.json}"

mkdir -p "${OUTPUT_DIR}"

python train_generalization_detector.py \
  --dataset-path "${DATASET_PATH}" \
  --split train \
  --test-split test \
  --plans-json "${PLANS_JSON}" \
  --normalize-mode raw \
  --classifier forest \
  --plan-workers "${PLAN_WORKERS}" \
  --output-dir "${OUTPUT_DIR}" \
  --metrics-json "${METRICS_JSON}"

echo "Wrote metrics to ${METRICS_JSON}"
