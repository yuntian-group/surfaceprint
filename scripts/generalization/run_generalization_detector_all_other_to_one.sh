#!/usr/bin/env bash
set -euo pipefail

# For each instruct model, train a detector on all other instruct models and
# evaluate on that held-out model only.
# Run from the repository root:
#   bash scripts/generalization/run_generalization_detector_all_other_to_one.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DATASET_PATH="${DATASET_PATH:-dataset/merged_dataset}"
PLANS_JSON="${PLANS_JSON:-scripts/generalization/all_other_to_one.json}"
OUTPUT_DIR="${OUTPUT_DIR:-models/generalization_all_other_to_one}"
if [[ -z "${PLAN_WORKERS:-}" ]]; then
  PLAN_WORKERS="$(python -c "import json; print(len(json.load(open('${PLANS_JSON}'))))")"
fi
METRICS_JSON="${METRICS_JSON:-${OUTPUT_DIR}/all_other_to_one.metrics.json}"

mkdir -p "${OUTPUT_DIR}"

python train_generalization_detector.py \
  --dataset-path "${DATASET_PATH}" \
  --split train \
  --test-split test \
  --plans-json "${PLANS_JSON}" \
  --normalize-mode whitespace \
  --classifier forest \
  --plan-workers "${PLAN_WORKERS}" \
  --output-dir "${OUTPUT_DIR}" \
  --metrics-json "${METRICS_JSON}"

echo "Wrote metrics to ${METRICS_JSON}"
