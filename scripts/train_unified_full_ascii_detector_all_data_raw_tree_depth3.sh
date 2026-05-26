#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-dataset/merged_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-models/merged_unified_ascii_full_detectors_raw_tree_depth3}"
mkdir -p "${OUTPUT_DIR}"

INSTRUCT_MODELS=(
  "gpt-5.4-mini"
  "gpt-4.1-mini"
  "gpt-4o-mini"
  "gpt-3.5-turbo"
  "anthropic/claude-3-haiku"
  "anthropic/claude-3.5-haiku"
  "anthropic/claude-sonnet-4"
  "claude-sonnet-4-6"
  "gemini-2.5-flash"
  "gemini-3.5-flash"
  "google/gemini-2.0-flash-001"
  "facebook/opt-iml-1.3b"
  "/model-weights/Llama-2-7b-chat-hf/"
  "meta-llama/Llama-3.1-8B-Instruct"
  "allenai/Olmo-3-7B-Instruct"
  "/model-weights/Qwen3.5-9B"
  "google/gemma-4-E4B-it"
)

PKL="${OUTPUT_DIR}/unified_full_raw_tree_depth3.pkl"
METRICS_JSON="${OUTPUT_DIR}/unified_full_raw_tree_depth3.metrics.json"

echo "=== Training unified full detector, tree max_depth=3 (${#INSTRUCT_MODELS[@]} instruct models) ==="
python train_merged_ascii_unified_detector.py \
  --dataset-path "${DATASET_PATH}" \
  --normalize-mode raw \
  --include-dataset writingprompts reddit_eli5 stackmathqa stackprogram \
  --instruct-source "${INSTRUCT_MODELS[@]}" \
  --split train \
  --test-split test \
  --output "${PKL}" \
  --classifier tree \
  --max-depth 3 \
  --metrics-json "${METRICS_JSON}"

python3 scripts/analyze_ascii_features.py --model "${PKL}"

echo "=== Done: ${PKL} ==="
echo "Per-model metrics table is in ${METRICS_JSON} under per_model_test"
