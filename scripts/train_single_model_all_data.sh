#!/usr/bin/env bash
set -euo pipefail

# Train one ASCII detector per model (human vs target model) on the merged dataset,
# write per-model metrics JSON next to each .pkl, and merge into one summary JSON.
# Run from the repository root:
#   bash scripts/train_single_model_all_data.sh

DATASET_PATH="${DATASET_PATH:-dataset/merged_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-models/merged_ascii_detectors_whitespace_normalized}"
mkdir -p "${OUTPUT_DIR}"

# Models to evaluate (same set as in dataset source column; "human" is the negative
# class for source-vs-human runs, so it is omitted here).
MODELS=(
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
  "facebook/opt-1.3b"
  "facebook/opt-iml-1.3b"
  "/model-weights/Llama-2-7b-hf/"
  "/model-weights/Llama-2-7b-chat-hf/"
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "allenai/Olmo-3-1025-7B"
  "allenai/Olmo-3-7B-Instruct"
  "Qwen/Qwen3.5-9B-Base"
  "/model-weights/Qwen3.5-9B"
  "google/gemma-4-E4B"
  "google/gemma-4-E4B-it"
)

METRICS_LIST_FILE="$(mktemp)"
cleanup() {
  rm -f "${METRICS_LIST_FILE}"
}
trap cleanup EXIT

for MODEL_SOURCE in "${MODELS[@]}"; do
  # Hugging Face IDs contain slashes; flatten for a single filename segment.
  MODEL_NAME_SAFE="${MODEL_SOURCE//\//_}"
  PKL="${OUTPUT_DIR}/source_vs_human_${MODEL_NAME_SAFE}.pkl"
  METRICS_JSON="${OUTPUT_DIR}/source_vs_human_${MODEL_NAME_SAFE}.metrics.json"
  echo "${METRICS_JSON}" >> "${METRICS_LIST_FILE}"

  echo "=== Training target_source=${MODEL_SOURCE} ==="
  python train_merged_ascii_detector.py \
    --dataset-path "${DATASET_PATH}" \
    --normalize-mode whitespace \
    --label-mode source \
    --target-source "${MODEL_SOURCE}" \
    --include-dataset writingprompts reddit_eli5 stackmathqa stackprogram\
    --include-source human "${MODEL_SOURCE}" \
    --output "${PKL}" \
    --metrics-json "${METRICS_JSON}"

  python3 scripts/analyze_ascii_features.py --model "${PKL}"
done

SUMMARY_JSON="${OUTPUT_DIR}/source_vs_human_all_models_summary.json"
python3 scripts/aggregate_merged_detector_metrics.py \
  --list-file "${METRICS_LIST_FILE}" \
  --output-json "${SUMMARY_JSON}"
