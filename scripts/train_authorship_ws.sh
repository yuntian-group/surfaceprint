#!/usr/bin/env bash
set -euo pipefail

# Example commands for training multiclass ASCII "author" (source) classifiers.
# Run from the repository root:
#   bash scripts/train_authorship_ws.sh

DATASET_PATH="${DATASET_PATH:-dataset/merged_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-models/merged_ascii_author_classifiers}"
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

python train_merged_ascii_author_classifier.py \
  --dataset-path "${DATASET_PATH}" \
  --include-source "${INSTRUCT_MODELS[@]}" \
  --classifier forest \
  --normalize-mode whitespace \
  --output "${OUTPUT_DIR}/models_only_forest.pkl" \
  --metrics-json "${OUTPUT_DIR}/results.json"
