#!/usr/bin/env bash
set -euo pipefail

# Evaluate a bert_author_attribute checkpoint on the test split and write
# per-class metrics JSON.
# Run from the repository root:
#   bash scripts/eval_bert_author_attribute.sh
#
# Override checkpoint or output path:
#   CHECKPOINT=models/bert_author_attribute/checkpoint-8480 \
#   METRICS_JSON=models/bert_author_attribute/checkpoint-8480/per_class_test_metrics.json \
#   bash scripts/eval_bert_author_attribute.sh

CHECKPOINT="${CHECKPOINT:-models/bert_author_attribute/checkpoint-8480}"
METRICS_JSON="${METRICS_JSON:-${CHECKPOINT}/per_class_test_metrics.json}"
DATASET_PATH="${DATASET_PATH:-dataset/merged_dataset}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
NORMALIZE_MODE="${NORMALIZE_MODE:-whitespace}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96c}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
DTYPE="${DTYPE:-bf16}"
TEST_SPLIT="${TEST_SPLIT:-test}"

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

python eval_bert_author_attribute.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-path "${DATASET_PATH}" \
  --include-source "${INSTRUCT_MODELS[@]}" \
  --normalize-mode "${NORMALIZE_MODE}" \
  --model-name "${MODEL_NAME}" \
  --metrics-json "${METRICS_JSON}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-length "${MAX_LENGTH}" \
  --dtype "${DTYPE}" \
  --test-split "${TEST_SPLIT}"
