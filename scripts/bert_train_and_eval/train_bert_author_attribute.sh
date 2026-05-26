#!/usr/bin/env bash
set -euo pipefail

# Fine-tune Qwen3-Embedding as an n-way author/source classifier on all
# instructional-tuned model sources (models only, no human).
# Run from the repository root:
#   bash scripts/train_bert_author_attribute.sh
#
# Multi-GPU (FSDP2 auto when WORLD_SIZE > 1):
#   NPROC_PER_NODE=2 bash scripts/train_bert_author_attribute.sh

DATASET_PATH="${DATASET_PATH:-dataset/merged_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-models/bert_author_attribute}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
NORMALIZE_MODE="${NORMALIZE_MODE:-whitespace}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
DTYPE="${DTYPE:-bf16}"
VAL_SPLIT="${VAL_SPLIT:-test}"
TEST_SPLIT="${TEST_SPLIT:-test}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

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

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC_PER_NODE}" train_bert_author_attribute.py \
    --dataset-path "${DATASET_PATH}" \
    --include-source "${INSTRUCT_MODELS[@]}" \
    --normalize-mode "${NORMALIZE_MODE}" \
    --model-name "${MODEL_NAME}" \
    --output "${OUTPUT_DIR}" \
    --metrics-json "${OUTPUT_DIR}/results.json" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --val-split "${VAL_SPLIT}" \
    --test-split "${TEST_SPLIT}"
else
  python train_bert_author_attribute.py \
    --dataset-path "${DATASET_PATH}" \
    --include-source "${INSTRUCT_MODELS[@]}" \
    --normalize-mode "${NORMALIZE_MODE}" \
    --model-name "${MODEL_NAME}" \
    --output "${OUTPUT_DIR}" \
    --metrics-json "${OUTPUT_DIR}/results.json" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --val-split "${VAL_SPLIT}" \
    --test-split "${TEST_SPLIT}"
fi
