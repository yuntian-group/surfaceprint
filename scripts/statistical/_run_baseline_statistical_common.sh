#!/usr/bin/env bash

run_baseline_statistical_for_model() {
  local DATASETS=("$@")

  : "${MODEL:?Set MODEL (target source for filtering/labels) before calling run_baseline_statistical_for_model}"

  local DATASET_PATH="${DATASET_PATH:-dataset/merged_dataset}"
  local OUTPUT_DIR="${OUTPUT_DIR:-results/baseline_statistical}"
  local GPU="${CUDA_VISIBLE_DEVICES:-0}"
  local SCORING="${SCORING_MODEL:-${MODEL}}"
  local MODEL_SAFE="${MODEL//\//_}"
  local OUT="${OUTPUT_DIR}/${MODEL_SAFE}"
  local CRITERIA=(
    lastde_pp
    dna_detectllm
    fast_detect_gpt
    likelihood
    logrank
    entropy
    lrr
  )

  export CUDA_VISIBLE_DEVICES="${GPU}"
  mkdir -p "${OUT}"

  for CRITERION in "${CRITERIA[@]}"; do
    echo "=== ${MODEL} | scorer=${SCORING} | ${CRITERION} ==="
    local EXTRA_ARGS=()
    if [[ -n "${SCORING_MODEL:-}" ]]; then
      EXTRA_ARGS+=(--scoring-model "${SCORING_MODEL}")
    fi
    if [[ "${CRITERION}" == "fast_detect_gpt" \
       || "${CRITERION}" == "lastde_pp" \
       || "${CRITERION}" == "dna_detectllm" ]] \
       && [[ -n "${REFERENCE_MODEL:-}" ]]; then
      EXTRA_ARGS+=(--reference-model "${REFERENCE_MODEL}")
    fi
    python run_baseline_statistical.py \
      --criterion "${CRITERION}" \
      --model "${MODEL}" \
      "${EXTRA_ARGS[@]}" \
      --dataset-path "${DATASET_PATH}" \
      --normalize-mode raw \
      --label-mode source \
      --target-source "${MODEL}" \
      --include-dataset "${DATASETS[@]}" \
      --include-source human "${MODEL}" \
      --device cuda \
      --metrics-json "${OUT}/"
  done
}
