#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/_run_baseline_statistical_common.sh"

MODEL="${MODEL:-/model-weights/Llama-2-7b-chat-hf/}"
# SCORING_MODEL="${SCORING_MODEL:-${MODEL}}"
# REFERENCE_MODEL="${REFERENCE_MODEL:-/model-weights/Llama-2-7b-hf/}"

# run_baseline_statistical_for_model \
#   writingprompts reddit_eli5 stackmathqa

SCORING_MODEL="meta-llama/Llama-3.1-8B-Instruct"
REFERENCE_MODEL="meta-llama/Llama-3.1-8B"

run_baseline_statistical_for_model \
  writingprompts reddit_eli5 stackmathqa
