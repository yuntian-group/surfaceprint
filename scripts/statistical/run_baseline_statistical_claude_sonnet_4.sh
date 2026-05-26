#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/_run_baseline_statistical_common.sh"

MODEL="${MODEL:-anthropic/claude-sonnet-4}"
SCORING_MODEL="${SCORING_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
REFERENCE_MODEL="${REFERENCE_MODEL:-meta-llama/Llama-3.1-8B}"

run_baseline_statistical_for_model \
  writingprompts reddit_eli5 stackmathqa stackprogram
