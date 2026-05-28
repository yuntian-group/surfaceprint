#!/bin/bash
# Template: fill in FILL_ME values. See README §2.

INIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${INIT_ROOT}/container/venv/surfaceprint.sif}"

export APPTAINERENV_HF_TOKEN="FILL_ME_huggingface_token"
export APPTAINERENV_OPENAI_API_KEY="FILL_ME_openai_key"
export APPTAINERENV_ANTHROPIC_API_KEY="FILL_ME_anthropic_key"
export APPTAINERENV_GOOGLE_API_KEY="FILL_ME_google_genai_key"
export APPTAINERENV_GEMINI_API_KEY="${APPTAINERENV_GOOGLE_API_KEY}"
export APPTAINERENV_OPENROUTER_API_KEY="FILL_ME_openrouter_key"

export APPTAINERENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export APPTAINERENV_REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export APPTAINERENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# export APPTAINERENV_HF_HOME="FILL_ME_hf_cache_dir"
# export APPTAINERENV_DATASET_BASE_PATH="FILL_ME_dataset_root"

_apptainer_bind_flags=(
  --bind "${INIT_ROOT}:${INIT_ROOT}"
  # --bind /model-weights:/model-weights
  # --bind /scratch:/scratch
)

# module load apptainer

if [[ ! -e "${CONTAINER_IMAGE}" ]]; then
  echo "init.sh: container image not found at ${CONTAINER_IMAGE}" >&2
  echo "  Build with: bash container/build.sh" >&2
fi

_inside='export PYTHONPATH="'"${INIT_ROOT}"':${PYTHONPATH:-}"'

run_in_apptainer() {
  apptainer exec --nv "${_apptainer_bind_flags[@]}" "${CONTAINER_IMAGE}" \
    bash -c "set -euo pipefail; cd '${INIT_ROOT}' && ${_inside} && exec \"\$@\"" bash "$@"
}

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 0 2>/dev/null

if [[ $# -eq 0 ]]; then
  apptainer exec --nv "${_apptainer_bind_flags[@]}" "${CONTAINER_IMAGE}" \
    bash -lc "cd '${INIT_ROOT}' && ${_inside} && exec bash -l"
else
  run_in_apptainer "$@"
fi
