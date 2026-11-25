#!/usr/bin/env bash

SURFACEPRINT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${SURFACEPRINT_ROOT}/container/venv/surfaceprint.sif}"

run_in_apptainer() (
    set -euo pipefail
    if [[ ! -f "$CONTAINER_IMAGE" ]]; then
        echo "Container image not found: $CONTAINER_IMAGE" >&2
        echo "Build it with: bash container/build.sh" >&2
        return 1
    fi
    command -v apptainer >/dev/null || { echo "apptainer is not installed." >&2; return 1; }

    if [[ -n "${HF_TOKEN:-}" ]]; then
        export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
    fi

    local apptainer_args=(exec)
    case "${SURFACEPRINT_GPU:-1}" in
        1) apptainer_args+=(--nv) ;;
        0) ;;
        *) echo "SURFACEPRINT_GPU must be 0 or 1." >&2; return 1 ;;
    esac

    apptainer "${apptainer_args[@]}" \
        --bind "${SURFACEPRINT_ROOT}:${SURFACEPRINT_ROOT}" "$CONTAINER_IMAGE" \
        bash -c '
            set -euo pipefail
            surfaceprint_root="$1"
            shift
            cd "$surfaceprint_root"
            export PYTHONPATH="$surfaceprint_root${PYTHONPATH:+:$PYTHONPATH}"
            exec "$@"
        ' surfaceprint "$SURFACEPRINT_ROOT" "$@"
)

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

if [[ $# -eq 0 ]]; then
    run_in_apptainer bash -l
else
    run_in_apptainer "$@"
fi
