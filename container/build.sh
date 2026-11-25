#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEF_FILE="${1:-${SCRIPT_DIR}/surfaceprint.def}"
SIF_NAME="${2:-surfaceprint.sif}"
DEST_DIR="${DEST_DIR:-${SCRIPT_DIR}/venv}"
BUILD_BASE="${BUILD_TMPDIR:-${TMPDIR:-/tmp}}"

case "$SIF_NAME" in
    ''|.|..|*/*) echo "Image name must be a filename: $SIF_NAME" >&2; exit 1 ;;
esac
[[ -f "$DEF_FILE" ]] || { echo "Definition file not found: $DEF_FILE" >&2; exit 1; }
command -v apptainer >/dev/null || { echo "apptainer is not installed." >&2; exit 1; }

mkdir -p "$BUILD_BASE" "$DEST_DIR"
BUILD_WORKSPACE="$(mktemp -d "${BUILD_BASE%/}/surfaceprint-build.XXXXXXXX")"
STAGED_IMAGE=""

cleanup() {
    rm -rf -- "$BUILD_WORKSPACE"
    if [[ -n "$STAGED_IMAGE" ]]; then
        rm -f -- "$STAGED_IMAGE"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export APPTAINER_TMPDIR="${BUILD_WORKSPACE}/build"
export APPTAINER_CACHEDIR="${BUILD_WORKSPACE}/cache"
export TMPDIR="${BUILD_WORKSPACE}/tmp"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$TMPDIR"
TMP_SIF="${BUILD_WORKSPACE}/${SIF_NAME}"

echo "Building $SIF_NAME in $BUILD_WORKSPACE"
apptainer build --fakeroot "$TMP_SIF" "$DEF_FILE"
[[ -s "$TMP_SIF" ]] || { echo "Build produced no image: $TMP_SIF" >&2; exit 1; }

STAGED_IMAGE="$(mktemp "${DEST_DIR%/}/.${SIF_NAME}.XXXXXXXX")"
cp "$TMP_SIF" "$STAGED_IMAGE"
chmod +x "$STAGED_IMAGE"
mv -f "$STAGED_IMAGE" "${DEST_DIR}/${SIF_NAME}"
STAGED_IMAGE=""
echo "Image saved to ${DEST_DIR}/${SIF_NAME}"
