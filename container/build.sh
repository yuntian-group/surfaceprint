#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEF_FILE="${1:-${SCRIPT_DIR}/surfaceprint.def}"
SIF_NAME="${2:-surfaceprint.sif}"
DEST_DIR="${DEST_DIR:-${SCRIPT_DIR}/venv}"

export APPTAINER_TMPDIR=/tmp/apptainer-build
export APPTAINER_CACHEDIR=/tmp/apptainer-cache
export TMPDIR=/tmp/${USER}

TMP_SIF="/tmp/${SIF_NAME}"

cleanup() {
    echo "Cleaning up build directories..."
    rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$TMPDIR"
    rm -f "$TMP_SIF"
}
trap cleanup EXIT

mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$TMPDIR" "$DEST_DIR"

echo "Build workspace:   $APPTAINER_TMPDIR"
echo "Layer cache:       $APPTAINER_CACHEDIR"
echo "TMPDIR:            $TMPDIR"
echo "Build output:      $TMP_SIF"
echo "Final destination: $DEST_DIR/$SIF_NAME"
echo ""
df -h /dev/shm /tmp
echo ""
echo "Building ${SIF_NAME} from ${DEF_FILE} ..."
echo ""

apptainer build --fakeroot "$TMP_SIF" "$DEF_FILE"

echo ""
echo "Build complete: $(ls -lh "$TMP_SIF" | awk '{print $5}')"
echo "Copying to ${DEST_DIR}/${SIF_NAME} ..."
cp "$TMP_SIF" "$DEST_DIR/$SIF_NAME"
rm -f "$TMP_SIF"
echo "Done. Image at ${DEST_DIR}/${SIF_NAME}"
