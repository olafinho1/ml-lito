#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-ml-lito:runtime}"
LEVEL="${2:-all}"
WEIGHTS_FLAG=()
if [[ "${3:-}" == "--weights" ]]; then
    WEIGHTS_FLAG=(--weights)
fi

docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=24g \
    "${IMAGE}" \
    python docker/smoke_test.py --level "${LEVEL}" "${WEIGHTS_FLAG[@]}"
