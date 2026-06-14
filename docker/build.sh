#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-runtime}"
TAG="${2:-ml-lito:${TARGET}}"

if [[ "${TARGET}" != "runtime" && "${TARGET}" != "ready" ]]; then
    echo "usage: $0 [runtime|ready] [image-tag]" >&2
    exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
git -C "${REPO_ROOT}" submodule update --init --recursive

docker build \
    --target "${TARGET}" \
    --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
    --build-arg LITO_CUDA_ARCH_LIST="${LITO_CUDA_ARCH_LIST:-8.6}" \
    --tag "${TAG}" \
    "${REPO_ROOT}"

echo "Built ${TAG}"
