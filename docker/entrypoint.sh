#!/usr/bin/env bash
set -e

export LITO_REPO="${LITO_REPO:-/opt/ml-lito}"
export LITO_ENV="${LITO_ENV:-${LITO_REPO}/.pixi/envs/default}"
export PATH="${LITO_ENV}/bin:/opt/pixi/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-${LITO_ENV}}"
export CPATH="${LITO_ENV}/targets/x86_64-linux/include:${CPATH:-}"
export LD_LIBRARY_PATH="${LITO_ENV}/lib:${LITO_ENV}/lib64:${LITO_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/opt/ml-cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/opt/ml-cache/torch}"
export U2NET_HOME="${U2NET_HOME:-/opt/ml-cache/rembg}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/opt/ml-cache/xdg}"

SOURCE_ROOT="${LITO_REPO}"
if [[ -f /workspace/ml-lito/pyproject.toml ]]; then
    SOURCE_ROOT=/workspace/ml-lito
fi
export PYTHONPATH="${SOURCE_ROOT}/src:${SOURCE_ROOT}/libraries/plibs/src:${SOURCE_ROOT}/libraries/blender_rendering/src:${SOURCE_ROOT}/third_party/TRELLIS:${PYTHONPATH:-}"

if [[ -f /opt/ml-lito/artifacts/lito_new.ckpt ]] && [[ -z "${LITO_TOKENIZER_CHECKPOINT:-}" ]]; then
    export LITO_TOKENIZER_CHECKPOINT=/opt/ml-lito/artifacts/lito_new.ckpt
fi
if [[ -f /opt/ml-lito/artifacts/lito_dit_rgba.ckpt ]] && [[ -z "${LITO_GENERATOR_CHECKPOINT:-}" ]]; then
    export LITO_GENERATOR_CHECKPOINT=/opt/ml-lito/artifacts/lito_dit_rgba.ckpt
fi

ulimit -n 262140 2>/dev/null || true

if [[ "${LITO_START_SSH:-0}" == "1" ]]; then
    mkdir -p /root/.ssh /run/sshd
    chmod 0700 /root/.ssh
    if [[ -n "${PUBLIC_KEY:-}" ]]; then
        printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
        chmod 0600 /root/.ssh/authorized_keys
    fi
    /usr/sbin/sshd
fi

cd "${SOURCE_ROOT}"
exec "$@"
