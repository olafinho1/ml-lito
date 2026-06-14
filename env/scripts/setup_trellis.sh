#!/usr/bin/env bash
set -euo pipefail

CUR_DIR="$(dirname -- "$(readlink -f -- "$0")")"
REPO_ROOT="$(dirname -- "$(dirname -- "$CUR_DIR")")"
: "${THIRD_PARTY_PKG_INSTALL_DIR:=/tmp/lito-package-build}"
: "${MAX_JOBS:=8}"
: "${LITO_CUDA_ARCH_LIST:=8.6}"
export MAX_JOBS
export TORCH_CUDA_ARCH_LIST="${LITO_CUDA_ARCH_LIST}"
# No GPU is visible during the image build, so torch.cuda.is_available() is
# False at compile time and kaolin would build CPU-only; force the CUDA build.
export FORCE_CUDA=1

KAOLIN_COMMIT="3915474ca2af92a569f6180c6a42efd92b17de26"
DIFFOCTREERAST_COMMIT="b09c20b84ec3aace4729e6e18a613112320eca3a"
VOX2SEQ_REVISION="559df5a42f3b3715e4801777f5e185511bae5e9b"

mkdir -p "${THIRD_PARTY_PKG_INSTALL_DIR}"

if ! python -c "import kaolin" >/dev/null 2>&1; then
    rm -rf "${THIRD_PARTY_PKG_INSTALL_DIR}/kaolin"
    git clone --recursive https://github.com/NVIDIAGameWorks/kaolin \
        "${THIRD_PARTY_PKG_INSTALL_DIR}/kaolin"
    git -C "${THIRD_PARTY_PKG_INSTALL_DIR}/kaolin" checkout "${KAOLIN_COMMIT}"
    python - "${THIRD_PARTY_PKG_INSTALL_DIR}/kaolin/setup.py" <<'PY'
from pathlib import Path
import sys

setup_path = Path(sys.argv[1])
setup_text = setup_path.read_text()
viz_requirements = """    with open(os.path.join(cwd, 'tools', 'viz_requirements.txt'), 'r') as f:
        requirements.extend(line.strip() for line in f)
"""
if viz_requirements not in setup_text:
    raise RuntimeError("Kaolin setup.py no longer matches the pinned patch")
setup_path.write_text(setup_text.replace(viz_requirements, ""))
PY
    python -m pip install \
        -r "${THIRD_PARTY_PKG_INSTALL_DIR}/kaolin/tools/build_requirements.txt" \
        -r "${THIRD_PARTY_PKG_INSTALL_DIR}/kaolin/tools/requirements.txt"
    (
        cd "${THIRD_PARTY_PKG_INSTALL_DIR}/kaolin"
        IGNORE_TORCH_VER=1 python -m pip install --no-build-isolation --no-deps .
    )
    python -m pip install "setuptools==82.0.1"
fi

if ! python -c "import diffoctreerast" >/dev/null 2>&1; then
    rm -rf "${THIRD_PARTY_PKG_INSTALL_DIR}/diffoctreerast"
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git \
        "${THIRD_PARTY_PKG_INSTALL_DIR}/diffoctreerast"
    git -C "${THIRD_PARTY_PKG_INSTALL_DIR}/diffoctreerast" checkout "${DIFFOCTREERAST_COMMIT}"
    python -m pip install --no-build-isolation "${THIRD_PARTY_PKG_INSTALL_DIR}/diffoctreerast"
fi

if ! python -c "import vox2seq" >/dev/null 2>&1; then
    VOX2SEQ_SOURCE="${REPO_ROOT}/third_party/TRELLIS/extensions/vox2seq"
    if [[ ! -d "${VOX2SEQ_SOURCE}" ]]; then
        VOX2SEQ_SNAPSHOT="${THIRD_PARTY_PKG_INSTALL_DIR}/trellis-3d-snapshot"
        rm -rf "${VOX2SEQ_SNAPSHOT}"
        python - "${VOX2SEQ_SNAPSHOT}" "${VOX2SEQ_REVISION}" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    "argojuni0506/TRELLIS-3D",
    repo_type="dataset",
    revision=sys.argv[2],
    allow_patterns=["extensions/vox2seq/*"],
    local_dir=sys.argv[1],
)
PY
        VOX2SEQ_SOURCE="${VOX2SEQ_SNAPSHOT}/extensions/vox2seq"
    fi
    python -m pip install --no-build-isolation "${VOX2SEQ_SOURCE}"
fi

cd "${REPO_ROOT}"
