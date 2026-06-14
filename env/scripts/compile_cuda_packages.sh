#!/usr/bin/env bash
set -euo pipefail

# These packages compile against the installed PyTorch/CUDA pair. Keep the
# source revisions fixed so rebuilding the image does not silently change ABI.
: "${MAX_JOBS:=8}"
: "${LITO_CUDA_ARCH_LIST:=8.6}"
: "${THIRD_PARTY_PKG_INSTALL_DIR:=/tmp/lito-package-build}"
if [[ -z "${FLASH_ATTN_CUDA_ARCHS:-}" ]]; then
    case "${LITO_CUDA_ARCH_LIST}" in
        8.0|8.6) FLASH_ATTN_CUDA_ARCHS=80 ;;
        9.0) FLASH_ATTN_CUDA_ARCHS=90 ;;
        10.0) FLASH_ATTN_CUDA_ARCHS=100 ;;
        12.0) FLASH_ATTN_CUDA_ARCHS=120 ;;
        *)
            echo "unsupported FlashAttention CUDA architecture: ${LITO_CUDA_ARCH_LIST}" >&2
            exit 2
            ;;
    esac
fi
export MAX_JOBS
export TORCH_CUDA_ARCH_LIST="${LITO_CUDA_ARCH_LIST}"
export FLASH_ATTN_CUDA_ARCHS
mkdir -p "${THIRD_PARTY_PKG_INSTALL_DIR}"

PYTORCH3D_COMMIT="33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"
GSPLAT_COMMIT="4e52698e45eaaed929ed3a5065e96a688d085df6"
FUSED_SSIM_COMMIT="a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
TORCHSPARSE_COMMIT="385f5ce8718fcae93540511b7f5832f4e71fd835"

python -m pip install --no-build-isolation "xformers==0.0.33.post2"
python -m pip install --no-build-isolation "flash-attn==2.8.3.post1"
PYTORCH3D_SOURCE="${THIRD_PARTY_PKG_INSTALL_DIR}/pytorch3d"
rm -rf "${PYTORCH3D_SOURCE}"
git clone https://github.com/facebookresearch/pytorch3d.git "${PYTORCH3D_SOURCE}"
git -C "${PYTORCH3D_SOURCE}" checkout "${PYTORCH3D_COMMIT}"
FPS_SOURCE="${PYTORCH3D_SOURCE}/pytorch3d/csrc/sample_farthest_points/sample_farthest_points.cu"
test "$(grep -c '<<<threads, threads' "${FPS_SOURCE}")" -eq 2
sed -i 's/<<<threads, threads/<<<blocks, threads/g' "${FPS_SOURCE}"
python -m pip install --no-build-isolation "${PYTORCH3D_SOURCE}"
python -m pip install --no-build-isolation "open3d==0.19.0"
python -m pip install --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_COMMIT}"
python -m pip install --no-build-isolation \
    "git+https://github.com/rahul-goel/fused-ssim.git@${FUSED_SSIM_COMMIT}"
ROOTPATH_BUILD_DIR="${THIRD_PARTY_PKG_INSTALL_DIR}/rootpath"
rm -rf "${ROOTPATH_BUILD_DIR}"
mkdir -p "${ROOTPATH_BUILD_DIR}"
python -m pip download --no-deps --no-binary=:all: \
    "rootpath==0.1.1" \
    --dest "${ROOTPATH_BUILD_DIR}"
tar -xzf "${ROOTPATH_BUILD_DIR}/rootpath-0.1.1.tar.gz" \
    -C "${ROOTPATH_BUILD_DIR}"
sed -i \
    's/^requirements = get_requirements()$/requirements = ["six >= 1.11.0"]/' \
    "${ROOTPATH_BUILD_DIR}/rootpath-0.1.1/setup.py"
sed -i \
    "/    'setup_requires': \\[/,+2c\\    'setup_requires': []," \
    "${ROOTPATH_BUILD_DIR}/rootpath-0.1.1/setup.py"
python -m pip install --no-build-isolation --no-deps \
    "backports.cached-property==1.0.2" \
    "${ROOTPATH_BUILD_DIR}/rootpath-0.1.1"
python -m pip install --no-build-isolation --no-deps \
    "git+https://github.com/mit-han-lab/torchsparse.git@${TORCHSPARSE_COMMIT}"
