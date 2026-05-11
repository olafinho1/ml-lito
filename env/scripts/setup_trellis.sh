#!/bin/bash

CUR_DIR="$(dirname -- "$(readlink -f -- "$0")")"
export CUR_DIR
REPO_ROOT="$(dirname -- "$CUR_DIR")"
REPO_ROOT="$(dirname -- "$REPO_ROOT")"

THIRD_PARTY_PKG_INSTALL_DIR=/mnt/pkg_install
mkdir -p ${THIRD_PARTY_PKG_INSTALL_DIR}

cd ${REPO_ROOT}

# kaoline
cd ${THIRD_PARTY_PKG_INSTALL_DIR}
rm -rf kaolin
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
# make it reproducible
git checkout 3915474ca2af92a569f6180c6a42efd92b17de26
pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
IGNORE_TORCH_VER=1 python setup.py develop

git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git ${THIRD_PARTY_PKG_INSTALL_DIR}/diffoctreerast
pip install --no-build-isolation ${THIRD_PARTY_PKG_INSTALL_DIR}/diffoctreerast

cp -r ${REPO_ROOT}/third_party/TRELLIS/extensions/vox2seq ${THIRD_PARTY_PKG_INSTALL_DIR}/vox2seq
pip install --no-build-isolation ${THIRD_PARTY_PKG_INSTALL_DIR}/vox2seq

cd ${REPO_ROOT}
