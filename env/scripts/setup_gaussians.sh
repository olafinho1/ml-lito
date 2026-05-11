#!/bin/bash

if pip show diff-gaussian-rasterization > /dev/null 2>&1; then
    echo "✅ Gaussian Splatting (diff-gaussian-rasterization) is already installed."
    exit 0
fi

echo "🚀 Installing Mip Splatting..."
#source environment/cuda_compat_switch.sh

TARGET_DIR="/tmp/extensions/mip-splatting"
if [ -d "$TARGET_DIR" ]; then
    echo "♻️  Cleaning up previous build artifact at $TARGET_DIR..."
    rm -rf "$TARGET_DIR"
fi

mkdir -p /tmp/extensions
git clone --recursive https://github.com/autonomousvision/mip-splatting.git "$TARGET_DIR"
pip install --no-build-isolation "$TARGET_DIR/submodules/diff-gaussian-rasterization/"

rm -rf "$TARGET_DIR"
echo "✅ Gaussian Splatting installed successfully."