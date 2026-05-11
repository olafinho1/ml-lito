#!/bin/bash

# Exit on error
set -e

echo "Checking installation for 'spz'..."

# 1. Check if already installed
if python -c "import spz" &> /dev/null; then
    echo "✅ spz is already installed. Skipping."
    exit 0
fi

# 2. Detect OS and Architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

# 3. Installation Logic
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "🍎 Detected MacOS (Apple Silicon). Installing spz with custom compilation flags..."

    # Flags required to force proper ARM64 compilation on Mac
    export ARCHFLAGS="-arch arm64"
    export CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64"

    # --no-binary=:all: forces pip to compile from source instead of looking for wheels
    pip install --no-binary=:all: "git+https://github.com/nianticlabs/spz.git"

else
    echo "🐧 Detected Linux (or Intel Mac). Installing spz normally..."

    # Standard installation for Linux/Intel
    pip install "git+https://github.com/nianticlabs/spz.git"
fi

echo "✅ spz installation complete."