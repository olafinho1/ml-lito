#!/bin/bash

#set -e

# set the file limit (to support loading lots of images)
ulimit -n 262140  # only works on AWS (whose default value is 1024).
ulimit -a

source env/scripts/backoff.sh

# setup aws to use notary
#bash /usr/local/bin/install-aws-efa

# install pixi
apt-get install -y wget
wget -qO- https://pixi.sh/install.sh | sh
export PATH="/root/.pixi/bin:$PATH"

# install libraries needed by headless open3d
apt-get install -y libosmesa6-dev

# install libraries needed by blender
apt-get install -y  libx11-6 libxi6 libxrender1 \
  libxrandr2 libxfixes3 \
  libxkbcommon0 libsm6 libglu1-mesa \
  libxxf86vm1 libxcb1 libxcb-xinerama0 \
  libfontconfig1 libxshmfence1 libwayland-client0 \
  libwayland-cursor0 libwayland-egl1 \
  libpulse0 libsndfile1

# create environment
# use the default pixi environment if PIXI_ENV is not set
: "${PIXI_ENV:=default}"
export PIXI_ENV
echo "using pixi environment $PIXI_ENV"
pixi install -e "$PIXI_ENV" # --locked
pixi run -e "$PIXI_ENV" post-install --as-is

# activate environment (for this shell), similar to conda activate env
# Note: we cannot do pixi shell in setup script (it launches a new shell so below will not be executed)
eval "$(pixi shell-hook -e "$PIXI_ENV")"

echo "Finished running setup_pixi.sh!"
