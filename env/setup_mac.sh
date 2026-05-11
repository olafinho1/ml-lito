#!/bin/bash
# run at the repo root

# install pixi
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="/HOME/.pixi/bin:$PATH"

# create environment
# use the default pixi environment if PIXI_ENV is not set
: "${PIXI_ENV:=default}"
export PIXI_ENV
echo "using pixi environment $PIXI_ENV"
pixi install -e "$PIXI_ENV"
pixi run -e "$PIXI_ENV" post-install --as-is

# activate environment (for this shell), similar to conda activate env
# Note: we cannot do pixi shell in setup script (it launches a new shell so below will not be executed)
eval "$(pixi shell-hook -e "$PIXI_ENV")"

echo "Finished running setup_pixi.sh!"
