# Reusable CUDA image

The image is built for LiTo research, training, notebooks, and the FastAPI
demo. Its default CUDA extension target is compute capability 8.6, matching
NVIDIA A40/A6000 GPUs.

## Build

Initialize the pinned TRELLIS submodule, then build one of two targets:

```bash
./docker/build.sh runtime ml-lito:runtime
./docker/build.sh ready ml-lito:ready
```

- `runtime` contains the locked Pixi environment and compiled CUDA extensions.
- `ready` additionally contains the recommended LiTo tokenizer and generator
  checkpoints plus the TRELLIS sparse decoder and DINOv2 caches. The Apple
  model license in `LICENSE_MODEL` limits the weights to research use.

For another GPU family, set its compute capability at build time:

```bash
LITO_CUDA_ARCH_LIST=8.0 ./docker/build.sh runtime ml-lito:a100
```

## Run

The host needs an NVIDIA driver compatible with CUDA 12.8 and the NVIDIA
Container Toolkit:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=24g \
  -v "$PWD:/workspace/ml-lito" \
  -v "$HOME/ml-lito-data:/workspace/data" \
  -v "$HOME/ml-lito-output:/workspace/output" \
  ml-lito:runtime
```

The environment remains under `/opt/ml-lito/.pixi` when source is mounted at
`/workspace/ml-lito`. The entrypoint prepends mounted source packages to
`PYTHONPATH`, so edits take effect without rebuilding the image.

Start common workflows:

```bash
python scripts/train.py --config configs/lito/tokenizer/lito_8k32.yaml
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
python demos/lito/fastapi_lito_demo.py \
  --checkpoint_url "${LITO_GENERATOR_CHECKPOINT:-https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt}" \
  --port 8000
```

Set `LITO_START_SSH=1` and pass `PUBLIC_KEY` to start `sshd` for platforms that
need SSH inside the custom image.

## Acceptance tests

Run the complete A40 test suite:

```bash
./docker/test.sh ml-lito:runtime all
./docker/test.sh ml-lito:ready all --weights
```

The image is considered complete when all of these pass:

1. Locked Python and project imports.
2. CUDA 12.8 visibility, BF16 matrix compute, and backward.
3. Executable xFormers, FlashAttention, PyTorch3D, nvdiffrast, and gsplat CUDA
   operations, plus imports for the remaining compiled extensions.
4. A small LiTo `DiffusionTransformer` forward, backward, and optimizer step.
5. For `ready`, minimum sizes and successful deserialization of both
   recommended checkpoints.

Full training still requires access to the private S3 dataset URLs referenced
by the released configs. The smoke test deliberately uses synthetic tensors so
image validation does not depend on those credentials.
