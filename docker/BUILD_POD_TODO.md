# Build-pod TODO — building the ml-lito CUDA image

This work was authored in a pod **without** Docker-in-Docker, so the image cannot
be built here. The source + Docker assets live on the fork
`olafinho1/ml-lito`, branch **`docker-pod-image`**. Follow the steps below on a
freshly-spawned pod that has Docker and an NVIDIA GPU.

Reference docs: [`docs/docker.md`](../docs/docker.md) (build/run/test details).

## 0. Pod prerequisites
- [ ] Docker available to the user (`docker info` succeeds — true DinD or a
      mounted host socket; this is exactly what the authoring pod lacked).
- [ ] NVIDIA driver compatible with **CUDA 12.8** + NVIDIA Container Toolkit
      (`docker run --rm --gpus all ubuntu:22.04 nvidia-smi` works).
- [ ] Disk: budget ~40–60 GB free (apt + Pixi env + compiled CUDA exts + layers;
      the `ready` target adds the two checkpoints on top).
- [ ] `git`, and `gh` authenticated as `olafinho1` (`gh auth status`).

## 1. Get the code
```bash
gh repo clone olafinho1/ml-lito
cd ml-lito
git checkout docker-pod-image
git submodule update --init --recursive   # pulls pinned third_party/TRELLIS
```
- [ ] Sanity check the submodule landed:
      `test -f third_party/TRELLIS/trellis/__init__.py && echo OK`
      (the Dockerfile asserts this and fails the build otherwise).

## 2. Pick the CUDA arch
Default target is compute capability **8.6** (A40 / A6000). For another GPU set
`LITO_CUDA_ARCH_LIST` (e.g. `8.0` for A100, `9.0` for H100) on the build command.

- [ ] Confirm the build pod's GPU arch matches the value you'll build for. The
      smoke test runs CUDA kernels, so building for the wrong arch will fail at
      test time.

## 3. Build
Two targets — `runtime` (env + compiled CUDA extensions) and `ready` (runtime +
LiTo checkpoints and TRELLIS/DINOv2 caches baked in):
```bash
# runtime only
./docker/build.sh runtime ml-lito:runtime

# OR full image with weights (research-use license, see LICENSE_MODEL)
./docker/build.sh ready ml-lito:ready

# different GPU family:
LITO_CUDA_ARCH_LIST=8.0 ./docker/build.sh runtime ml-lito:a100
```
- [ ] Optionally raise `MAX_JOBS` (default 8) if the pod has many cores/RAM to
      speed up the CUDA extension compile: `MAX_JOBS=16 ./docker/build.sh ...`.
- [ ] The `ready` target downloads checkpoints from `ml-site.cdn-apple.com` and
      verifies SHA256 — make sure the pod has outbound network.

## 4. Acceptance test (needs a GPU on the build pod)
```bash
./docker/test.sh ml-lito:runtime all
# for the ready image, also validate the baked checkpoints:
./docker/test.sh ml-lito:ready all --weights
```
Image is complete when the smoke test passes: locked imports, CUDA 12.8 + BF16
compute/backward, xFormers/FlashAttention/PyTorch3D/nvdiffrast/gsplat kernels, a
LiTo `DiffusionTransformer` train step, and (for `ready`) checkpoint
deserialization.

## 5. Publish the image (so the next pod skips the build)
Decide a registry and push, e.g.:
```bash
docker tag ml-lito:ready <registry>/<ns>/ml-lito:ready
docker push <registry>/<ns>/ml-lito:ready
```
- [ ] Record the final image ref back in this repo / wherever pods are launched.

## 6. Merge back
- [ ] If the build needed fixes, commit them on `docker-pod-image` and push to
      the fork.
- [ ] When green, open a PR `olafinho1:docker-pod-image -> apple:main`
      (`gh pr create --repo apple/ml-lito --head olafinho1:docker-pod-image`).

## Notes / gotchas
- The authoring pod could not run any of step 3–4 (no Docker) — none of it has
  been executed yet, so treat the Dockerfile as **unverified end-to-end**.
- Running the image: see `docs/docker.md`. Mounting source at
  `/workspace/ml-lito` lets edits take effect without rebuilding (entrypoint
  prepends mounted `src` to `PYTHONPATH`).
- Set `LITO_START_SSH=1` + `PUBLIC_KEY=...` to get sshd inside the container.
