# ml-lito images for RunPod

Two prebuilt CUDA images so a pod is usable immediately without compiling the
environment. Both are published **private** on Docker Hub.

| Image | Docker Hub ref | Content | Use it for |
|-------|----------------|---------|------------|
| runtime | `pirxe/ml-lito-runtime:latest` | Full env + all compiled CUDA extensions. **No model weights.** | Bring-your-own checkpoints; mount/download weights yourself. |
| ready | `pirxe/ml-lito-ready:latest` | Everything in runtime **plus** baked LiTo checkpoints and the TRELLIS/DINOv2 caches. | One-click inference/training, no first-run downloads. |

`ready` is built `FROM` runtime — same environment, just with weights added.

> **GPU requirement:** these are compiled for **compute capability 8.0** (A100).
> They will fail CUDA kernels on other GPUs. For A40/A6000 (8.6), H100 (9.0),
> etc., rebuild with `LITO_CUDA_ARCH_LIST` (see `docs/docker.md`). When picking a
> RunPod GPU, choose an **A100** for these tags.

## RunPod template settings

- **Container Image:** `pirxe/ml-lito-ready:latest` (or `…-runtime:latest`)
- **Registry Credentials:** required — the repos are private. Add a RunPod
  container-registry credential with your Docker Hub username + an access token.
- **Exposed ports:** `22` (SSH), `8000`, `8888` — expose whichever you need
  (TCP for SSH, HTTP for the others).
- **Container disk / volume:** runtime needs ~30 GB; ready ~50 GB. Add a
  persistent volume mounted at `/workspace` for anything you want to keep across
  pod restarts.

### Environment variables

| Var | Effect |
|-----|--------|
| `LITO_START_SSH=1` | Start `sshd` on boot (pair with `PUBLIC_KEY`). |
| `PUBLIC_KEY=ssh-ed25519 …` | Authorized key for SSH; written to `/root/.ssh/authorized_keys`. |
| `LITO_TOKENIZER_CHECKPOINT` / `LITO_GENERATOR_CHECKPOINT` | Override checkpoint paths. Preset in `ready`; set these in `runtime` if you supply your own weights. |

The entrypoint already wires up `PATH`, `CUDA_HOME`, `LD_LIBRARY_PATH`,
`PYTHONPATH`, and the cache dirs (`HF_HOME`, `TORCH_HOME`, etc.) — you don't need
to set those.

## Live code edits without rebuilding

Mount a checkout of this repo at **`/workspace/ml-lito`**. The entrypoint detects
it and prepends its `src` to `PYTHONPATH`, so your edits take effect without
rebuilding the image. Otherwise the baked copy at `/opt/ml-lito` is used.

## Sanity check on a fresh pod

```bash
python docker/smoke_test.py --level all            # runtime
python docker/smoke_test.py --level all --weights  # ready (also checks baked weights)
```

A green run confirms CUDA 12.8 + BF16, all CUDA extensions, a LiTo train step,
and (with `--weights`) that the checkpoints load.
