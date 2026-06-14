#!/usr/bin/env python
"""Acceptance tests for the reusable LiTo CUDA image."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
TRELLIS_ROOT = REPO_ROOT / "third_party" / "TRELLIS"
if TRELLIS_ROOT.is_dir():
    sys.path.insert(0, str(TRELLIS_ROOT))


def timed(name, fn):
    start = time.perf_counter()
    print(f"[RUN ] {name}", flush=True)
    fn()
    print(f"[PASS] {name} ({time.perf_counter() - start:.2f}s)", flush=True)


def test_core_imports():
    modules = [
        "av",
        "blender_rendering",
        "cv2",
        "fastapi",
        "lightning",
        "lito",
        "numpy",
        "omegaconf",
        "open3d",
        "plibs",
        "rembg",
        "spz",
        "torch",
        "torchvision",
        "transformers",
        "trimesh",
        "uvicorn",
        "wandb",
        "webdataset",
    ]
    for module in modules:
        importlib.import_module(module)


def test_environment():
    import torch

    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.__version__.startswith("2.9.1"), torch.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert Path(os.environ.get("CUDA_HOME", "")).joinpath("bin", "nvcc").is_file()


def test_cuda():
    import torch

    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() >= 1
    capability = torch.cuda.get_device_capability(0)
    print(
        {
            "device": torch.cuda.get_device_name(0),
            "capability": capability,
            "torch_cuda": torch.version.cuda,
            "bf16": torch.cuda.is_bf16_supported(),
        },
        flush=True,
    )
    assert capability >= (8, 0), capability

    left = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    right = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    loss = (left @ right).float().square().mean()
    loss.backward()
    assert left.grad is not None and torch.isfinite(left.grad).all()


def test_cuda_extensions():
    import xformers.ops as xops

    import torch

    q = torch.randn(2, 128, 4, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    out = xops.memory_efficient_attention(q, q, q)
    out.float().mean().backward()
    assert out.shape == q.shape

    from flash_attn import flash_attn_func

    q = torch.randn(2, 128, 4, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    out = flash_attn_func(q, q, q)
    out.float().mean().backward()
    assert out.shape == q.shape

    from pytorch3d.ops import knn_points, sample_farthest_points

    points = torch.randn(2, 256, 3, device="cuda", requires_grad=True)
    knn = knn_points(points, points, K=4)
    knn.dists.mean().backward()
    assert knn.idx.shape == (2, 256, 4)

    small_cloud = torch.randn(1, 7, 3, device="cuda")
    _, sampled_indices = sample_farthest_points(
        small_cloud,
        K=5,
        random_start_point=True,
    )
    torch.cuda.synchronize()
    assert sampled_indices.shape == (1, 5)

    import nvdiffrast.torch as dr

    context = dr.RasterizeCudaContext()
    assert context is not None

    import gsplat

    means = torch.tensor([[0.0, 0.0, 2.0]], device="cuda", requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    scales = torch.full((1, 3), 0.1, device="cuda")
    opacities = torch.ones(1, device="cuda")
    colors = torch.ones(1, 3, device="cuda")
    viewmats = torch.eye(4, device="cuda").unsqueeze(0)
    intrinsics = torch.tensor(
        [[[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]]],
        device="cuda",
    )
    image, alpha, _ = gsplat.rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        64,
        64,
    )
    (image.mean() + alpha.mean()).backward()
    assert image.shape[-3:] == (64, 64, 3)

    for module in [
        "diffoctreerast",
        "fused_ssim",
        "kaolin",
        "spconv.pytorch",
        "torchsparse",
        "trellis",
        "vox2seq",
    ]:
        importlib.import_module(module)


def test_training_step():
    import torch

    from lito.models.dit import DiffusionTransformer

    model = DiffusionTransformer(
        time_embedder_config={
            "target": "lito.models.layers.FourierEmbed",
            "params": {
                "dim_pos": 1,
                "include_input": False,
                "min_freq_log2": 0,
                "max_freq_log2": 8,
                "num_freqs": 16,
                "log_sampling": True,
            },
        },
        num_latent=64,
        dim_latent=32,
        dim_cond_token=64,
        patch_size=1,
        cond_drop_prob=0.0,
        num_blocks=2,
        dim_hidden=128,
        num_self_heads=4,
        use_rmsnorm=True,
        use_swiglu=True,
        mlp_ratio=2,
    )
    model.init_positional_embedding()
    model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randn(2, 64, 32, device="cuda")
    cond = torch.randn(2, 37, 64, device="cuda")
    timestamp = torch.rand(2, device="cuda")
    target = torch.randn_like(tokens)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(tokens=tokens, t=timestamp, cond=cond)
        loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    print({"training_loss": loss.item(), "peak_vram_mb": torch.cuda.max_memory_allocated() // 2**20})


def test_baked_weights():
    import torch

    expected = {
        "lito_new.ckpt": 1_000_000_000,
        "lito_dit_rgba.ckpt": 7_000_000_000,
    }
    artifact_dir = Path(os.environ.get("LITO_REPO", "/opt/ml-lito")) / "artifacts"
    for filename, minimum_size in expected.items():
        path = artifact_dir / filename
        assert path.is_file(), path
        assert path.stat().st_size >= minimum_size, (path, path.stat().st_size)
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        assert isinstance(checkpoint, dict) and "state_dict" in checkpoint
        del checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        choices=["core", "cuda", "extensions", "training", "all"],
        default="all",
    )
    parser.add_argument("--weights", action="store_true")
    args = parser.parse_args()

    timed("core imports", test_core_imports)
    timed("version and environment contract", test_environment)
    if args.level in {"cuda", "extensions", "training", "all"}:
        timed("CUDA compute and backward", test_cuda)
    if args.level in {"extensions", "training", "all"}:
        timed("compiled CUDA extensions", test_cuda_extensions)
    if args.level in {"training", "all"}:
        timed("LiTo training-style forward/backward/optimizer step", test_training_step)
    if args.weights:
        timed("baked model weights", test_baked_weights)

    print("[PASS] ml-lito image acceptance suite", flush=True)


if __name__ == "__main__":
    main()
