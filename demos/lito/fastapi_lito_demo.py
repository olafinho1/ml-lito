#!/usr/bin/env python
"""
FastAPI demo for LiTo image-to-3D generation pipeline.

Usage:
    # at repo root
    python demos/lito/fastapi_lito_demo.py

"""

import argparse
import base64
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import typing as T
from urllib.parse import urlparse
import uuid

# repo_root = "/mnt/shape_tokenization"
# sys.path.insert(0, repo_root)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import numpy as np
from PIL import Image, ImageOps
import rembg
import spz
import uvicorn

import torch

from lito.eval_scripts import st_paper_utils
from lito.eval_scripts.st_model_utils import load_model
from lito.trainers import lito_dit_trainer, lito_trainer
from plibs import gs_utils, sh_utils

# ============================================================================
# Configuration
# ============================================================================
repo_root = os.path.normpath(os.path.join(__file__, "../../.."))
RESULTS_DIR = os.path.join(repo_root, "results", "tempdir")
ASSETS_DIR = os.path.join(repo_root, "results", "viewer_assets")

model: lito_dit_trainer.LiToDiTTrainer = None
st_model: lito_trainer.LightTokenizationTrainer = None
device = None
img_resolution = None

# Cache for preprocessed images (keyed by preprocess_id)
preprocess_cache: T.Dict[str, torch.Tensor] = {}

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Image to 3D Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Model Loading
# ============================================================================


def load_models(
    checkpoint_url: str,
    download_dir_root: str = "artifacts",
):
    """
    Load LiTo generative model and tokenzier.

    Args:
        checkpoint_url:
            str, It can be the url of the checkpoint file, or a local filename.
        download_dir_root:
            str, If checkpoint_url is remote, where to download the checkpoint locally.

    Notes:
        The function loads models into the global variables.
    """
    global model, st_model, device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float
    print(f"[Model] Using device: {device}")

    print("[Model] Loading generative model...")
    mdict = load_model(
        checkpoint_url=checkpoint_url,
        download_dir_root=download_dir_root,
        overwrite=False,
        dtype=dtype,
        device=device,
        load_params=True,
        # config_overwrite=dict(
        #     plm_config=dict(
        #         params=dict(
        #             load_pretrained_tokenizer_checkpoint=False,
        #         ),
        #     ),
        # ),
        # config_overwrite_allow_new_key=True,
    )
    model = mdict["model"]
    model.to(device=device)
    model.eval()
    model.freeze()

    # compile
    print(f"[Model] Compiling, may take some time...")
    model = torch.compile(model)

    st_model = model.pretrained_tokenizer
    print("[Model] Models loaded successfully!")


# ============================================================================
# Image Preprocessing
# ============================================================================


def preprocess_image(
    input_img: Image.Image,
    crop: bool = True,
    remove_bg: bool = True,
    fill_ratio: float = 0.8,
    keep_optical_axis: bool = True,
    img_resolution: int = 518,
) -> T.Dict[str, torch.Tensor]:
    """Preprocess the input image.

    Args:
        input_img: Input PIL image
        crop: Whether to crop/center the object
        remove_bg: Whether to remove background using rembg
        fill_ratio: Target ratio of object size to canvas size
    """
    # Apply EXIF orientation to fix rotated images (especially from phones)
    input_img = ImageOps.exif_transpose(input_img)

    print(f"crop: {crop}, remove_bg: {remove_bg}, fill_ratio: {fill_ratio}, keep_optical_axis: {keep_optical_axis}")
    print(f"img_resolution: {img_resolution}")

    if remove_bg:
        # Check if image already has alpha channel with transparency
        has_alpha = False
        if input_img.mode == "RGBA":
            alpha = np.array(input_img)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True

        if has_alpha:
            output = input_img  # (h, w, 4) uint8
        else:
            # remove background ourselves
            input_img = input_img.convert("RGB")
            output = rembg.remove(input_img)  # (h, w, 4) uint8

        output_np = np.array(output)  # (h, w, 4) uint8
        alpha = output_np[:, :, 3]  # (h, w) uint8
    else:
        # No background removal - just convert to RGBA with full opacity
        if input_img.mode != "RGBA":
            input_img = input_img.convert("RGBA")
        output_np = np.array(input_img)  # (h, w, 4) uint8
        # Ensure full opacity alpha channel
        output_np[:, :, 3] = 255
        alpha = output_np[:, :, 3]  # (h, w)

    if crop:
        cdict = st_paper_utils.determine_crop_and_pad(
            alpha=torch.from_numpy(alpha).float() / 255.0,
            keep_optical_axis=keep_optical_axis,
            fill_ratio=fill_ratio,
            th_alpha=0.8,
            pad_x_ratio=0.5,
            pad_y_ratio=0.5,
        )
        crop_x1 = cdict["crop_x1"]
        crop_y1 = cdict["crop_y1"]
        crop_x2 = cdict["crop_x2"]
        crop_y2 = cdict["crop_y2"]
        pad_left = cdict["pad_left"]
        pad_right = cdict["pad_right"]
        pad_top = cdict["pad_top"]
        pad_bottom = cdict["pad_bottom"]

        # actually crop
        rgba = torch.from_numpy(output_np)  # (h, w, 4rgba) uint8
        rgba = rgba[crop_y1:crop_y2, crop_x1:crop_x2].clone()  # (h', w', 4) uint8

        print(f"rgba shape: {rgba.shape}")
        print(f"pad_left: {pad_left}, pad_right: {pad_right}, pad_top: {pad_top}, pad_bottom: {pad_bottom}")
        print(f"new_w: {rgba.size(1) + pad_left + pad_right}, new_h: {rgba.size(0) + pad_top + pad_bottom}")

        assert pad_left >= 0 and pad_right >= 0 and pad_top >= 0 and pad_bottom >= 0, (
            f"{pad_left}, {pad_right}, {pad_top}, {pad_bottom}"
        )

        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            rgba = torch.nn.functional.pad(
                rgba,  # (h', w', 4)
                ((0, 0, pad_left, pad_right, pad_top, pad_bottom)),
                mode="constant",
                value=0,
            )  # (h', w', 4)
            print(f"after pad: rgba shape: {rgba.shape}")

        assert rgba.dtype == torch.uint8
        output_np = rgba.detach().cpu().clone().numpy()  # (h, w, 4rgba) uint8

    else:
        # No object-based cropping - make image square
        h, w = output_np.shape[:2]
        if remove_bg:
            # With background removal: pad to square
            max_dim = max(h, w)
            pad_h = max_dim - h
            pad_w = max_dim - w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            output_np = np.pad(
                output_np,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            # Without background removal: center crop to square
            min_dim = min(h, w)
            start_y = (h - min_dim) // 2
            start_x = (w - min_dim) // 2
            output_np = output_np[start_y : start_y + min_dim, start_x : start_x + min_dim]

    # resize
    output = Image.fromarray(output_np.astype(np.uint8))
    output = output.resize((img_resolution, img_resolution), Image.Resampling.LANCZOS)

    output = np.array(output).astype(np.float32) / 255  # (h, w, 4) [0, 1]
    output_with_alpha = output.copy()
    output = output[:, :, :3] * output[:, :, 3:4]

    return dict(
        premultiplied_rgb=torch.from_numpy(output).float(),  # premultiplied (h, w, 3) [0, 1]
        rgba=torch.from_numpy(output_with_alpha).float(),  # (h, w, 4rgba) [0, 1] straight
    )


# ============================================================================
# 3D Generation with Progress
# ============================================================================


def generate_3d_with_progress(
    sampling_steps: int,
    cfg_scale: float,
    cond_rgba: torch.Tensor,
    compress_spz: bool = False,
):
    """Generator that yields progress updates during 3D generation.

    Args:
        sampling_steps:
            int, Number of sampling steps
        cfg_scale:
            float, CFG scale for generation
        cond_rgba:
            (h, w, 4rgba) [0, 1] Pre-processed conditioning image tensor
        compress_spz: Whether to compress to SPZ format
    """
    global model, st_model, device

    if model is None or st_model is None:
        yield {"type": "error", "message": "Models not loaded yet"}
        return

    timings = {}
    num_steps = 5 if compress_spz else 4

    # Step 1: Preprocessing (already done, just mark as complete)
    yield {
        "type": "progress",
        "step": "preprocess",
        "progress": 0,
        "message": "Using cached preprocessed image...",
        "num_steps": num_steps,
    }
    timings["preprocess"] = 0.0
    yield {"type": "progress", "step": "preprocess", "progress": 100, "time": timings["preprocess"]}

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join(RESULTS_DIR, f"{timestamp}_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)

    input_img_path = os.path.join(output_dir, "input.png")
    Image.fromarray((cond_rgba.numpy() * 255).astype(np.uint8)).save(input_img_path)

    # Step 2: Sampling
    yield {"type": "progress", "step": "sampling", "progress": 0, "message": "Sampling shape tokens..."}
    t0 = time.time()

    if device.type == "cuda":
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            out_dict = model.inference_sample_latent(
                cond_rgba=cond_rgba.unsqueeze(0).unsqueeze(0),  # (b=1, q=1, h, w, 4rgba) [0, 1] rgb is straight
                ode_sampling_method="heun",
                ode_num_steps=sampling_steps,
                cfg_scale=cfg_scale,
                use_ema=True,
            )
    elif device.type in ["mps", "cpu"]:
        # use mlx
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            out_dict = model.inference_sample_latent_mlx(
                cond_rgba=cond_rgba.unsqueeze(0).unsqueeze(0),  # (b=1, q=1, h, w, 4rgba) [0, 1] rgb is straight
                ode_sampling_method="heun",
                ode_num_steps=sampling_steps,
                cfg_scale=cfg_scale,
                use_ema=True,
                mlx_compute_dtype="float16",  # MLPs internally promote to fp32 for accumulation, bfloat16 is slower than float16
            )
    else:
        raise NotImplementedError(device)

    timings["sampling"] = time.time() - t0
    yield {"type": "progress", "step": "sampling", "progress": 100, "time": timings["sampling"]}

    # Step 3: Decoding
    yield {"type": "progress", "step": "decoding", "progress": 0, "message": "Decoding gaussians..."}
    t0 = time.time()

    if st_model.voxel_decoder is not None:
        init_coord_src = "voxel_decoder"
    else:
        init_coord_src = "sample_xyz"

    if device.type == "cuda":
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=True):
            gs_dicts = st_model.inference_estimate_gaussians(
                fpoint_latent=out_dict["unnormalized_latent"],  # (b, nl, dl)
                init_coord_src=init_coord_src,
                steps_for_sample_xyz=50,
            )
            gs_dict = gs_dicts[0]
    elif device.type in ["mps", "cpu"]:
        # use mlx — staging's PyTorch perceiver path requires xformers, which
        # is unavailable on macOS, so route the gaussian decoder through MLX.
        # (The voxel_decoder used to compute init_coord still runs in PyTorch.)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=True):
            gs_dicts = st_model.inference_estimate_gaussians_mlx(
                fpoint_latent=out_dict["unnormalized_latent"],  # (b, nl, dl)
                init_coord_src=init_coord_src,
                steps_for_sample_xyz=50,
                mlx_compute_dtype="float16",
            )
            gs_dict = gs_dicts[0]
    else:
        raise NotImplementedError(device)

    timings["decoding"] = time.time() - t0
    yield {"type": "progress", "step": "decoding", "progress": 100, "time": timings["decoding"]}

    # Step 4: Saving
    yield {"type": "progress", "step": "saving", "progress": 0, "message": "Saving 3D model..."}
    t0 = time.time()

    _sh_degree = int(gs_dict["rgb_sh"].size(-2) ** 0.5) - 1
    ply_path = os.path.join(output_dir, "output.ply")

    _sh_degree = sh_utils.get_sh_degree_from_total_dim(gs_dict["rgb_sh"].size(-2))
    ngs = math.prod(gs_dict["xyz_w"].shape[:-1])
    gs = gs_utils.Gaussians(
        sh_degree=_sh_degree,
        xyz_w=gs_dict["xyz_w"].reshape(ngs, 3),  # (n, 3xyz)
        rgb_sh=gs_dict["rgb_sh"].reshape(ngs, -1, 3),
        rgb_sh_dc=None,
        rgb_sh_rest=None,
        scaling_logit=None,
        quaternion_prenorm=None,
        opacity_logit=None,
        scaling=gs_dict["scaling"].reshape(ngs, 3),  # (n, 3xyz)
        quaternion=gs_dict["quaternion"].reshape(ngs, 4),  # (n, 4xyzw)
        opacity=gs_dict["opacity"].reshape(ngs, 1),  # (n, 1)
        min_scaling=0,  # handled by network
        scaling_activation_type="none",
    )
    gs.save_ply(filename=ply_path)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    asset_filename = f"{timestamp}_{unique_id}.ply"
    asset_path = os.path.join(ASSETS_DIR, asset_filename)
    shutil.copy2(ply_path, asset_path)

    timings["saving"] = time.time() - t0
    yield {"type": "progress", "step": "saving", "progress": 100, "time": timings["saving"]}

    # Step 5: Compress to SPZ (optional)
    if compress_spz:
        yield {"type": "progress", "step": "compressing", "progress": 0, "message": "Compressing to SPZ..."}
        t0 = time.time()

        spz_filename = f"{timestamp}_{unique_id}.spz"
        spz_path = os.path.join(ASSETS_DIR, spz_filename)

        try:
            # Load PLY and convert to SPZ using spz library
            unpack_options = spz.UnpackOptions()
            # unpack_options.to_coord = spz.CoordinateSystem.RUB  # no need to unnecessarily rotate
            cloud = spz.load_splat_from_ply(asset_path, unpack_options)

            pack_options = spz.PackOptions()
            pack_options.version = 3  # sparkjs only supports up to version 3 as of March 20.
            # pack_options.from_coord = spz.CoordinateSystem.RUB  # no need to unnecessarily rotate
            success = spz.save_spz(cloud, pack_options, spz_path)
            assert success

            # Use SPZ file
            asset_filename = spz_filename
            asset_path = spz_path
        except Exception as e:
            print(f"[Warning] SPZ compression failed: {e}")
            # Fall back to PLY if compression fails
            compress_spz = False

        timings["compressing"] = time.time() - t0
        yield {"type": "progress", "step": "compressing", "progress": 100, "time": timings["compressing"]}

    # Get file sizes
    ply_filename = f"{timestamp}_{unique_id}.ply"
    ply_path = os.path.join(ASSETS_DIR, ply_filename)
    ply_size_bytes = os.path.getsize(ply_path)

    def format_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    ply_size_str = format_size(ply_size_bytes)

    # Calculate total time
    total_time = sum(timings.values())

    # Build response with both formats available
    result = {
        "type": "complete",
        "ply_filename": ply_filename,
        "ply_url": f"./assets/{ply_filename}",
        "ply_size": ply_size_str,
        "ply_size_bytes": ply_size_bytes,
        "timings": timings,
        "total_time": total_time,
    }

    if compress_spz:
        spz_size_bytes = os.path.getsize(asset_path)
        result["spz_filename"] = asset_filename
        result["spz_url"] = f"./assets/{asset_filename}"
        result["spz_size"] = format_size(spz_size_bytes)
        result["spz_size_bytes"] = spz_size_bytes
        result["has_spz"] = True
    else:
        result["has_spz"] = False

    # Done
    yield result


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page (read fresh on each request for hot-reload)."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index_lito.html")
    with open(html_path, "r") as f:
        return f.read()


@app.post("/preprocess")
async def preprocess_endpoint(
    image: UploadFile = File(...),
    crop: str = Form("true"),
    remove_bg: str = Form("true"),
    keep_optical_axis: str = Form("true"),
):
    """Preprocess an image, cache it, and return as base64 with a cache ID."""
    global preprocess_cache

    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    if pil_image.mode not in ("RGB", "RGBA"):
        pil_image = pil_image.convert("RGB")

    crop_bool = crop.lower() == "true"
    remove_bg_bool = remove_bg.lower() == "true"
    keep_optical_axis_bool = keep_optical_axis.lower() == "true"
    img_dict = preprocess_image(
        pil_image,
        crop=crop_bool,
        remove_bg=remove_bg_bool,
        keep_optical_axis=keep_optical_axis_bool,
        img_resolution=img_resolution,
    )
    cond_rgba = img_dict["rgba"]  # (h, w, 4rgba) [0, 1]

    # Generate a unique ID and cache the preprocessed tensor
    preprocess_id = str(uuid.uuid4())
    preprocess_cache[preprocess_id] = cond_rgba  # (h, w, 4rgba) [0, 1] tensor

    # Clean up old cache entries (keep only last 10)
    if len(preprocess_cache) > 10:
        oldest_key = next(iter(preprocess_cache))
        del preprocess_cache[oldest_key]

    # Convert to base64 for preview
    img_np = (cond_rgba.numpy() * 255).astype(np.uint8)  # (h, w, 4rgba) uint8
    pil_result = Image.fromarray(img_np)

    buffer = io.BytesIO()
    pil_result.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"image_base64": img_base64, "preprocess_id": preprocess_id}


@app.post("/generate-stream")
async def generate_stream(
    preprocess_id: str = Form(...),
    sampling_steps: int = Form(20),
    cfg_scale: float = Form(3.0),
    compress_spz: str = Form("false"),
):
    """Generate with streaming progress (SSE) using cached preprocessed image.

    The crop setting is already applied during preprocessing, so we just use
    the cached tensor directly. When the user changes the crop checkbox,
    a new preprocess call happens which updates the cache with a new ID.
    """
    global preprocess_cache

    # Get cached preprocessed image
    if preprocess_id not in preprocess_cache:
        return StreamingResponse(
            iter(
                [
                    f"data: {json.dumps({'type': 'error', 'message': 'Preprocessed image not found. Please re-upload the image.'})}\n\n"
                ]
            ),
            media_type="text/event-stream",
        )

    cond_rgba = preprocess_cache[preprocess_id]  # (h, w, 4rgba) [0, 1] tensor
    compress_spz_bool = compress_spz.lower() == "true"

    def event_generator():
        for update in generate_3d_with_progress(
            sampling_steps=sampling_steps,
            cfg_scale=cfg_scale,
            cond_rgba=cond_rgba,
            compress_spz=compress_spz_bool,
        ):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/assets/{filename}")
async def serve_asset(filename: str):
    """Serve PLY or SPZ file with correct headers."""
    file_path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename,
    )


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "model_loaded": model is not None, "device": str(device) if device else None}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiTo image to 3D demo")
    parser.add_argument(
        "--port",
        type=int,
        default=os.environ.get("WEBPAGE_PORT", 7860),
        help="the port to run the server on",
    )
    parser.add_argument(
        "--checkpoint_url",
        type=str,
        default="https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt",
        help=(
            "Checkpoint location of the generative model. Either a local path or "
            "an http(s):// URL; URLs are downloaded into ./artifacts and cached."
        ),
    )
    parser.add_argument(
        "--img_resolution",
        type=int,
        default=518,
        help="conditioning image resolution",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print("=" * 60)
    print("LiTo - Image to 3D Generator")
    print("=" * 60)

    print("\nLoading models...")
    load_models(
        checkpoint_url=args.checkpoint_url,
        download_dir_root=os.path.abspath("artifacts"),
    )

    # Warmup: run full pipeline with a dummy image to trigger torch.compile
    print("\n[Warmup] Running generation with dummy image to compile the model...")
    img_resolution = int(args.img_resolution)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (img_resolution, img_resolution, 3), dtype=np.uint8))
    warmup_dict = preprocess_image(dummy_img, crop=False, remove_bg=False, img_resolution=img_resolution)
    warmup_rgba = warmup_dict["rgba"]  # (h, w, 4rgba) [0, 1]
    for update in generate_3d_with_progress(
        sampling_steps=20,
        cfg_scale=3.0,
        cond_rgba=warmup_rgba,
        compress_spz=False,
    ):
        if update["type"] == "progress":
            print(f"  [Warmup] {update.get('message', update['step'])} {update['progress']}%")
        elif update["type"] == "complete":
            print(f"  [Warmup] Done in {update['total_time']:.1f}s")
        elif update["type"] == "error":
            print(f"  [Warmup] Error: {update['message']}")
    print("[Warmup] Model compilation complete!\n")

    port = int(args.port)
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
