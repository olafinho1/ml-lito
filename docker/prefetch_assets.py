#!/usr/bin/env python
"""Populate non-LiTo model caches used during first startup."""

from huggingface_hub import snapshot_download

import torch

DINO_REPOSITORY = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"

snapshot_download(
    repo_id="microsoft/TRELLIS-image-large",
    revision="25e0d31ffbebe4b5a97464dd851910efc3002d96",
    allow_patterns=[
        "ckpts/ss_enc_conv3d_16l8_fp16.json",
        "ckpts/ss_enc_conv3d_16l8_fp16.safetensors",
        "ckpts/ss_dec_conv3d_16l8_fp16.json",
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ],
)

# LiTo's image conditioner loads this exact model through torch.hub.
torch.hub.load(DINO_REPOSITORY, "dinov2_vitl14_reg")
