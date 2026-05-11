#!/bin/bash
# glibc 2.39  (getconf GNU_LIBC_VERSION)
# torch2.9.1+cu12.8

# xformers (might need to compile from source if want to use on B200)
pip install xformers==v0.0.33.post2 --no-build-isolation

# flash attention
MAX_JOBS=32 pip install flash-attn --no-build-isolation

# pytorch3d
pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation

# open3d (you might need to follow https://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html
# if you have a headless machine)
pip install open3d --no-build-isolation

# gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation

# fused_ssim
pip install git+https://github.com/rahul-goel/fused-ssim --no-build-isolation

# torchsparse
# apt install -y libsparsehash-dev  # potentially needed for compiling torchsparse
pip install git+https://github.com/mit-han-lab/torchsparse.git --no-build-isolation

#"xformers ; sys_platform == 'linux'",
#"flash_attn ; sys_platform == 'linux'",
#"pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git ; sys_platform == 'linux'",
#"open3d",
#"gsplat @ git+https://github.com/nerfstudio-project/gsplat.git ",
#"fused_ssim @ git+https://github.com/rahul-goel/fused-ssim ; sys_platform == 'linux'",
#"torchsparse @ git+https://github.com/mit-han-lab/torchsparse.git ; sys_platform == 'linux'",

