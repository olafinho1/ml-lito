# LiTo: Surface Light Field Tokenization

This website accompanies the research paper: 

**LiTo: Surface Light Field Tokenization, ICLR 2026.**<br>
[Jen-Hao Rick Chang*](https://rick-chang.github.io), [Xiaoming Zhao*](https://xiaoming-zhao.com/), [Dorian Chan](https://dorianchan.com/), [Oncel Tuzel](https://www.onceltuzel.net).

<p align="center"><a href="https://arxiv.org/abs/2603.11047"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://apple.github.io/ml-lito/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>

## Abstract

We propose a latent 3D representation that jointly models object geometry and view-dependent appearance. Our approach leverages the fact that RGB-depth images provide samples of a surface light field. By encoding random subsamples of this surface light field into a compact set of latent vectors, our model learns to represent both geometry and appearance within a unified 3D latent space. This representation can reproduce view-dependent effects such as lighting reflections and Fresnel reflections under complex lighting. We further train an image-to-3D model, enabling the generation of 3D objects with appearances consistent with the lighting and materials in the input. Experiments show that our approach achieves higher reconstruction quality and better separation of geometry and appearance than existing methods.


## Features
- **View-dependent appearance**: It models effects like specular highlight, Fresnel reflection.  
- **Speed**: 4.7 secs image-to-3D generation on a H100 (after torch compile).
- **Align with image**: Generated objects align with image (not in arbitrary coordinate frame). 

## Installation
### Prerequisites
- **System**: We support two platforms, with different capabilities:
  - **Linux with an NVIDIA GPU** (verified on A100, H100, B200): full support — training, the interactive image-to-3D demo, and the tokenizer notebook.
  - **macOS with Apple Silicon** (M-series): supports the interactive image-to-3D demo only, using [MLX](https://github.com/ml-explore/mlx).
- **Software**:   
  - We tested our code with PyTorch 2.5-2.9, paired with the corresponding xformers / flash attention and PyTorch3D. We found that it is most robust to compile these packages on the running system (eg, `pip install xxx --no-build-isolation`).
  - We use [pixi](https://pixi.prefix.dev/) as our environment managing system. We provide lock file to reproduce the entire environment (cuda, pytorch, xformers, etc).

### Installation Steps
1. Clone the repo:
    ```sh
    git clone --recurse-submodules https://github.com/apple/ml-lito.git
    cd ml-lito
    ```

2. Install the dependencies:
    
    We use [pixi](https://pixi.prefix.dev/) to create a virtual environment under `.pixi`. The environment contains cuda and python packages. 
    
    ```sh
    # The following command will install pixi and create the environment. 
    # on ubuntu
    # bash env/setup.sh
    
    # on mac 
    # bash env/setup_mac.sh
    ```

For a reusable Linux CUDA image, including A40-specific build and acceptance
instructions, see [`docs/docker.md`](docs/docker.md).

## Pretrained Models

We provide the following pretrained models:

| Model | Description | Download |
| --- | --- | --- |
| LiTo tokenizer (recommended) | Point-cloud tokenizer with a bug fix over the paper version — use this. | [lito_new.ckpt](https://ml-site.cdn-apple.com/models/lito/lito_new.ckpt) |
| LiTo tokenizer (paper) | Point-cloud tokenizer used in the paper. | [lito.ckpt](https://ml-site.cdn-apple.com/models/lito/lito.ckpt) |
| LiTo image-to-3D (recommended) | Image-to-3D generative model with a bug fix over the paper version — use this. | [lito_dit_rgba.ckpt](https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt) |
| LiTo image-to-3D (paper) | Image-to-3D generative model used in the paper. | [lito_dit.ckpt](https://ml-site.cdn-apple.com/models/lito/lito_dit.ckpt) |

You can pass any of the URLs above directly to the demo (`--checkpoint_url`) or the tokenizer notebook — the checkpoint will be downloaded and cached under `artifacts/` on first use. Or alternatively, you can download them and pass the local path.

## Usage

### Run interactive image-to-3D demo

We use FastAPI to serve an interactive LiTo demo. It runs on both Linux with an NVIDIA GPU and macOS with Apple Silicon. Start a local server with:

```bash
# at repo root (on linux or mac)
pixi run python demos/lito/fastapi_lito_demo.py --port 8000
```

Then open `http://localhost:8000` in your browser to access the demo.

Useful flags:
- `--checkpoint_url`: local path or URL to a generative-model checkpoint (defaults to `lito_dit_rgba.ckpt`). URLs are downloaded into `./artifacts/` and cached.
- `--port`: port to serve on (default `7860`).

**Note**: 
- When the demo starts, it automatically runs one generation (for compilation) for one-time compilation — `torch.compile` on CUDA, MLX compilation on macOS. 
- On Mac, the code will print `xformers` and `flash_attn` not found, and they are normal.
- The typical runtime (20 heun steps with CFG) on H100 is ~4.6 seconds, and on M4 Max is ~160 seconds. 


### Using the pretrained point-cloud tokenizer

See [`notebooks/demo_tokenizer.ipynb`](/notebooks/demo_tokenizer.ipynb) for a worked example. The notebook currently requires Linux with an NVIDIA GPU.

The notebook walks through:
1. Loading the tokenizer checkpoint (`lito_new.ckpt` — see [Pretrained Models](#pretrained-models)).
2. Loading an example point cloud from [`notebooks/assets/bunny.npz`](/notebooks/assets/bunny.npz).
3. Encoding the point cloud into latent tokens.
4. Decoding the latents into 3D Gaussians, a mesh, and a resampled point cloud — each saved as a PLY under `notebooks/recon_results/`.

The pretrained tokenizer is trained with 2^20 input points and 8192 output tokens (32-dim features), but we found it is robust to different point counts and token counts — feel free to experiment with other values.



## Code structure

The repo is structured into 3 main packages. They are installed as editable pip package when you ran `pixi install` or `bash env/setup.sh`.
1. **lito**: contains pytorch lightning trainers and model definitions.  It is in `src/lito`. 
2. **plibs**: contains 3D utilities like sampling points from meshes, rendering with gsplat and nvdiffrast, our data structures for RGBD images and meshes.    
3. **blender_rendering**:  contains our blender rendering scripts to render RGBD images with blender. 



## Training

To train the tokenizer to learn the latent representation:
```bash
# at repo root
# option 1: using pixi 
pixi run python scripts/train.py --config configs/lito/tokenizer/lito_8k32.yaml

# option 2: activate pixi environment  
eval "$(pixi shell-hook -e default)"  
python scripts/train.py --config configs/lito/tokenizer/lito_8k32.yaml
```

Similarly, to learn the image to 3D generative model:
```bash
# at repo root
eval "$(pixi shell-hook -e default)"  
python scripts/train.py --config configs/lito/generator/lito_dit_8k32.yaml
```


## Coordinate systems
We use the same coordinate system as that used by Open3D and Gaussian Splatting: x points right, y points up, and z points toward the viewer.

For the image coordinate system: x points to right of the image, y points to the bottom of the image, z (depth) increases away from the camera, and the origin is the top-left corner of the image. 


## Data
We provide our functions to rendering with Blender 4.2 and functions that tar the rendered samples for our dataloader.

See [notebooks/render_data.ipynb](/notebooks/render_data.ipynb) for how we render a mesh into multiview RGBD images, 
and how to save it into a tar. 

The functions can be used to construct the entire dataset.

### Data split
We provide our train/valid/test splits in [assets/data_splits/obj_split_dict.json](/assets/data_splits/obj_split_dict.json) and  [assets/data_splits/objxl_split_dict.json](/assets/data_splits/objxl_split_dict.json), which are for Objaverse and ObjaverseXL, respectively.  

The splits are the intersection between a split created by random sampling and the TRELLIS500k dataset. We also filter out samples that contain significant transparent surfaces.    

In result, we have 84,825 training samples from Objaverse and 155,275 training samples from ObjaverseXL (total 240.1k training samples).   



## License

- Repository is released under [LICENSE](./LICENSE). 
- All generated samples provided here are licensed under [LICENSE_generated_samples](./LICENSE_generated_samples).
- All pretrainede models provided here are licensed under [LICENSE_MODEL](./LICENSE_MODEL).

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](./ACKNOWLEDGEMENTS) for more details. 

We also thank Muhammed Kocabas for his contribution to the FastAPI demo.


## Citations

```
@inproceedings{chang2026lito,
    author    = {Jen-Hao Rick Chang$^\ast$ and Xiaoming Zhao$^\ast$ and Dorian Chan and Oncel Tuzel},
    title     = {{LiTo: Surface Light Field Tokenization}},
    booktitle = {International Conference on Learning Representations},
    year      = {2026},
}
```
