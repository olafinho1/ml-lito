# syntax=docker/dockerfile:1.7
FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PIXI_VERSION=v0.70.2
ARG MAX_JOBS=8
ARG LITO_CUDA_ARCH_LIST=8.6

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIXI_HOME=/opt/pixi \
    PATH=/opt/pixi/bin:/opt/ml-lito/.pixi/envs/default/bin:${PATH} \
    LITO_REPO=/opt/ml-lito \
    LITO_ENV=/opt/ml-lito/.pixi/envs/default \
    LITO_CUDA_ARCH_LIST=${LITO_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    CUDA_HOME=/opt/ml-lito/.pixi/envs/default \
    TORCH_CUDA_ARCH_LIST=${LITO_CUDA_ARCH_LIST} \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/opt/ml-cache/huggingface \
    TORCH_HOME=/opt/ml-cache/torch \
    U2NET_HOME=/opt/ml-cache/rembg \
    XDG_CACHE_HOME=/opt/ml-cache/xdg \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        awscli \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        htop \
        less \
        libfontconfig1 \
        libglu1-mesa \
        libjpeg-dev \
        libosmesa6-dev \
        libpulse0 \
        libsm6 \
        libsndfile1 \
        libsparsehash-dev \
        libwayland-client0 \
        libwayland-cursor0 \
        libwayland-egl1 \
        libx11-6 \
        libxcb-xinerama0 \
        libxcb1 \
        libxfixes3 \
        libxi6 \
        libxkbcommon0 \
        libxrandr2 \
        libxrender1 \
        libxshmfence1 \
        libxxf86vm1 \
        nano \
        ninja-build \
        openssh-client \
        openssh-server \
        pkg-config \
        rsync \
        tmux \
        unzip \
        vim-tiny \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /opt/ml-cache /opt/pixi/bin /workspace \
    && git lfs install --system

RUN curl -fsSL --retry 5 \
        "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
        | tar -xz -C /usr/local/bin \
    && mv /usr/local/bin/pixi /opt/pixi/bin/pixi \
    && pixi --version

WORKDIR /opt/ml-lito
COPY . /opt/ml-lito

RUN test -f third_party/TRELLIS/trellis/__init__.py \
    && pixi install --locked -e default \
    && eval "$(pixi shell-hook -e default)" \
    && bash env/scripts/setup_spz.sh \
    && bash env/scripts/setup_trellis.sh \
    && bash env/scripts/compile_cuda_packages.sh \
    && python -m pip check \
    && rm -rf /tmp/lito-package-build /root/.cache/pip

COPY docker/entrypoint.sh /usr/local/bin/lito-entrypoint
RUN chmod 0755 /usr/local/bin/lito-entrypoint \
    && chmod 0755 docker/build.sh \
    && mkdir -p /workspace/ml-lito \
    && ln -s /opt/ml-lito/artifacts /workspace/artifacts

WORKDIR /workspace
EXPOSE 22 8000 8888
ENTRYPOINT ["/usr/local/bin/lito-entrypoint"]
CMD ["bash"]

FROM runtime AS ready

ARG LITO_TOKENIZER_URL=https://ml-site.cdn-apple.com/models/lito/lito_new.ckpt
ARG LITO_GENERATOR_URL=https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt
ARG LITO_TOKENIZER_SHA256=fedb56ee0de93ba6fced8d25c16c25993f563deb7c33a2a07e036dd2380b2440
ARG LITO_GENERATOR_SHA256=60971f6f37bc08c6e77873d95c5468bdb7d1a3389b4a4f4be039a3da1fb22f05

RUN mkdir -p /opt/ml-lito/artifacts \
    && curl -fL --retry 8 --retry-all-errors \
        "${LITO_TOKENIZER_URL}" -o /opt/ml-lito/artifacts/lito_new.ckpt \
    && curl -fL --retry 8 --retry-all-errors \
        "${LITO_GENERATOR_URL}" -o /opt/ml-lito/artifacts/lito_dit_rgba.ckpt \
    && echo "${LITO_TOKENIZER_SHA256}  /opt/ml-lito/artifacts/lito_new.ckpt" \
        | sha256sum --check - \
    && echo "${LITO_GENERATOR_SHA256}  /opt/ml-lito/artifacts/lito_dit_rgba.ckpt" \
        | sha256sum --check - \
    && python /opt/ml-lito/docker/prefetch_assets.py

ENV LITO_TOKENIZER_CHECKPOINT=/opt/ml-lito/artifacts/lito_new.ckpt \
    LITO_GENERATOR_CHECKPOINT=/opt/ml-lito/artifacts/lito_dit_rgba.ckpt
