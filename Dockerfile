# ============================================================
# Stage 1: Classification (lightweight, runtime image)
# ============================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS classification

WORKDIR /model-export-example

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository 'ppa:deadsnakes/ppa'

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    curl \
    git \
    libgl1

RUN apt-get remove -y python3.8 && \
    ln -sfn /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sfn /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

COPY requirements.base.txt requirements.base.txt
RUN python3 -m pip install --no-cache-dir -r requirements.base.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Object Detection (extends with devel image + mmdetection)
# ============================================================
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS detection

WORKDIR /model-export-example

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository 'ppa:deadsnakes/ppa'

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    curl \
    git \
    libgl1

RUN apt-get remove -y python3.8 && \
    ln -sfn /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sfn /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

COPY requirements.base.txt requirements.base.txt
RUN python3 -m pip install --no-cache-dir -r requirements.base.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# --- mmdetection dependencies ---
RUN pip install setuptools wheel ninja

RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    pip install "mmcv==2.1.0" \
    --no-binary=mmcv \
    --force-reinstall \
    --no-cache-dir \
    --no-build-isolation

RUN mkdir -p /app/notebooks/mmdetection && \
    git -C /app/notebooks/mmdetection init && \
    git -C /app/notebooks/mmdetection remote add origin https://github.com/open-mmlab/mmdetection.git && \
    git -C /app/notebooks/mmdetection fetch --depth 1 origin cfd5d3a985b0249de009b67d04f37263e11cdf3d && \
    git -C /app/notebooks/mmdetection checkout FETCH_HEAD
RUN pip install -v -e /app/notebooks/mmdetection --no-build-isolation