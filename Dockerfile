FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

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
    git

RUN apt-get remove -y python3.8 && \ 
    ln -sfn /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sfn /usr/bin/python3 /usr/bin/python

# https://stackoverflow.com/questions/70260339/cant-run-pip-on-python-3-11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

COPY requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126