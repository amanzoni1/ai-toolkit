# ARM64-compatible PyTorch + CUDA base
FROM nvcr.io/nvidia/pytorch:24.01-py3-igpu

LABEL maintainer="amanzoni1"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/amanzoni1/ai-toolkit.git /app/ai-toolkit && \
    cd /app/ai-toolkit && \
    git submodule update --init --recursive

WORKDIR /app/ai-toolkit

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir transformers accelerate peft diffusers safetensors
