# Use the Triton container with Python backend support
FROM nvcr.io/nvidia/tritonserver:25.02-pyt-python-py3

# Set working directory
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    rapidjson-dev libsm6 libxext6 libxrender-dev \
    curl ca-certificates sudo git bzip2 \
    libx11-6 build-essential wget unzip tmux ccache \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and torchvision with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir huggingface_hub transformers tokenizers numpy scikit-learn pyvi vllm autoawq torch qwen-vl-utils

RUN pip install --no-cache-dir transformers accelerate pillow ninja packaging 


# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Expose Triton server ports
EXPOSE 8000 8001 8002

# Set the entry point to Triton server
ENTRYPOINT ["/opt/tritonserver/bin/tritonserver", "--model-repository=/models"]