# Sentinel ML (Model B + E) GPU image. Python 3.11 + PyTorch 2.1.0 (CUDA 12.1 wheels bundle the CUDA libs,
# so only the NVIDIA *driver* is needed on the host). Code and data are bind-mounted, not baked in.
FROM python:3.11-slim-bookworm

ARG UID=1000
ARG GID=1000
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libglib2.0-0 libgl1 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r /tmp/requirements-gpu.txt

# Run as a normal user with the host's UID/GID so files written to data/outputs are not root-owned.
RUN groupadd -g ${GID} dev && useradd -m -u ${UID} -g ${GID} dev
USER dev
ENV HOME=/home/dev PYTHONPATH=/workspace/ml/src
WORKDIR /workspace
CMD ["bash"]
