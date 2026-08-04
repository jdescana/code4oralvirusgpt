FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        libsm6 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/oral_virus_gpt

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

ENV TORCH_DISTRIBUTED_DEBUG=DETAIL \
    NCCL_ASYNC_ERROR_HANDLING=1

CMD ["python", "-m", "oral_virus_gpt.runners.cli", "--help"]
