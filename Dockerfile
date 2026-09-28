# RunPod serverless worker for Laya. The CUDA libraries come from the torch wheel, so a slim
# Python base is enough; the host driver is injected by the NVIDIA container runtime.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}

# cu128 wheels run on any RunPod host whose driver reports CUDA >= 12.8 (the endpoint is
# pinned to those in scripts/deploy.py). torch 2.11 is the newest cu128 build; laya needs
# >= 2.0. Build with TORCH_INDEX=cpu for a local smoke test on a machine without a GPU.
ARG TORCH_INDEX=cu128
ARG TORCH_VERSION=2.11.0
ARG LAYA_VERSION=0.3.21
# Checkpoints baked into the image (~2.3 GB for all three). Empty = download on cold start.
ARG LAYA_BAKE_MODELS=english,multilingual,typed-decisions

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false

RUN pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
    && python -c "import torch, sys; v = torch.version.cuda; want = '${TORCH_INDEX}'; \
sys.exit(0 if (want == 'cpu' and v is None) or (want.startswith('cu') and v and want == 'cu' + v.replace('.', '')) \
else 'torch %s does not match TORCH_INDEX=%s' % (torch.__version__, want))"

WORKDIR /opt/laya-runpod
COPY requirements.txt ./
RUN pip install "laya[serve]==${LAYA_VERSION}" -r requirements.txt && pip check

COPY scripts/bake_models.py ./scripts/
RUN LAYA_BAKE_MODELS="${LAYA_BAKE_MODELS}" python scripts/bake_models.py

# Runtime defaults; every one can be overridden in the RunPod template's env.
# TORCH_DISABLE_NATIVE_JIT: newer torch swaps some eager CUDA ops for Triton kernels compiled on
# first use, which needs a C compiler this image doesn't carry (upstream laya #365).
# HF_HUB_OFFLINE: the baked cache is complete; set it to 0 if you bake nothing.
ENV LAYA_DEVICE=cuda \
    LAYA_PRELOAD=1 \
    LAYA_WARMUP=1 \
    TORCH_DISABLE_NATIVE_JIT=1 \
    HF_HUB_OFFLINE=1

COPY src/handler.py ./
COPY test_input.json ./

CMD ["python", "-u", "/opt/laya-runpod/handler.py"]
