# DocBank pipeline web app — portable container.
# Runs on any Docker host: Hugging Face Spaces (Docker), Render, Fly.io, etc.
#
# Stack: DocLayout-YOLO detection + PaddleOCR text + pix2tex formulas,
# output as searchable/reconstructed PDF.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_ROOT=/tmp/docbank \
    LAYOUT_WEIGHTS=/app/doclayout_yolo_docstructbench_imgsz1024.pt \
    OCR_LANG=korean \
    HF_HOME=/tmp/cache/hf \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/cache/mpl \
    HOME=/tmp/home \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=7860

# ---- system deps: runtime libs for opencv / paddle + fonts ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ca-certificates \
        curl \
        fonts-nanum \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- python deps ----
# CPU-only torch first so ultralytics / doclayout-yolo / pix2tex reuse it
# instead of pulling the multi-GB CUDA wheel.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Use the CPU deploy requirements.
COPY requirements-deploy.txt ./
RUN pip install -r requirements-deploy.txt

# ---- app code + baked model weights ----
COPY . .

# Make runtime cache/output dirs world-writable.
RUN mkdir -p /tmp/docbank /tmp/cache/hf /tmp/cache/mpl /tmp/home /tmp/Ultralytics \
    && chmod -R 777 /tmp/docbank /tmp/cache /tmp/home /tmp/Ultralytics

EXPOSE 7860

# Bind to 0.0.0.0 and the platform-provided $PORT. serve() uses waitress.
CMD ["sh", "-c", "python -m docbank_pipeline serve --host 0.0.0.0 --port ${PORT:-7860}"]

