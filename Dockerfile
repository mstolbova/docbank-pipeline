# DocBank pipeline web app — portable container.
# Runs on any Docker host: Hugging Face Spaces (Docker), Render, Fly.io, etc.
#
# Stack: DocLayout-YOLO (YOLOv10) detection + PaddleOCR text + pix2tex formulas,
# output as a SEARCHABLE PDF (reportlab: original image + invisible OCR layer).
# No LaTeX needed. Needs ~3-4 GB RAM; first upload downloads OCR weights.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Where the app writes per-job artefacts + caches (must be writable).
    DATA_ROOT=/tmp/docbank \
    # DocLayout-YOLO detector checkpoint (read via cfg.layout_weights).
    LAYOUT_WEIGHTS=/app/doclayout_yolo_docstructbench_imgsz1024.pt \
    # PaddleOCR recognition language (Korean docs with Hanja). Override as needed.
    OCR_LANG=korean \
    # Keep every model/cache download under writable /tmp (HF Spaces runs as
    # a non-root user; /app is read-only at runtime there).
    HF_HOME=/tmp/cache/hf \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/cache/mpl \
    # HF Spaces runs as a non-root user; give PaddleOCR (~/.paddleocr) and
    # Ultralytics a writable HOME so their downloads don't fail at runtime.
    HOME=/tmp/home \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    # Default port; platforms that inject $PORT (Render) override at runtime.
    PORT=7860

# ---- system deps: runtime libs for opencv / paddle (no TeX Live anymore) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 \
        ca-certificates curl \
        fonts-nanum
        fonts-noto-cjk
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- python deps ----
# CPU-only torch first so ultralytics / doclayout-yolo / pix2tex reuse it
# instead of pulling the multi-GB CUDA wheel.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Use the CPU deploy requirements (no CUDA torch index).
COPY requirements-deploy.txt ./
RUN pip install -r requirements-deploy.txt

# ---- app code + baked model weights (incl. the DocLayout-YOLO .pt) ----
COPY . .

# Make runtime cache/output dirs world-writable (non-root hosts like HF Spaces).
RUN mkdir -p /tmp/docbank /tmp/cache/hf /tmp/cache/mpl /tmp/home /tmp/Ultralytics \
    && chmod -R 777 /tmp/docbank /tmp/cache /tmp/home /tmp/Ultralytics

EXPOSE 7860

# Bind to 0.0.0.0 and the platform-provided $PORT. serve() uses waitress.
CMD ["sh", "-c", "python -m docbank_pipeline serve --host 0.0.0.0 --port ${PORT:-7860}"]
