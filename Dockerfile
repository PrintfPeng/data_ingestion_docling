# =====================================================================
# AI Data Ingestion Pipeline - Backend Dockerfile
#
# Base: PyTorch 2.4 + CUDA 12.1 + cuDNN 9 (Ubuntu 22.04)
# ให้ GPU (NVIDIA) รัน sentence-transformers, docling, torch ได้
# =====================================================================
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf_cache \
    TZ=Asia/Bangkok

# ---------------------------------------------------------------------
# System dependencies
#   libgl1 + libglib2.0-0 : opencv (cv2)
#   ghostscript + tk      : camelot-py (table extraction)
#   poppler-utils         : docling / pdf tools
#   libmagic1             : docling file-type detection
#   fonts-thai-tlwg       : rendering Thai text in matplotlib/PIL
#   curl                  : healthcheck
# ---------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ghostscript \
        poppler-utils \
        libmagic1 \
        tk \
        fonts-thai-tlwg \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------
# Python dependencies (layer เดียว - cache ใหม่เฉพาะเมื่อ requirements.txt เปลี่ยน)
# ---------------------------------------------------------------------
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install "uvicorn[standard]" "gunicorn"

# ---------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------
COPY . .

# โฟลเดอร์ข้อมูล runtime (จะถูก mount override โดย docker-compose)
RUN mkdir -p /app/ingested /app/uploads /app/chroma_db /app/logs /opt/hf_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# uvicorn + reload OFF สำหรับ production
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
