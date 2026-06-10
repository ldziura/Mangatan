FROM python:3.12-slim
WORKDIR /app/ocr-server

# System libraries: OpenCV (pulled by mokuro/comic-text-detector) needs libGL + glib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Bake model weights into the image so the first request is instant and the container can
# run fully offline: comictextdetector.pt (~76 MB) -> $XDG_CACHE_HOME, manga-ocr-base
# (~444 MB) -> $HF_HOME.
ENV HF_HOME=/models/hf \
    XDG_CACHE_HOME=/models/cache

RUN pip install --no-cache-dir uv
COPY ocr-server/pyproject.toml ocr-server/uv.lock ./
# uv.lock is committed with the full resolved tree (CUDA torch on linux, manga-ocr,
# mokuro, transformers 4.x). --frozen = reproducible, network-pinned builds.
RUN uv sync --frozen --no-dev

COPY ocr-server/ .

# Pre-download both models (CPU at build time; runtime still uses CUDA).
RUN uv run python -c "from mokuro import MangaPageOcr; MangaPageOcr(pretrained_model_name_or_path='kha-white/manga-ocr-base', force_cpu=True)"

EXPOSE 5588
# No -e flag: server.py defaults to 'mangaocr+lens' (local manga-ocr first, Google Lens
# fallback). Override with the OCR_ENGINE env var or '-e lens' for pure cloud OCR.
CMD ["uv", "run", "server.py"]
