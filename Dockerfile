FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# ffmpeg: normaliza áudios; libasound2/libssl3: exigidos pelo SDK do Azure Speech;
# libgl1/libglib2.0-0: dependências de runtime do OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libasound2 libssl3 ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Deno: runtime JavaScript que o yt-dlp exige para baixar do YouTube (binário oficial, ~100 MB)
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

WORKDIR /app
COPY requirements.txt requirements-video.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# YOLOv8-pose em CPU (PyTorch só-CPU, ~250 MB). Use --build-arg WITH_YOLO=false para pular:
# o vídeo cai no plano B (OpenCV) e a imagem fica bem menor.
ARG WITH_YOLO=true
RUN if [ "$WITH_YOLO" = "true" ]; then \
      pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir -r requirements-video.txt; \
    fi

COPY src ./src
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "vitaltrace.web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
