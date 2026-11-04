FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# ffmpeg: normaliza áudios; libasound2/libssl3: exigidos pelo SDK do Azure Speech
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libasound2 libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts

EXPOSE 8000 8501
CMD ["uvicorn", "medmonitor.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
