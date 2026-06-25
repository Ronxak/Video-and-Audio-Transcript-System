# Optional: reproducible CPU build.
# (For GPU, install locally with the NVIDIA toolkit — see the README.)
FROM python:3.11-slim

# ffmpeg is required to decode audio from media files.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to CPU inside the container.
ENV DEVICE=cpu
ENV COMPUTE_TYPE=int8
ENV WHISPER_MODEL=small
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860
CMD ["python", "app.py"]
