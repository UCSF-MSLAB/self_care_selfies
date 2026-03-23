FROM python:3.9-slim

WORKDIR /app

# Added 'time' to the install list
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 time && \
    rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python-headless mediapipe

COPY . .
