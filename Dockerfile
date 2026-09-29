FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MPLBACKEND=Agg
ENV PIP_NO_CACHE_DIR=1
ENV DISPLAY=:99

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    pkg-config \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    libgtk-3-0 \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrender1 \
    libsm6 \
    libxrandr2 \
    libxxf86vm1 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libdbus-1-3 \
    libnss3 \
    libasound2 \
    libpulse0 \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    xvfb \
    xauth \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /app/requirements-docker.txt

RUN python -m pip install --upgrade \
    "pip==24.3.1" \
    "setuptools==70.3.0" \
    "wheel==0.43.0" \
    "packaging==24.0"

RUN python -m pip install -r /app/requirements-docker.txt

RUN python -m pip install "psychopy>=2024.1.4,<2025.1.0" --no-deps

RUN python -m pip install --force-reinstall \
    "setuptools==70.3.0" \
    "packaging==24.0" \
    "pyglet==1.5.27"

COPY . /app

RUN python -m pip install -e . --no-deps

CMD ["python", "-m", "tobii_pytracker.main", "--eyetracker_config_file", "./configs/mouse_eyetracker_config.yaml", "--enable_eyetracker"]
