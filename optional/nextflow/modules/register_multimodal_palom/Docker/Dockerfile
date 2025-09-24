# syntax=docker/dockerfile:1.7
FROM mambaorg/micromamba:1.5.10-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /workspace

# --- add system libs needed by OpenCV (libGL) and friends ---
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
USER mambauser
# ------------------------------------------------------------

# Copy environment spec
COPY environment.yml /tmp/environment.yml

# Remove prefix and build env
RUN sed '/^prefix:/d' /tmp/environment.yml > /tmp/env.yml \
 && micromamba create -y -n palom -f /tmp/env.yml \
 && micromamba clean -a -y

# Copy your Python code into the image
COPY RegisterOneImage_palom.py /workspace/

# Quick sanity check
RUN micromamba run -n palom python -c "import cv2, sys; print('cv2 ok:', cv2.__version__)"

# Always run inside the env
ENTRYPOINT ["micromamba", "run", "-n", "palom"]
CMD ["bash"]

COPY Dockerfile /docker/
