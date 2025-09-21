# Extended Dockerfile for "example-tool"
# This builds on top of the atomic image and adds extra tools.

# Start from the atomic runtime
FROM example-tool:latest

# Add extra system or Python tools (only here, not in atomic)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      git \
    && rm -rf /var/lib/apt/lists/*

# Optionally install extra Python libs
RUN pip install --no-cache-dir pandas==2.2.2 matplotlib==3.9.2

# Same entrypoint as atomic
ENTRYPOINT ["example-tool"]
