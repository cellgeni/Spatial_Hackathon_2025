# Atomic Dockerfile for "example-tool"
# Goal: only install what's strictly needed to run the CLI

FROM python:3.11-slim

# Install only minimal runtime dependencies
RUN pip install --no-cache-dir click==8.1.7 rich==13.7.1

WORKDIR /work

# Copy the CLI into PATH
COPY bin/example-tool /usr/local/bin/example-tool
RUN chmod +x /usr/local/bin/example-tool

ENTRYPOINT ["example-tool"]
