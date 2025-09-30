# Dockerfile
# ThermADNet — minimal CPU image with Python + PyTorch + scientific stack
FROM python:3.10-slim

# System deps (just what's needed to run numpy/pandas/matplotlib nicely)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make Python friendlier in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Optional: include Jupyter tools at build time
ARG INCLUDE_JUPYTER=false

# Copy only requirements first (better layer caching)
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install core deps (torch CPU wheels come from PyPI by default)
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    if [ "$INCLUDE_JUPYTER" = "true" ]; then \
        pip install jupyterlab ipykernel; \
    fi

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Copy source (docs + src; data will be mounted at runtime)
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser README.md /app/README.md
COPY --chown=appuser:appuser docs/ /app/docs/

# Default working directory
WORKDIR /app

# Default command (no long-running process; override in `docker run`)
CMD ["python", "-c", "print('ThermADNet container ready. Use -v $(pwd)/data:/app/data and run your scripts in /app/src')"]
