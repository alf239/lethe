# Lethe Container - Safe Mode
# Access restricted to /workspace only

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install agent-browser for browser automation
RUN npm install -g agent-browser

# Install uv (move to /usr/local/bin so non-root user can access)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ 2>/dev/null || true
ENV PATH="/usr/local/bin:$PATH"

# Create workspace directory (this will be mounted from host)
RUN mkdir -p /workspace /app

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY config/ ./config/

# Install dependencies (CPU-only PyTorch to save ~2GB)
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN uv sync --frozen --index-strategy unsafe-best-match

# Create non-root user for safety
RUN useradd -m -s /bin/bash lethe
RUN chown -R lethe:lethe /app /workspace

USER lethe

# Environment (must match pydantic-settings field names)
ENV WORKSPACE_DIR=/workspace
ENV MEMORY_DIR=/workspace/data/memory
ENV LETHE_CONFIG_DIR=/app/config

# Run
CMD ["uv", "run", "lethe"]
