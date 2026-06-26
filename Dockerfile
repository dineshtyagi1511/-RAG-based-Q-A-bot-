# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile
# Stage 1 (builder) : install all Python deps into a venv
# Stage 2 (runtime) : copy only the venv + app code — minimal image
#
# CPU-only build: avoids the 3+ GB CUDA runtime that llm-guard would pull in
# if you install the default torch.  Set ENABLE_LLM_GUARD=false (default) to
# skip llm-guard entirely and keep the image small.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip wheel

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Optional: install llm-guard with CPU-only torch ──────────────────────────
# Uncomment the block below if ENABLE_LLM_GUARD=true in your env.
# This adds ~2 GB to the image.
#
# RUN pip install --no-cache-dir \
#       torch --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir llm-guard
# ─────────────────────────────────────────────────────────────────────────────


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Runtime system libs (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --chown=appuser:appgroup . .

# Streamlit config — run on 0.0.0.0 and skip browser auto-open
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]