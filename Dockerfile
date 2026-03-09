# ─────────────────────────────────────────────
#  Dockerfile — Lavanya's RAG API (FastAPI)
# ─────────────────────────────────────────────

# Use slim Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY phase4_fast_api.py .
COPY knowledge_base.json .

# Copy pre-built ChromaDB (already embedded in Phase 2)
COPY chroma_db/ ./chroma_db/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the API
CMD ["uvicorn", "phase4_fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
