# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_CACHE_DIR=/tmp/.uv-cache

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy the project files
COPY . .

# Install dependencies using uv
RUN uv sync --frozen

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
