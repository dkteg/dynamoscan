FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    strace \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY dynamoscan/ ./dynamoscan/

# Install base dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -e .

# Install extra packages if provided
COPY extra_requirements.txt* /tmp/
RUN if [ -f /tmp/extra_requirements.txt ]; then \
        pip install --no-cache-dir --root-user-action=ignore -r /tmp/extra_requirements.txt; \
    fi && \
    rm -f /tmp/extra_requirements.txt

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash scanner && \
    chown -R scanner:scanner /app

# Switch to non-root user
USER scanner

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["dynamoscan"]
