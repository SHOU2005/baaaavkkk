# Root-level Dockerfile for Render deployment
# Copy of backend/Dockerfile for platforms that expect Dockerfile at root

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Copy requirements first for better caching
COPY backend/requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code from backend directory
COPY backend/ .

# Create required directories
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

# Expose port (will be overridden by Render/PORT env var)
EXPOSE 8000

# Run the application - use PORT env var if available
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

