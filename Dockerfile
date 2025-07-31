# Gunakan image Python yang ringan
FROM python:3.9-slim

# Railway injects env vars at build time, declare them here
ARG RAILWAY_SERVICE_NAME
ARG RAILWAY_ENVIRONMENT

# Debug: tampilkan informasi build
RUN echo "Service: $RAILWAY_SERVICE_NAME | Environment: $RAILWAY_ENVIRONMENT"

# Set workdir
WORKDIR /app

# Salin file requirements terlebih dahulu untuk cache pip
COPY requirements.txt .

# Install dependencies dan gunakan cache pip
RUN --mount=type=cache,id=s/railway-pip-cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Salin seluruh isi project ke container
COPY . .

# Default command saat container dijalankan
CMD ["python", "run_pipeline.py"]
