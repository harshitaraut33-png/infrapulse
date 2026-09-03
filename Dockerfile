# InfraPulse — container image.
# Works on Hugging Face Spaces (Docker SDK), Render, Railway, Fly, or any host that
# runs a container.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# OpenCV needs these even in its headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only build of PyTorch FIRST and explicitly.
# The default PyPI wheel on Linux pulls the CUDA runtime — roughly 2.5 GB of GPU
# libraries we will never use on a CPU host. It blows past free-tier build limits and
# balloons memory. This index serves CPU wheels only.
RUN pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Writable locations for the database and uploaded photographs. Hosts run the container
# as a non-root user, so these must be world-writable.
ENV INFRAPULSE_DB=/app/data/infrapulse.db \
    INFRAPULSE_UPLOADS=/app/data/uploads \
    INFRAPULSE_MODEL=/app/model/infrapulse_model.pt
RUN mkdir -p /app/data/uploads && chmod -R 777 /app/data

# Hugging Face Spaces expects 7860; other hosts inject their own $PORT.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "python seed_staff.py; uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
