FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends iputils-ping && \
    apt-get install -y curl && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install uv

# Copy requirements and install
COPY requirements.txt .
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000