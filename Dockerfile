# Base Python image (slim = lightweight)
FROM python:3.10-slim

# Install system dependencies required for some packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory in the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Launch the FastAPI server
CMD ["python", "run_fast_api.py"]


