# DeepFlyer ML Components Docker Image
# Simple container with just the ML/RL components for integration

FROM python:3.10-slim

# Install system dependencies for ML
RUN apt-get update && apt-get install -y \
    python3-opencv \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /deepflyer

# Copy ML requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only ML components (not deployment infrastructure)
COPY rl_agent/ ./rl_agent/
COPY api/ml_interface.py ./api/ml_interface.py
COPY weights/ ./weights/
COPY msg/ ./msg/
COPY nodes/ ./nodes/

# Set Python path
ENV PYTHONPATH=/deepflyer:$PYTHONPATH

# Entry point for ML components
CMD ["python", "-m", "rl_agent"] 