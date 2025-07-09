# DeepFlyer Production Docker Image
FROM ros:humble-ros-base-jammy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PX4 messages
RUN apt-get update && apt-get install -y \
    ros-humble-px4-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /deepflyer

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install YOLO and ClearML
RUN pip3 install --no-cache-dir \
    ultralytics==8.0.200 \
    clearml==1.13.2 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy the entire project
COPY . .

# Build custom messages
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select deepflyer_msgs

# Create entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/ros/humble/setup.bash\n\
source /deepflyer/install/setup.bash\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set environment variables
ENV ROS_DOMAIN_ID=1
ENV PYTHONPATH=/deepflyer:$PYTHONPATH
ENV YOLO_MODEL_PATH=/deepflyer/weights/best.pt

# Expose ROS2 DDS ports
EXPOSE 7400-7500/udp

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "deepflyer", "full_system.launch.py"] 