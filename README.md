# DeepFlyer - Educational Drone Reinforcement Learning Platform

DeepFlyer is a production-ready educational drone reinforcement learning platform that teaches autonomous navigation through direct RL control using the P3O (Procrastinated Policy Optimization) algorithm.

## Overview

DeepFlyer implements **direct reinforcement learning control** for drones using **PX4-ROS-COM** as the primary communication protocol. Unlike traditional approaches that use RL to tune PID controllers, our approach directly outputs control commands to the drone, providing greater flexibility and performance.

## Key Features

- **Direct RL Control**: P3O algorithm outputs control commands directly (thrust, roll rate, pitch rate, yaw rate)
- **PX4-ROS-COM Integration**: Lower latency communication with PX4 flight controllers
- **ZED Camera Integration**: Real-time visual perception for navigation
- **Educational Focus**: Intuitive interface for learning reinforcement learning concepts
- **Safety Layer**: Prevents dangerous actions while maintaining learning flexibility
- **Sim-to-Real**: Train in simulation, deploy on real hardware

## Implementation Status ✅

All core components are implemented with production-ready code:

### 1. **Custom ROS2 Message Types** (`/msg/`)
- **VisionFeatures.msg** - YOLO11 vision processing results
- **RLAction.msg** - 3D action commands for drone control
- **RewardFeedback.msg** - Educational reward component breakdowns
- **CourseState.msg** - Course navigation and progress tracking
- **DroneState.msg** - Comprehensive drone state information

### 2. **ZED Mini Camera Integration** (`rl_agent/env/zed_integration.py`)
- **ZEDInterface** abstract base class for camera integration
- **ROS-based** and **Direct SDK** interface implementations
- **Mock interface** for testing without hardware

### 3. **PX4-ROS-COM Communication** (`rl_agent/env/px4_comm/`)
- **PX4Interface** - Primary PX4-ROS-COM communication (recommended)
- **MAVROSBridge** - Legacy MAVROS communication (fallback)
- **MessageConverter** - Coordinate transformations and message utilities

### 4. **Production Environment Classes** (`rl_agent/env/px4_base_env.py`)
- **PX4DroneEnv** - Main environment class for drone control
- **PX4BaseEnv** - Common functionality base class

### 5. **P3O Algorithm** (`rl_agent/algorithms/p3o.py`)
- **Complete P3O implementation** (Procrastinated Proximal Policy Optimization)
- **Procrastination mechanism** for stable learning
- **GAE advantage estimation** and **policy/value networks**

## Installation

### Prerequisites

- ROS2 (Rolling or Humble)
- Python 3.8 or later
- NVIDIA GPU recommended for training
- PX4 flight controller (for hardware deployment)

### Setup

1. Create a ROS2 workspace and clone this repository:

```bash
mkdir -p ~/deepflyer_ws/src
cd ~/deepflyer_ws/src
git clone https://github.com/your-username/DeepFlyer.git
```

2. Install Python dependencies:

```bash
cd DeepFlyer
pip install -r requirements.txt
```

3. Build the ROS2 workspace:

```bash
cd ~/deepflyer_ws
colcon build
source install/setup.bash
```

## Quick Start

### Create Training Environment
```python
from rl_agent.env import make_px4_env

# Create environment
env = make_px4_env(
    use_zed=True,  # Enable camera
    spawn_position=(0.0, 0.0, 1.0)
)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Training a Direct Control Agent

```bash
cd ~/deepflyer_ws
source install/setup.bash
python scripts/test_direct_control.py --train --collect_time 300 --save_model ./models/direct_p3o_agent.pt
```

### Testing a Trained Agent

```bash
cd ~/deepflyer_ws
source install/setup.bash
python scripts/test_direct_control.py --test --test_time 120 --load_model ./models/direct_p3o_agent.pt
```

## Communication Architecture

### Primary: PX4-ROS-COM (Recommended)
- **Direct PX4 integration** via PX4-ROS-COM DDS protocol
- **Lower latency** and **higher performance** than MAVROS
- **Native ROS2 integration** with PX4 flight controller

### Legacy: MAVROS (Fallback Support)
- Traditional MAVROS bridge for backward compatibility
- Higher latency compared to PX4-ROS-COM

## P3O Algorithm

The project implements the P3O algorithm, an advanced reinforcement learning method specifically designed for drone navigation tasks. Key features include:

- **Procrastinated Updates**: Postpones on-policy updates to improve sample efficiency
- **Blended Learning**: Combines on-policy and off-policy gradients for better stability
- **Adaptive Exploration**: Uses entropy regularization to maintain appropriate exploration

### P3O Configuration
```python
from rl_agent.algorithms import P3O, P3OConfig

# Configure P3O
config = P3OConfig(
    procrastination_factor=0.1,  # Key P3O parameter
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10
)

# Create and train P3O agent
agent = P3O(env, config)
agent.train(total_timesteps=100000)
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   P3O Agent     │    │  PX4-ROS-COM     │    │  PX4 Flight     │
│                 │◄──►│  Interface       │◄──►│  Controller     │
│  • Policy Net   │    │  (Primary)       │    │                 │
│  • Value Net    │    └──────────────────┘    └─────────────────┘
│  • Procrastin.  │    ┌──────────────────┐    ┌─────────────────┐
└─────────────────┘    │  MAVROS Bridge   │    │  ZED Mini       │
                       │  (Legacy)        │    │  Camera         │
                       └──────────────────┘    └─────────────────┘
```

## Project Structure

- `rl_agent/`: Reinforcement learning algorithms and models
  - `algorithms/p3o.py`: P3O implementation for direct drone control
  - `env/px4_base_env.py`: Environment implementation for drone control
  - `env/zed_integration.py`: ZED camera interface
  - `env/px4_comm/`: PX4 communication utilities
- `nodes/`: Standalone ROS2 nodes for vision processing and reward calculation
- `scripts/`: Utility scripts for training and deployment
- `msg/`: Custom ROS2 message definitions

## Safety Features

The platform includes comprehensive safety features:

- **Speed limits**: Configurable maximum velocities
- **Boundary enforcement**: Geographic flight area restrictions
- **Emergency landing**: Automatic safety responses
- **Collision avoidance**: Obstacle detection and avoidance
- **Action filtering**: Smooth control command processing

## Contributing

Contributions to DeepFlyer are welcome! Please feel free to submit pull requests or open issues to improve the framework.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use DeepFlyer in your research, please cite:

```
@misc{deepflyer2023,
  author = {DeepFlyer Team},
  title = {DeepFlyer: Direct Deep Reinforcement Learning for Autonomous UAV Control},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/your-username/DeepFlyer}}
}
```