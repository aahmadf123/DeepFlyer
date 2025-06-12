# DeepFlyer - RL as Supervisor for PID Control

This project implements a reinforcement learning (RL) agent that acts as a supervisor for a PID controller, adjusting PID gains to improve path following performance for a drone.

## Overview

The RL agent observes the drone's state (position, velocity, orientation) and path following errors (cross-track and heading errors), and outputs PID gain adjustments. The PID controller then uses these gains to compute velocity commands for the drone.

## Architecture

The project consists of the following components:

1. **PID Controller**: Computes velocity commands based on cross-track and heading errors.
2. **Error Calculator**: Computes cross-track and heading errors for path following.
3. **RL Supervisor**: Observes the state and errors, and outputs PID gain adjustments.
4. **Reward Function**: Computes rewards based on errors and control effort.
5. **ROS2 Node**: Integrates with PX4 for real-world deployment.
6. **Training Environment**: Simulates a drone following a path for training the RL agent.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepFlyer.git
cd DeepFlyer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the RL Supervisor

```bash
python scripts/train_supervisor.py --timesteps 100000 --with-disturbance
```

### Running the ROS2 Node

```bash
ros2 run deepflyer rl_pid_node
```

## Components

### PID Controller

The PID controller computes velocity commands based on cross-track and heading errors, using gains adjusted by the RL agent.

### Error Calculator

The error calculator computes cross-track and heading errors for path following, based on the drone's position, orientation, and the desired path.

### RL Supervisor

The RL supervisor observes the drone's state and path following errors, and outputs PID gain adjustments to improve performance.

### Reward Function

The reward function computes rewards based on cross-track error, heading error, and control effort, encouraging the RL agent to minimize errors while avoiding excessive control changes.

### ROS2 Node

The ROS2 node subscribes to position, velocity, and orientation topics from MAVROS, computes errors, and publishes velocity commands to control the drone.

### Training Environment

The training environment simulates a drone following a path, with optional wind disturbances, for training the RL agent.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
