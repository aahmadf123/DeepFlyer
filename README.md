# DeepFlyer - Direct RL Control for Drones

DeepFlyer is a deep reinforcement learning framework for autonomous drone control and navigation using direct control.

## Overview

DeepFlyer implements direct reinforcement learning control for drones using the P3O (Procrastinated Policy-based Observer) algorithm. Unlike traditional approaches that use RL to tune PID controllers, our approach directly outputs control commands to the drone, providing greater flexibility and performance.

## Key Features

- Direct RL control using P3O algorithm
- Seamless integration with ROS2 and MAVROS
- ZED camera integration for visual perception
- Path planning and obstacle avoidance
- Safety layer to prevent dangerous actions

## P3O Algorithm

The project implements the P3O algorithm, which combines the advantages of on-policy (PPO) and off-policy (SAC) learning methods. Key features of P3O include:

- **Procrastinated Updates**: Postpones on-policy updates to improve sample efficiency
- **Blended Learning**: Combines on-policy and off-policy gradients for better stability
- **Adaptive Exploration**: Uses entropy regularization to maintain appropriate exploration

## Direct Control Approach

Unlike our previous RL-as-Supervisor approach, the direct control approach:

1. Outputs control commands directly (thrust, roll rate, pitch rate, yaw rate)
2. Eliminates the need for intermediate PID controllers
3. Can discover control strategies not possible with PID control
4. Includes a safety layer to prevent dangerous actions

## Installation

### Prerequisites

- ROS2 (Rolling or Humble)
- Python 3.8 or later
- NVIDIA GPU recommended for training

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

## Usage

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

### Advanced Parameters

The P3O implementation provides several hyperparameters to customize the learning behavior:

- `--procrastination_factor`: Controls how often on-policy updates occur (default: 0.95)
- `--alpha`: Blend factor between on-policy and off-policy learning (default: 0.2)
- `--entropy_coef`: Entropy regularization coefficient (default: 0.01)
- `--batch_size`: Batch size for learning updates (default: 256)
- `--learn_interval`: Time between learning updates in seconds (default: 1.0)
- `--safety_layer`: Enable safety constraints (default: enabled)
- `--no_safety_layer`: Disable safety constraints

## Project Structure

- `rl_agent/`: Reinforcement learning algorithms and models
  - `direct_control_agent.py`: P3O implementation for direct drone control
  - `direct_control_network.py`: Neural network architecture for direct control
  - `direct_control_node.py`: ROS2 node for direct drone control
  - `models/`: Base model classes and utilities
- `scripts/`: Utility scripts for training and deployment
  - `test_direct_control.py`: Script for training and testing direct control
- `config/`: Configuration files for MAVROS and ZED camera

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