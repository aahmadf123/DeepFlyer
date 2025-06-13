# DeepFlyer

DeepFlyer is a deep reinforcement learning framework for autonomous drone control and navigation.

## Overview

DeepFlyer combines classical control methods with deep reinforcement learning to create robust and efficient flight controllers for UAVs. The framework leverages the P3O (Procrastinated Policy-based Observer) algorithm for tuning PID controllers in real-time, providing improved adaptability and performance over traditional fixed-gain controllers.

## Key Features

- P3O-based adaptive PID gain tuning
- Seamless integration with ROS2 and MAVROS
- ZED camera integration for visual perception
- Path planning and obstacle avoidance

## P3O Algorithm

The project implements the P3O algorithm, which combines the advantages of on-policy (PPO) and off-policy (SAC) learning methods. Key features of P3O include:

- **Procrastinated Updates**: Postpones on-policy updates to improve sample efficiency
- **Blended Learning**: Combines on-policy and off-policy gradients for better stability
- **Adaptive Exploration**: Uses entropy regularization to maintain appropriate exploration

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

### Training a P3O agent

```bash
cd ~/deepflyer_ws
source install/setup.bash
ros2 run deepflyer scripts/test_rl_supervisor.py --train --collect_time 300 --save_model ./models/p3o_agent.pt
```

### Testing a trained agent

```bash
cd ~/deepflyer_ws
source install/setup.bash
ros2 run deepflyer scripts/test_rl_supervisor.py --test --test_time 120 --load_model ./models/p3o_agent.pt
```

### Advanced parameters

The P3O implementation provides several hyperparameters to customize the learning behavior:

- `--procrastination_factor`: Controls how often on-policy updates occur (default: 0.95)
- `--alpha`: Blend factor between on-policy and off-policy learning (default: 0.2)
- `--entropy_coef`: Entropy regularization coefficient (default: 0.01)
- `--batch_size`: Batch size for learning updates (default: 256)
- `--learn_interval`: Time between learning updates in seconds (default: 1.0)

## Project Structure

- `rl_agent/`: Reinforcement learning algorithms and models
  - `env/`: Environment interfaces for MAVROS and ZED camera
  - `models/`: Neural network architectures
  - `supervisor_agent.py`: P3O implementation for PID tuning
- `api/`: REST API for remote monitoring and control
- `config/`: Configuration files for MAVROS and ZED camera
- `scripts/`: Utility scripts for training and deployment
- `tests/`: Unit and integration tests

## Contributing

Contributions to DeepFlyer are welcome! Please feel free to submit pull requests or open issues to improve the framework.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use DeepFlyer in your research, please cite:

```
@misc{deepflyer2023,
  author = {DeepFlyer Team},
  title = {DeepFlyer: Deep Reinforcement Learning for Autonomous UAV Control},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/your-username/DeepFlyer}}
}
```
