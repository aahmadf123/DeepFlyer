# DeepFlyer

DeepFlyer is an educational platform for learning reinforcement learning with drones, inspired by AWS DeepRacer but extending into 3D aerial environments. The platform is designed to support two distinct user modes - Explorer (beginners) and Researcher (advanced) - to accommodate users across a wide range of technical abilities and ages.

## Overview

DeepFlyer provides a comprehensive platform for:
- Learning reinforcement learning concepts with drones
- Testing algorithms in both simulated and real-world environments
- Progressive learning from basic to advanced drone control
- Integration with ROS and MAVROS for UAV control

## Architecture

The platform consists of several key components:

- **RL Agent**: Reinforcement learning algorithms and models
- **Environment**: Gymnasium-compatible environments for training
- **ROS Integration**: Interface with Robot Operating System
- **MAVROS Support**: Control of PX4-based flight controllers
- **ZED Camera Integration**: Stereo vision for depth perception

## User Modes

### Explorer Mode (Ages 11-22)
- Simplified interface for beginners
- Basic flight controls and navigation tasks
- Restricted flight envelope for safety
- Focused on core RL concepts

### Researcher Mode
- Advanced interface for university students and researchers
- Full access to all system capabilities
- Custom algorithm implementation
- Extended sensor data and control options

## ROS / MAVROS Integration

DeepFlyer includes robust integration with ROS and MAVROS to control real drones:

- **ROS Environment**: Full Gymnasium-compatible environment
- **MAVROS Interface**: Control of PX4-based flight controllers
- **Mock Implementation**: Development without physical hardware
- **ZED Camera Support**: Stereo vision for depth perception and obstacle avoidance

### Supported Hardware

- **Flight Controller**: Pixhawk 6c with PX4 firmware
- **Drone Frame**: Holybro S500 or similar
- **Camera**: ZED Mini stereo camera
- **Companion Computer**: Raspberry Pi 4B
- **Communications**: Telemetry radio for monitoring

### Environment Classes

- `RosEnv`: Base environment with ROS integration
- `MAVROSEnv`: Environment specialized for MAVROS/PX4 control
- `MAVROSExplorerEnv`: Simplified environment for beginners
- `MAVROSResearcherEnv`: Full-featured environment for advanced users

## Getting Started

### Prerequisites

- Python 3.8+
- ROS2 (optional for simulation)
- MAVROS (optional for real drone control)
- ZED SDK (optional for ZED camera support)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deepflyer.git
cd deepflyer

# Install dependencies
pip install -e .
```

### Testing the Environment

```bash
# Test with mock implementation (no ROS required)
python scripts/test_mavros_env.py --mode explorer

# Test with real ROS/MAVROS (if installed)
python scripts/test_mavros_env.py --mode researcher

# Test all functions
python scripts/test_mavros_env.py --mode all
```

## Development

DeepFlyer is designed to be extensible, allowing for various reinforcement learning algorithms to be implemented and tested. The project follows a modular architecture where components can be replaced or extended.

### Adding New Algorithms

New reinforcement learning algorithms can be added in the `rl_agent/models` directory.

### Customizing Environments

Custom environments can be created by extending the base environments in `rl_agent/env`.

## License

[License information]
