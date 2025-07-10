# DeepFlyer - Educational Drone Reinforcement Learning Platform

DeepFlyer is a production-ready educational drone reinforcement learning platform that teaches autonomous navigation through direct RL control using the P3O (Procrastinated Policy Optimization) algorithm.

## Project Navigation

Start here:

### For Team Members
- **[Jay (Backend/UI Developer)](api/JAY_INTEGRATION_GUIDE.md)** - Complete backend integration documentation
- **[Uma (Simulation/ROS Developer)](UMA_INTEGRATION_GUIDE.md)** - Simulation and ROS integration specifications
- **[Technical Overview](TEAM_OVERVIEW.md)** - Detailed technical reference for all ML/RL implementation

### For Project Understanding
- **[Integration Overview](INTEGRATION_GUIDE.md)** - High-level system architecture and integration strategy
- **[Detailed Design Docs](docs/)** - In-depth technical design documentation

## Overview

DeepFlyer implements **direct reinforcement learning control** for drones using **PX4-ROS-COM** as the primary communication protocol. Unlike traditional approaches that use RL to tune PID controllers, our approach directly outputs control commands to the drone, providing greater flexibility and performance.

### MVP Flight Trajectory
The Minimum Viable Product demonstrates:
1. **Takeoff** from Point A to 0.8m altitude
2. **360° scan** to detect hoops using ZED Mini + YOLO11
3. **Navigate** toward single detected hoop
4. **Fly through** hoop with precision alignment
5. **Return** through same hoop from other side
6. **Land** at original Point A

## Key Features

- **Direct RL Control**: P3O algorithm outputs control commands directly (thrust, roll rate, pitch rate, yaw rate)
- **PX4-ROS-COM Integration**: Lower latency communication with PX4 flight controllers
- **ZED Camera Integration**: Real-time visual perception for navigation
- **Educational Focus**: Intuitive interface for learning reinforcement learning concepts
- **Safety Layer**: Prevents dangerous actions while maintaining learning flexibility
- **Sim-to-Real**: Train in simulation, deploy on real hardware

## Implementation Status

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

- ROS2 (Humble)
- Python 3.8 or later
- NVIDIA GPU recommended for training
- PX4 flight controller (for hardware deployment)

### Setup

1. Create a ROS2 workspace and clone this repository:

```bash
mkdir -p ~/deepflyer_ws/src
cd ~/deepflyer_ws/src
git clone https://github.com/aahmadf123/DeepFlyer.git
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

### Primary: PX4-ROS-COM (What we are using)
- **Direct PX4 integration** via PX4-ROS-COM DDS protocol
- **Lower latency** and **higher performance** than MAVROS
- **Native ROS2 integration** with PX4 flight controller

### Legacy: MAVROS (Fallback)
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

## 📁 Project File Structure

### Root Directory
```
DeepFlyer/
├── README.md                    # Main project documentation (this file)
├── TEAM_OVERVIEW.md            # Technical reference for all teammates
├── INTEGRATION_GUIDE.md        # High-level system architecture
├── UMA_INTEGRATION_GUIDE.md    # ROS/Simulation integration
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Python project configuration
├── package.xml                # ROS2 package definition
├── CMakeLists.txt            # Build configuration
├── docker-compose.yml        # Docker setup for development
└── Dockerfile.ml             # ML training container
```

### Core Implementation Directories

#### `rl_agent/` - Reinforcement Learning Core
**What**: Complete P3O algorithm implementation and training infrastructure
**Who**: Primary responsibility

```
rl_agent/
├── config.py                  # P3O hyperparameters & course configuration
├── algorithms/
│   ├── p3o.py                # P3O algorithm implementation  
│   └── replay_buffer.py      # Experience replay for training
├── models/
│   └── base_model.py         # Neural network architectures
├── rewards/
│   └── rewards.py            # Student-tunable reward functions
├── env/                      # 🌍 Training environments (not for teammates)
├── direct_control_agent.py   # 🎮 Direct RL control agent
├── direct_control_node.py    # 📡 ROS2 node for direct control
├── px4_training_node.py      # 🚁 PX4 training integration
└── utils.py                  # Utility functions
```

#### 🌐 `api/` - Backend Integration
**What**: ML interface for backend integration with ClearML and databases
**Who**: (Backend/UI) - primary integration point

```
api/
├── JAY_INTEGRATION_GUIDE.md   # Complete backend integration guide
├── ml_interface.py            # Main ML API interface (Jay's entry point)
├── ros_bridge.py              # ROS-to-REST API bridge
└── neon_database_schema.sql   # Database schema for student data
```

#### `nodes/` - ROS2 System Nodes  
**What**: Production ROS2 nodes for system integration
**Who**: (ROS/Simulation) - these are what simulation must interface with

```
nodes/
├── vision_processor_node.py   # YOLO11 hoop detection + ZED Mini
├── rl_agent_node.py           # General RL training infrastructure
├── p3o_agent_node.py          # P3O algorithm + 8D→4D control
├── px4_interface_node.py      # PX4-ROS-COM + safety layer
├── reward_calculator_node.py  # Student-tunable reward computation
└── course_manager_node.py     # MVP trajectory coordination
```

#### 📨 `msg/` - ROS2 Message Definitions
**What**: Custom message types for system communication
**Who**: (ROS/Simulation) - these define interface contracts

```
msg/
├── DroneState.msg             # Complete drone state information
├── VisionFeatures.msg         # YOLO11 vision processing results
├── CourseState.msg            # Course navigation & progress
├── RLAction.msg               # 4D action commands [vx,vy,vz,yaw_rate]
└── RewardFeedback.msg         # Educational reward breakdowns
```

### Development & Testing

#### 🧪 `scripts/` - Testing & Integration
**What**: Essential testing scripts for system validation
**Who**: for testing their integrations

```
scripts/
├── test_integration.py        # Complete system integration test
├── test_direct_control.py     # P3O direct control testing
└── test_yolo11_vision.py      # Vision pipeline testing
```

#### 🔬 `tests/` - Unit Testing
**What**: Core component unit tests
**Who**: Development validation

```
tests/
├── test_rewards.py            # Reward function testing
├── test_env.py                # Environment testing
├── test_logger.py             # Logging system testing
└── test_registry.py           # Component registry testing
```

### Configuration & Deployment

#### `launch/` - ROS2 Launch Files
**What**: System startup configurations
**Who**: (ROS/Simulation) for system deployment

```
launch/
├── deepflyer_ml.launch.py     # ML training system launch
└── mvp_system.launch.py       # MVP demonstration launch
```

#### 📚 `docs/` - Technical Documentation  
**What**: Detailed technical design documents
**Who**: Reference material for all team members

```
docs/
├── DEEPFLYER_CONCEPT.md       # Project concept & motivation
├── PX4_RL_IMPLEMENTATION.md   # PX4 integration details
├── YOLO11_INTEGRATION_GUIDE.md # Vision system integration
└── APPROACH_EVOLUTION.md      # Technical approach evolution
```

#### `weights/` - Model Assets
**What**: Pre-trained model weights
**Who**: Used by vision processing and RL training

```
weights/
└── best.pt                    # Pre-trained YOLO11 hoop detection model
```

### Quick Navigation for Team Members

**(Backend/UI) - Start Here:**
- `api/JAY_INTEGRATION_GUIDE.md` - Your complete integration guide
- `api/ml_interface.py` - Main entry point for backend integration  
- `api/neon_database_schema.sql` - Database schema

**(ROS/Simulation) - Start Here:**  
- `UMA_INTEGRATION_GUIDE.md` - Your complete integration guide
- `msg/` - Message definitions your simulation must publish/subscribe
- `nodes/` - ROS2 nodes your simulation must interface with

**Technical Implementation Details:**
- `TEAM_OVERVIEW.md` - Complete technical reference
- `rl_agent/config.py` - All system parameters and configuration

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

