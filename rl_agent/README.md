# RL-as-Supervisor for Drone Path Following

This module implements a Reinforcement Learning (RL) approach for tuning PID controllers in drone path following tasks. Instead of directly controlling the drone, the RL agent acts as a supervisor that adjusts the PID gains in response to changing conditions like wind disturbances.

## Overview

### Architecture

The system consists of the following components:

1. **Error Calculator**: Computes cross-track and heading errors based on the drone's position and orientation relative to a defined path.

2. **PID Controller**: Uses the errors to compute velocity commands for the drone.

3. **RL Agent**: Observes the state and errors, and outputs PID gains to optimize path following performance.

4. **ROS2 Interface**: Manages communication with the PX4 flight controller through MAVROS.

### Workflow

1. The ROS2 node subscribes to position, velocity, and orientation topics from MAVROS.
2. It computes cross-track and heading errors from the current position and orientation.
3. The RL agent observes these states and errors, and outputs a new PID gain.
4. The PID controller uses this gain to compute velocity commands.
5. These commands are published to MAVROS to control the drone.

## Implementation Details

### Observations (State)

The RL agent observes:
- Drone position (x, y, z)
- Drone velocity (vx, vy, vz)
- Drone orientation (roll, pitch, yaw)
- Cross-track error
- Heading error

### Actions

The RL agent outputs:
- Proportional gain (P) for the PID controller
  - (Can be extended to include I and D gains)

### Reward Function

The reward is computed as:
- Negative of (cross-track error + β × heading error + α × control effort)
- This penalizes being off the path, misaligned heading, and excessive control changes.

## How to Use

### Prerequisites

- ROS2 installed
- MAVROS configured with PX4
- Python 3.8+ with dependencies (see requirements.txt)

### Running the System

1. **Start the RL PID Supervisor**:

```bash
python3 -m scripts.test_rl_supervisor --train --collect_time 60 --train_steps 1000 --save_model models/rl_pid_model.pkl
```

2. **Visualize Path Following**:

```bash
python3 -m scripts.visualize_path --origin 0 0 1.5 --target 10 0 1.5
```

3. **Simulate Wind Disturbances**:

```bash
python3 -m scripts.simulate_wind --direction 1.57 --speed 1.0 --gust-freq 0.1 --variability 0.3
```

### Testing with Wind Disturbances

To evaluate how well the RL-tuned PID performs under wind disturbances:

1. **Train without wind**:
```bash
python3 -m scripts.test_rl_supervisor --train --save_model models/normal_model.pkl
```

2. **Train with wind**:
```bash
python3 -m scripts.simulate_wind --speed 1.0 &
python3 -m scripts.test_rl_supervisor --train --save_model models/wind_model.pkl
```

3. **Test both models with wind**:
```bash
python3 -m scripts.simulate_wind --speed 1.5 &
python3 -m scripts.test_rl_supervisor --test --load_model models/normal_model.pkl
# Then try with the wind-trained model
python3 -m scripts.test_rl_supervisor --test --load_model models/wind_model.pkl
```

## Extending the System

### Adding I and D Gains

To extend the system to tune all three PID gains (P, I, D):

1. Modify the action space in `rl_pid_supervisor.py`:
```python
action_space = spaces.Box(
    low=np.array([0.0, 0.0, 0.0]),
    high=np.array([10.0, 1.0, 1.0]),
    shape=(3,),
    dtype=np.float32
)
```

2. Update the PID controller with all three gains:
```python
self.pid_controller.update_gains(gain[0], gain[1], gain[2])
```

### Training for Different Wind Conditions

Create different models for various wind conditions:

```bash
# Light wind
python3 -m scripts.simulate_wind --speed 0.5 &
python3 -m scripts.test_rl_supervisor --train --save_model models/light_wind.pkl

# Strong wind
python3 -m scripts.simulate_wind --speed 2.0 &
python3 -m scripts.test_rl_supervisor --train --save_model models/strong_wind.pkl

# Gusty wind
python3 -m scripts.simulate_wind --speed 1.0 --variability 0.8 &
python3 -m scripts.test_rl_supervisor --train --save_model models/gusty_wind.pkl
```

## Understanding the Implementation

### Key Files

- `rl_pid_supervisor.py`: Main ROS2 node implementing the RL-as-Supervisor approach
- `pid_controller.py`: PID controller for generating velocity commands
- `error_calculator.py`: Computes cross-track and heading errors
- `supervisor_agent.py`: RL agent that tunes PID gains
- `mavros_utils/mavros_interface.py`: Interface for communication with MAVROS

### Scripts

- `test_rl_supervisor.py`: Script to run the system for training and testing
- `visualize_path.py`: Visualization tool for drone path and errors
- `simulate_wind.py`: Tool to simulate wind disturbances

## Customization

### Path Definition

Set a new path by modifying the origin and target points:

```python
supervisor.set_path([0.0, 0.0, 1.5], [10.0, 5.0, 2.0])
```

### Reward Function

Adjust the reward weights in `rl_pid_supervisor.py`:

```python
# Current
reward = -(abs(self.cross_track_error) + 0.1 * abs(self.heading_error))

# Modified with control penalty
control_effort = abs(linear_vel) + abs(angular_vel)
reward = -(abs(self.cross_track_error) + 0.1 * abs(self.heading_error) + 0.05 * control_effort)
``` 