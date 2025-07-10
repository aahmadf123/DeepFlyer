## 12-Week Summer Plan (May 14 – July 27, 2025)

Below is a concise, Markdown-friendly table showing each team member's responsibilities over the 12-week period. Roles: **Uma (Simulation & CAD)**, **Jay (UI & Backend Integration)**, **Ahmad [Me] (RL & AI)**.

| Week | Dates        | Uma: Simulation & CAD                                                                                                                                                 | Jay: UI & Backend Integration                                                                                                                                                                                                                                                 | Ahmad: RL & AI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :--: | :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1  | May 14–20    | • Set up ROS 2 & Gazebo environment.<br>• Create bare-bones URDF visual model (no sensors).<br>• Verify drone spawns in Gazebo.                                       | • Scaffold FastAPI backend with placeholder endpoints.<br>• Initialize React/Next.js project skeleton.<br>• Define empty MongoDB schema.<br>• Create a stub "Training" page.                                                                                                  | • Install Python, PyTorch + CUDA, RL libraries.<br>• Clone simulation repo and verify ROS 2 topics.<br>• Start a Python package for the RL agent with a basic P3O skeleton.<br>• Implement a "hello-world" loop sending zero velocities and logging status.                                                                                                                                                                                                                                                                                     |
|   2  | May 21–27    | • Add IMU & collision tags to URDF.<br>• Attach front-facing camera plugin (publish to `/drone/camera/front/image_raw`).<br>• Optimize meshes and validate in Gazebo. | • Expose `/api/rewards/list` and `/api/train/start`, `/api/train/status` (dummy).<br>• Build Mission Selector UI stub: dropdown for reward presets, "Start Training" button. | • Define simplified two-term reward approach: `follow_trajectory` (cross-track error) and `heading_error`.<br>• Implement a `RewardRegistry` mapping IDs → functions, including metadata (friendly names, descriptions).<br>• Begin designing the custom-reward sandbox & validation framework (signature checks, safe execution).<br>• Expose `RewardRegistry.list_presets()` to FastAPI. |
|   3  | May 28–Jun 3 | • Add motor plugins so drone can fly under velocity commands.<br>• Validate hover via Gazebo's GUI.                                                                   | • Populate dropdown by calling `/api/rewards/list`.<br>• On "Start Training," POST preset & defaults to `/api/train/start`.<br>• Show spinner awaiting `/api/train/status`.<br>• Set up WebSocket skeleton for future streaming.                                              | • Implement baseline P3O training loop:<br>  – Subscribe to `/drone/odom` and `/drone/camera/front/image_raw`.<br>  – Publish actions to `/drone/cmd_vel`.<br>  – Reward = –distance_to_goal.<br>  – Update every 64 steps.<br>  – Refactor logging into a unified metrics module (standardized JSON schema for reward breakdown and training stats) and log metrics to CSV/JSON ready for live streaming. |
|   4  | Jun 4–10     | • Test simple "move to waypoint" scenario in Gazebo.<br>• Add downward-facing camera plugin for SLAM later.                                                           | • Connect `/api/train/start` to spawn background training job.<br>• Return `job_id` & set status to "started."<br>• Implement `/api/train/status` by reading JSON status file.                                                                                                | • Refactor to support `path_efficiency` reward.<br>• Accept `preset_id` from Jay's UI and use corresponding function.<br>• Validate that `path_efficiency_reward` changes logged rewards meaningfully.                                                                                                                                                                                                                                                                                                                                          |
|   5  | Jun 11–17    | • Integrate PX4/MAVROS so Gazebo simulates real autopilot behavior.<br>• Finalize Mission Selector UI readiness.                                                      | • Build "Simulation Viewer": embed Gazebo camera feed via WebSocket or MJPEG.<br>• Add code fields for max velocity & acceleration, send to `/api/train/start`.                                                                                                                   | • Subscribe to downward camera, run simple SLAM (ORB-SLAM2 wrapper) to get altitude or 2D map.<br>• Implement `energy_efficiency_reward` penalizing throttle usage.<br>• Compare energy-efficiency vs. path-efficiency in small experiments.                                                                                                                                                                                                                                                                                                    |
|   6  | Jun 18–24    | • Debug MAVROS communication and data routing to ROS topics.<br>• Ensure collision detection works accurately.                                                        | • Integrate front camera stream into dashboard.<br>• Create a "Telemetry Overlay" showing episode number, last reward, `preset_id`.<br>• Route hyperparameter code changes to `/api/train-start`.                                                                                  | • Route training parameters (e.g. collision penalty weight) to ROS topics so simulation can use them.<br>• Implement & validate `collision_avoidance_reward` and `fly_smoothly_reward`.<br>• Run brief tests to confirm correct behavior.                                                                                                                                                                                                                                                                                                       |
|   7  | Jun 25–Jul 1 | • Validate full URDF + PX4 + sensor stack under new presets.<br>• Build Gazebo world 2 (Map 2: multi-path complex with dynamic obstacles).                            | • Add real-time charts: episode reward vs. episode, crash counts, variance.<br>• Enable side-by-side comparison of two training jobs' reward curves.                                                                                                                          | • Build hyperparameter code editor interface:<br>  – Code fields for `learning_rate`, `gamma`, `entropy_coef`.<br>  – On change, send new values to `/api/train/start`.<br>• Implement the core auto-tune assistant: monitor reward plateaus and crash rates, prototype grid/Bayesian search over key hyperparameters, and output structured suggestions to JSON for Jay's UI.                                                                                                                                                                                                                                                |
|   8  | Jul 2–8       | • Finish Map 2: three route options, dynamic barriers, wind zones.<br>• Test SLAM & collision avoidance in Map 2.                                                     | • Develop "Evaluation Dashboard":<br>  – Graph: reward vs. episode with annotations.<br>  – Metrics: standard deviation of rewards, crashes per 10 episodes.<br>  – Flight-replay mini-map colored by speed or confusion.                                                     | • Integrate XAI overlays:<br>  – Compute Grad-CAM saliency from P3O's CNN layers.<br>  – Overlay onto camera stream and publish via WebSocket.<br>  – Allow toggling "Show Saliency" in UI.<br>• Instrument training loop to emit per-step and per-episode reward breakdown (distance, collision, energy components) in JSON for the frontend.                                                                                                                                                                                                                                                                                     |
|   9  | Jul 9–15     | • Build Map 3: multi-level complex (vertical levels, floating platforms, dynamic lighting).<br>• Validate SLAM & sensors on Map 3.                                    | • Create an "RL Glossary" hover component: definitions for "episode," "reward," "policy."<br>• Add tooltips next to hyperparameter code fields (e.g. explain "learning rate" in plain language).<br>• Integrate saliency stream so users can toggle it on/off.                    | • Expand RL tutorial overlays:<br>  – Show "Agent is exploring" indicator when entropy high.<br>  – Refine AI Coach: analyze training logs (reward, loss, entropy) and trigger contextual tips (e.g. "lower learning rate" or "reduce collisions at Platform 2").<br>  – Expose `/api/coach/suggestions` for Jay's UI.                                                                                                                                                                                                                          |
|  10  | Jul 16–22     | • Draft Curriculum Mode: three scenarios in sequence (Maps 1 → 2 → 3).<br>• Configure Gazebo to load scenarios sequentially based on triggers.                        | • Build "Scenario Creator" UI:<br>  – Code interface for obstacle placement with coordinates.<br>  – Set waypoints and checkpoints.<br>  – Save JSON to `/api/scenario/upload`.<br>• Create "Curriculum" page where users pick scenario sequence.                                              | • Implement `ScenarioLoader`:<br>  – Read JSON from `/api/scenario/list`.<br>  – Spawn/Delete models via ROS 2 services.<br>  – Load chosen scenario at training start.<br>• Create `CurriculumRunner` that:<br>  1. Trains on Scenario 1 until reward threshold.<br>  2. Saves model, loads Scenario 2, resumes training.<br>  3. Repeats for Scenario 3.<br>• Test end-to-end with `curriculum=true` flag.<br>• Harden the CurriculumRunner: add checkpointing, early-stop criteria, failure recovery, and model rollback between scenarios. |
|  11  | Jul 23–29     | • Validate multi-objective rewards in Map 3 under Curriculum Mode.<br>• Add necessary sensors/triggers for multi-objective tasks.                                     | • Add UI code editor for multi-objective weights:<br>  – Code fields for path_efficiency, collision_avoidance, energy_saving, speed.<br>  – Send weight vector to `/api/train/start`.<br>• Build "Custom Reward Function" uploader: upload Python script, call Ahmad's validator. | • Implement `multi_objective_reward(state, action, weights)`:<br>  – Weighted sum of `reach_target`, `avoid_crashes`, `save_energy`, and step-wise speed.<br>  – Normalize each component.<br>• Implement `dynamic_mission_reward` to handle mid-mission goal changes.<br>• Write a "reward validator" script that:<br>  1. Loads the user's uploaded Python file and checks for signature `def custom_reward(state, action) -> float`.<br>  2. Runs a few dummy state/action tests.<br>  3. Returns pass/fail to Jay's `/api/reward/validate`.<br>• Implement `adaptive_disturbance_reward` and an `intrinsic_motivation_reward` preset for novelty-based exploration. |
|  12  | Jul 30–Aug 5  | • Finalize domain randomization in Gazebo (vary wind, lighting, sensor noise).<br>• Create "Sim-to-Real" checklist PDF.                                               | • Build "Sim-to-Real" tutorial page:<br>  – Embed checklist steps.<br>  – Provide "Download Model" button to fetch final ONNX file.<br>  – Offer "Dry-Run in Simulation" feature for final validation.                                                                        | • Integrate domain-randomization callbacks into the RL training loop to sample new sensor and force noise levels each episode.<br>• Complete domain randomization:<br>  – Randomize IMU noise, camera noise, minor force disturbances per episode.<br>  – Confirm models generalize across all three Gazebo worlds.<br>  – Export final P3O model to ONNX and test loading in a lightweight runner.<br>• Write `sim_to_real_runner.py` that:<br>  1. Loads ONNX model.  2. Connects to real drone's ROS 2 topics or staging simulator.  3. Streams telemetry and logs behavior differences.<br>• Deliver "Sim-to-Real" documentation for hardware team.                                                                      |

---

## Key Parameter Categories & Specific Parameters

Below are the **parameter categories** and **specific parameters** you'll need to define for an indoor-only RL setup.

### 1. Environment & Simulation Parameters

*(Primarily configured by Uma in Gazebo; Jay exposes toggles/fields but does not handle the physics directly.)*

* **Map Dimensions & Boundaries**

  * *Floor plan size (X × Y)*: 10 m × 10 m
  * *Ceiling height*: 3 m
  * *Obstacle density*: 0.1 obs/m²
  * *Obstacle shapes & positions*: e.g. (box, cylinder), coordinates in meters

* **Lighting Conditions**

  * *Ambient light intensity range*: 200–800 lux
  * *Shadow variation*: boolean (on/off) or intensity parameter

* **Physics & Collision**

  * *Gravity*: 9.81 m/s²
  * *Air friction/damping*: drag coefficient 0.1–0.3
  * *Collision restitution (bounciness)*: 0.0 for hard collisions
  * *Floor friction coefficient*: 0.5

* **Wind & External Disturbances** (optional indoors)

  * *Wind gain*: 0 (no wind) or small gusts (0.1–0.3 m/s)
  * *Random force noise magnitude*: ±0.01 N per timestep

* **Sensor Noise / Domain Randomization**

  * *Camera Gaussian noise σ*: e.g. 5 intensity levels
  * *IMU noise floor*: accel σ=0.02 m/s², gyro σ=0.01 rad/s
  * *Depth-sensor noise*: ±0.05 m
  * *Randomization ranges*: e.g. IMU noise ∼ U(0.01, 0.03)

### 2. State / Observation Parameters

*(Ahmad's RL code reads these; Uma publishes them; Jay may show them as "observation info" in the UI.)*

* **Sensor Frame Rates**

  * *Camera FPS*: 15 fps (front), 10 fps (downward)
  * *IMU update rate*: 100 Hz
  * *SLAM update rate*: 10 Hz

* **Camera Resolution & Field of View**

  * *Resolution (W × H)*: 640 × 480 pixels
  * *FOV horizontal*: 90°
  * *FOV vertical*: 60°

* **SLAM / Localization Outputs**

  * *Map resolution*: 0.05 m per grid cell
  * *Pose noise threshold*: 0.1 m
  * *Scan match tolerance*: 0.05 m

* **Observation Vector Contents**

  * Position (x, y, z) and orientation (quaternion)
  * Linear & angular velocities (IMU data)
  * Depth or point-cloud slice from ToF/LiDAR
  * Front camera image (downsampled or grayscale)
  * Collision flag or distance_to_obstacle
  * Battery or energy estimate (optional)

### 3. Action / Control Parameters

*(Ahmad's RL loop writes these; Uma must accept them in Gazebo; Jay provides UI fields for end-users.)*

* **Velocity & Acceleration Limits**

  * *Max linear velocity (m/s)*: 1.5 m/s
  * *Max linear acceleration (m/s²)*: 2.0 m/s²
  * *Max angular velocity (rad/s)*: π/2 rad/s
  * *Max angular acceleration (rad/s²)*: π rad/s²

* **Control Mode**

  * *Velocity vs. attitude commands*: use `/cmd_vel` (linear & angular velocities)
  * *PID gains for velocity loop*: Kₚ=0.5, Kᵢ=0.1, K𝒹=0.05

* **Action Discretization vs. Continuous**

  * *Discrete example*:
        • 5 bins for linear x/y/z
        • 3 bins for yaw (left, none, right)
  * *Continuous example*:
        • Action vector ∈ ℝ⁴: \[vₓ, vᵧ, v_z, ω_yaw], with bounds as above

### 4. Reward Function Parameters

*(Ahmad's RL code reads these; Jay provides code editor for user tuning; Uma ensures simulation publishes required state values.)*

* **Path Following Component (Cross-Track Error)**

  * *cross_track_weight*: 1.0
  * *max_error*: 2.0 m

* **Heading Error Component**

  * *heading_weight*: 0.1
  * *max_heading_error*: π rad

* **Two-Term Reward Function**

  * Simple, educational approach combining path following and heading alignment
  * Normalized components for consistent scaling
  * Easy to understand and tune for beginners

### 5. RL Hyperparameters

*(Ahmad sets defaults; Jay provides code editor for user tuning.)*

* **Learning Rate (α)**: 1e-5 → 1e-2 (default 3e-4)
* **Discount Factor (γ)**: 0.90 → 0.999 (default 0.99)
* **Batch Size**: {32, 64, 128, 256}
* **Clip (P3O ε)**: 0.1 → 0.3 (default 0.2)
* **Entropy Coefficient**: 0.0 → 0.1 (default 0.01)
* **Value Loss Coefficient (ϰ_v)**: 0.5 → 1.0 (default 0.5)
* **Epochs per Update**: 3 → 10 (default 5)
* **Max Gradient Norm**: 0.1 → 1.0 (default 0.5)
* **Replay Buffer Size** (if off-policy): 10k → 100k (default 50k)

### 6. Training & Episode Parameters

*(Ahmad's code uses these; Jay provides code interface.)*

* **Max Episodes**: 1000
* **Max Steps/Episode**: 500 (∆t=0.05 s, total 25 s)
* **Warm‐Up (random actions)**: 1000 steps
* **Evaluation Frequency**: every 50 episodes (evaluate 5 episodes)
* **Checkpoint Frequency**: every 100 episodes

### 7. Curriculum Mode Parameters

*(Advanced features; code interface generates a JSON list.)*

* **Scenario Sequence**: \["map_simple", "map_complex", "map_multilevel"]
* **Episode thresholds to progress**: average reward ≥ 50 over last 20 episodes
* **Reward thresholds**: \[30, 40, 50] per scenario
* **Reset condition**: 3 crashes in a row

### 8. XAI & AI Coach Parameters

*(Advanced features; relevant to experienced users.)*

* **Saliency overlay threshold**: 0.6 activation
* **Coach plateau length**: 50 episodes with <5 reward improvement
* **Collision rate threshold**: >20% crashes over 20 episodes
* **Max suggestion frequency**: once every 20 episodes

---

## Preset Reward Functions

Below are the full definitions (in Python‐style pseudocode) of each preset reward, grouped by functionality.

### Core Reward Functions (6 Presets)

1. **`reach_target_reward`**

   ```python
   def reach_target_reward(state, action):
       # state["position"] = (x, y, z)
       # state["goal"]     = (gx, gy, gz)
       # state["max_room_diagonal"] = diagonal length (precomputed)
       dist = euclidean_distance(state["position"], state["goal"])
       # Reward ∈ [0, 1], with 1 at the goal, falls off linearly
       return max(0.0, 1.0 - (dist / state["max_room_diagonal"]))
   ```

2. **`avoid_crashes_reward`**

   ```python
   def avoid_crashes_reward(state, action):
       # state["collision"] = boolean
       # state["obstacle_distance"] = distance to nearest obstacle (m)
       if state["collision"]:
           return -1.0
       elif state["obstacle_distance"] < 0.5:
           # Penalty that grows as we approach obstacles
           penalty = (0.5 - state["obstacle_distance"]) / 0.5
           return -0.5 * penalty
       else:
           return 0.0
   ```

3. **`fly_smoothly_reward`**

   ```python
   def fly_smoothly_reward(state, action):
       # action = [vx, vy, vz, yaw_rate]
       # penalize large velocity commands, encourage gentle movements
       velocity_magnitude = np.linalg.norm(action[:3])
       yaw_rate_magnitude = abs(action[3])
       
       # Reward smooth, controlled flight
       if velocity_magnitude > 1.0:
           return -0.2 * (velocity_magnitude - 1.0)
       elif yaw_rate_magnitude > 0.5:
           return -0.1 * (yaw_rate_magnitude - 0.5)
       else:
           return 0.1  # Small positive reward for smooth flight
   ```

4. **`energy_efficiency_reward`**

   ```python
   def energy_efficiency_reward(state, action):
       # action = [vx, vy, vz, yaw_rate]
       # Approximate energy as sum of squared velocity commands
       energy_usage = np.sum(np.array(action[:3]) ** 2)
       max_energy = 3.0  # Assume max velocity of 1 m/s per axis
       
       # Reward efficient flight
       normalized_energy = energy_usage / max_energy
       return max(0.0, 1.0 - normalized_energy)
   ```

5. **`path_efficiency_reward`**

   ```python
   def path_efficiency_reward(state, action):
       # state["trajectory_index"] = current waypoint index
       # state["trajectory"] = list of waypoints
       # state["position"] = current position
       
       current_waypoint = state["trajectory"][state["trajectory_index"]]
       dist_to_waypoint = euclidean_distance(state["position"], current_waypoint)
       
       if dist_to_waypoint < 0.3:  # Close to waypoint
           return 1.0  # Big reward for reaching waypoint
       else:
           # Reward getting closer to the waypoint
           max_dist = 5.0  # Assume max room diagonal
           return max(0.0, 1.0 - (dist_to_waypoint / max_dist))
   ```

6. **`hover_stability_reward`**

   ```python
   def hover_stability_reward(state, action):
       # state["linear_velocity"] = [vx, vy, vz]
       # state["angular_velocity"] = [wx, wy, wz]
       
       linear_velocity = np.array(state["linear_velocity"])
       angular_velocity = np.array(state["angular_velocity"])
       
       # Reward staying still (hovering)
       linear_stability = 1.0 - min(1.0, np.linalg.norm(linear_velocity) / 0.5)
       angular_stability = 1.0 - min(1.0, np.linalg.norm(angular_velocity) / 0.5)
       
       return 0.5 * (linear_stability + angular_stability)
   ```

### Advanced Reward Functions (3 Additional)

1. **`dynamic_obstacle_avoidance_reward`**

   ```python
   def dynamic_obstacle_avoidance_reward(state, action):
       # state["obstacles"] = list of obstacle positions and velocities
       # More sophisticated avoidance that predicts obstacle movement
       
       future_position = np.array(state["position"]) + np.array(action[:3]) * 0.1
       
       total_penalty = 0.0
       for obstacle in state["obstacles"]:
           obstacle_future_pos = obstacle["position"] + obstacle["velocity"] * 0.1
           distance = np.linalg.norm(future_position - obstacle_future_pos)
           
           if distance < 1.0:
               penalty = (1.0 - distance) / 1.0
               total_penalty += penalty
       
       return max(-1.0, -total_penalty)
   ```

2. **`formation_flying_reward`**

   ```python
   def formation_flying_reward(state, action):
       # state["other_drones"] = positions of other drones in formation
       # state["formation_target"] = desired relative position to leader
       
       if len(state["other_drones"]) == 0:
           return 0.0  # Single drone, no formation
       
       leader_position = state["other_drones"][0]  # First drone is leader
       desired_position = leader_position + state["formation_target"]
       
       distance_error = np.linalg.norm(
           np.array(state["position"]) - desired_position
       )
       
       return max(0.0, 1.0 - (distance_error / 2.0))
   ```

3. **`adaptive_mission_reward`**

   ```python
   def adaptive_mission_reward(state, action):
       # state["mission_phase"] = "takeoff", "cruise", "landing", etc.
       # Different reward focus depending on mission phase
       
       if state["mission_phase"] == "takeoff":
           # Focus on altitude gain and stability
           altitude_reward = min(1.0, state["position"][2] / 1.5)
           stability_reward = hover_stability_reward(state, action)
           return 0.7 * altitude_reward + 0.3 * stability_reward
           
       elif state["mission_phase"] == "cruise":
           # Focus on path following and efficiency
           path_reward = path_efficiency_reward(state, action)
           energy_reward = energy_efficiency_reward(state, action)
           return 0.8 * path_reward + 0.2 * energy_reward
           
       elif state["mission_phase"] == "landing":
           # Focus on gentle descent and precision
           precision_reward = reach_target_reward(state, action)
           smooth_reward = fly_smoothly_reward(state, action)
           return 0.6 * precision_reward + 0.4 * smooth_reward
           
       else:
           return 0.0
   ```

---

## User Interface Reward Function Editor

*(This describes what students will see in the code editor interface.)*

Like AWS DeepRacer, students directly edit Python code to shape the drone's behavior. The interface provides:

### AWS DeepRacer-Style Code Editor

Students directly edit Python code in a web-based code editor, just like AWS DeepRacer:

**Code Editor Interface:**
- Syntax highlighting for Python
- Code completion and error detection
- Built-in examples and templates
- Save/load functionality for different strategies

**What Students Edit:**
Students modify the `reward_function` directly:

```python
def reward_function(params):
    """
    DeepFlyer Hoop Navigation Reward Function
    
    Input Parameters:
    - hoop_detected (bool): Is a hoop visible in camera?
    - hoop_distance (float): Distance to visible hoop in meters  
    - hoop_alignment (float): How centered hoop is (-1.0 to 1.0, 0=center)
    - approaching_hoop (bool): Is drone getting closer to target hoop?
    - hoop_passed (bool): Did drone just pass through a hoop?
    - collision (bool): Did drone hit something?
    - out_of_bounds (bool): Is drone outside safe flight area?
    """
    
    # Student-modifiable parameters
    HOOP_APPROACH_REWARD = 10.0      # Getting closer to target hoop
    HOOP_PASSAGE_REWARD = 50.0       # Successfully passing through hoop
    VISUAL_ALIGNMENT_REWARD = 5.0    # Keeping hoop centered in camera view
    
    # Safety penalties (adjustable severity)
    COLLISION_PENALTY = -100.0       # Hitting hoop or obstacle
    OUT_OF_BOUNDS_PENALTY = -200.0   # Flying outside safe area
    
    total_reward = 0.0
    
    # Reward for approaching hoops
    if params.get('approaching_hoop', False):
        total_reward += HOOP_APPROACH_REWARD
    
    # Major rewards for successful passages
    if params.get('hoop_passed', False):
        total_reward += HOOP_PASSAGE_REWARD
    
    # Penalties for dangerous behavior
    if params.get('collision', False):
        total_reward += COLLISION_PENALTY
        
    if params.get('out_of_bounds', False):
        total_reward += OUT_OF_BOUNDS_PENALTY
    
    return float(total_reward)
```

**Educational Features:**
- Comprehensive parameter documentation
- Example strategies (speed-focused, precision-focused, balanced)
- Real-time syntax validation
- Helpful error messages and debugging tips
- Code templates for different approaches

---

## Hyperparameter Optimization (Random Search)

Our system implements **random search hyperparameter optimization** exactly like AWS DeepRacer:

### Student-Configurable Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `learning_rate` | 3e-4 | 1e-4 to 3e-3 | Step size for optimizer |
| `clip_ratio` | 0.2 | 0.1 to 0.3 | Controls P3O-style policy update clipping |
| `entropy_coef` | 0.01 | 1e-3 to 0.1 | Weight for entropy term to encourage exploration |
| `batch_size` | 64 | [64, 128, 256] | Minibatch size for updates |
| `rollout_steps` | 512 | [512, 1024, 2048] | Environment steps per update |
| `num_epochs` | 10 | 3 to 10 | Epochs per policy update |
| `gamma` | 0.99 | 0.9 to 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | 0.9 to 0.99 | GAE parameter for advantage estimation |

### Training Time Configuration (REQUIRED)

Students **MUST** specify training time in minutes via the UI (no default value):

```python
--train_time 45  # Student must enter value (1-180 minutes) - NO DEFAULT
```

Under the hood, it maps to total environment steps:
```python
max_steps = steps_per_second * 60 * train_time_minutes
```

**UI Behavior:**
- Training time field appears empty (no default)
- Students must enter a value before training can start
- Valid range: 1-180 minutes
- Form validation prevents submission without training time

### Random Search Implementation

**How It Works:**
1. **Sample Configuration**: Randomly sample hyperparameters from defined ranges
2. **Train Model**: Train P3O agent with sampled configuration
3. **Evaluate Performance**: Measure average reward over evaluation episodes
4. **Update Best**: Track best configuration found so far
5. **Generate Suggestions**: AI analyzes results and suggests improvements

**Implementation Details:**
- **Log Scale Sampling**: Learning rate and entropy coefficient use logarithmic sampling
- **Discrete Options**: Batch size and rollout steps choose from predefined options
- **Linear Sampling**: Clip ratio, gamma, and GAE lambda use linear sampling
- **Performance Tracking**: All trials logged to ClearML for visualization

**Student Experience:**
1. Click "Start Hyperparameter Optimization"
2. System runs 20 random trials (configurable)
3. Live dashboard shows trial results
4. AI provides suggestions: "Try higher learning rates for faster learning"
5. Students can apply best configuration or modify based on suggestions

### ClearML Integration for Live Training

**Real-Time Metrics (Updated every 2-3 seconds):**
- Episode reward progression
- Policy loss, value loss, entropy
- Hoop completion rate
- Collision rate
- Training time elapsed

**Hyperparameter Optimization Dashboard:**
- Trial-by-trial results table
- Performance vs. hyperparameter visualizations  
- Best configuration tracking
- AI-generated optimization suggestions

**Data Structure in ClearML:**
```python
# Scalar metrics automatically logged
"Episode Reward": episode_reward_values
"Policy Loss": policy_loss_values  
"Value Loss": value_loss_values
"Hoop Completion Rate": success_rate_values

# Hyperparameter trials
"Trial Performance": trial_performance_values
"Best Configuration": best_hyperparameters

# Reward components breakdown
"Reward Components": {
    "hoop_approach": approach_rewards,
    "hoop_passage": passage_rewards,
    "collision_penalty": collision_penalties
}
```

**Integration for Jay (Backend Developer):**
```python
# Get live training metrics every 2-3 seconds
from api.ml_interface import DeepFlyerMLInterface

ml = DeepFlyerMLInterface()
metrics = ml.get_live_training_metrics()
# Returns: current_episode, total_reward, policy_loss, etc.

# Start training with student configuration  
ml.start_training(
    training_minutes=60,
    hyperparameters={'learning_rate': 1e-3, 'clip_ratio': 0.2}
)

# Get hyperparameter optimization results
trials = ml.get_optimization_trials()
best_config = ml.get_best_hyperparameters()
suggestions = ml.get_optimization_suggestions()
```

This implementation provides the exact AWS DeepRacer experience but for drone RL with our P3O algorithm.
