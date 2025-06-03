## 12-Week Summer Plan (May 14 – July 27, 2025)

Below is a concise, Markdown-friendly table showing each team member’s responsibilities over the 12-week period. Roles: **Uma (Simulation & CAD)**, **Jay (UI & Backend Integration)**, **Ahmad [Me] (RL & AI)**.

| Week | Dates        | Uma: Simulation & CAD                                                                                                                                                 | Jay: UI & Backend Integration                                                                                                                                                                                                                                                 | Ahmad: RL & AI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :--: | :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1  | May 14–20    | • Set up ROS 2 & Gazebo environment.<br>• Create bare-bones URDF visual model (no sensors).<br>• Verify drone spawns in Gazebo.                                       | • Scaffold FastAPI backend with placeholder endpoints.<br>• Initialize React/Next.js project skeleton.<br>• Define empty MongoDB schema.<br>• Create a stub “Training” page.                                                                                                  | • Install Python, PyTorch + CUDA, RL libraries.<br>• Clone simulation repo and verify ROS 2 topics.<br>• Start a Python package for the RL agent with a basic PPO skeleton.<br>• Implement a “hello-world” loop sending zero velocities and logging status.                                                                                                                                                                                                                                                                                     |
|   2  | May 21–27    | • Add IMU & collision tags to URDF.<br>• Attach front-facing camera plugin (publish to `/drone/camera/front/image_raw`).<br>• Optimize meshes and validate in Gazebo. | • Expose `/api/rewards/list` and `/api/train/start`, `/api/train/status` (dummy).<br>• Build Mission Selector UI stub: dropdown for reward presets, “Start Training” button.                                                                                                  | • Define six Explorer-mode reward function signatures: `reach_target`, `avoid_crashes`, `save_energy`, `fly_steady`, `fly_smoothly`, `be_fast`.<br>• Implement a `RewardRegistry` mapping IDs → functions.<br>• Expose `RewardRegistry.list_presets()` to FastAPI.                                                                                                                                                                                                                                                                              |
|   3  | May 28–Jun 3 | • Add motor plugins so drone can fly under velocity commands.<br>• Validate hover via Gazebo’s GUI.                                                                   | • Populate dropdown by calling `/api/rewards/list`.<br>• On “Start Training,” POST preset & defaults to `/api/train/start`.<br>• Show spinner awaiting `/api/train/status`.<br>• Set up WebSocket skeleton for future streaming.                                              | • Implement baseline PPO training loop:<br>  – Subscribe to `/drone/odom` and `/drone/camera/front/image_raw`.<br>  – Publish actions to `/drone/cmd_vel`.<br>  – Reward = –distance\_to\_goal.<br>  – Update every 64 steps.<br>  – Log metrics to CSV and JSON status file.                                                                                                                                                                                                                                                                   |
|   4  | Jun 4–10     | • Test simple “move to waypoint” scenario in Gazebo.<br>• Add downward-facing camera plugin for SLAM later.                                                           | • Connect `/api/train/start` to spawn background training job.<br>• Return `job_id` & set status to “started.”<br>• Implement `/api/train/status` by reading JSON status file.                                                                                                | • Refactor to support `path_efficiency` reward.<br>• Accept `preset_id` from Jay’s UI and use corresponding function.<br>• Validate that `path_efficiency_reward` changes logged rewards meaningfully.                                                                                                                                                                                                                                                                                                                                          |
|   5  | Jun 11–17    | • Integrate PX4/MAVROS so Gazebo simulates real autopilot behavior.<br>• Finalize Mission Selector UI readiness.                                                      | • Build “Simulation Viewer”: embed Gazebo camera feed via WebSocket or MJPEG.<br>• Add sliders for max velocity & acceleration, send to `/api/train/start`.                                                                                                                   | • Subscribe to downward camera, run simple SLAM (ORB-SLAM2 wrapper) to get altitude or 2D map.<br>• Implement `energy_efficiency_reward` penalizing throttle usage.<br>• Compare energy-efficiency vs. path-efficiency in small experiments.                                                                                                                                                                                                                                                                                                    |
|   6  | Jun 18–24    | • Debug MAVROS communication and data routing to ROS topics.<br>• Ensure collision detection works accurately.                                                        | • Integrate front camera stream into dashboard.<br>• Create a “Telemetry Overlay” showing episode number, last reward, `preset_id`.<br>• Route hyperparameter sliders to `/api/train-start`.                                                                                  | • Route training parameters (e.g. collision penalty weight) to ROS topics so simulation can use them.<br>• Implement & validate `collision_avoidance_reward` and `fly_smoothly_reward`.<br>• Run brief tests to confirm correct behavior.                                                                                                                                                                                                                                                                                                       |
|   7  | Jun 25–Jul 1 | • Validate full URDF + PX4 + sensor stack under new presets.<br>• Build Gazebo world 2 (Map 2: multi-path complex with dynamic obstacles).                            | • Add real-time charts: episode reward vs. episode, crash counts, variance.<br>• Enable side-by-side comparison of two training jobs’ reward curves.                                                                                                                          | • Build interactive hyperparameter UI component:<br>  – Sliders for `learning_rate`, `gamma`, `entropy_coef`.<br>  – On change, send new values to `/api/train/start`.<br>• Implement a simple “auto-tune assistant” that reviews reward plateau patterns and writes suggestions to a JSON file.                                                                                                                                                                                                                                                |
|   8  | Jul 2–8      | • Finish Map 2: three route options, dynamic barriers, wind zones.<br>• Test SLAM & collision avoidance in Map 2.                                                     | • Develop “Evaluation Dashboard”:<br>  – Graph: reward vs. episode with annotations.<br>  – Metrics: standard deviation of rewards, crashes per 10 episodes.<br>  – Flight-replay mini-map colored by speed or confusion.                                                     | • Integrate XAI overlays:<br>  – Compute Grad-CAM saliency from PPO’s CNN layers.<br>  – Overlay onto camera stream and publish via WebSocket.<br>  – Allow toggling “Show Saliency” in UI.                                                                                                                                                                                                                                                                                                                                                     |
|   9  | Jul 9–15     | • Build Map 3: multi-level complex (vertical levels, floating platforms, dynamic lighting).<br>• Validate SLAM & sensors on Map 3.                                    | • Create an “RL Glossary” hover component: definitions for “episode,” “reward,” “policy.”<br>• Add tooltips next to hyperparameter sliders (e.g. explain “learning rate” in plain language).<br>• Integrate saliency stream so users can toggle it on/off.                    | • Expand RL tutorial overlays:<br>  – Show “Agent is exploring” indicator when entropy high.<br>  – Refine AI Coach: analyze training logs (reward, loss, entropy) and trigger contextual tips (e.g. “lower learning rate” or “reduce collisions at Platform 2”).<br>  – Expose `/api/coach/suggestions` for Jay’s UI.                                                                                                                                                                                                                          |
|  10  | Jul 16–22    | • Draft Curriculum Mode: three scenarios in sequence (Maps 1 → 2 → 3).<br>• Configure Gazebo to load scenarios sequentially based on triggers.                        | • Build “Scenario Creator” UI:<br>  – Drag & drop obstacles onto blank grid.<br>  – Set waypoints and checkpoints.<br>  – Save JSON to `/api/scenario/upload`.<br>• Create “Curriculum” page where users pick scenario sequence.                                              | • Implement `ScenarioLoader`:<br>  – Read JSON from `/api/scenario/list`.<br>  – Spawn/Delete models via ROS 2 services.<br>  – Load chosen scenario at training start.<br>• Create `CurriculumRunner` that:<br>  1. Trains on Scenario 1 until reward threshold.<br>  2. Saves model, loads Scenario 2, resumes training.<br>  3. Repeats for Scenario 3.<br>• Test end-to-end with `curriculum=true` flag.                                                                                                                                    |
|  11  | Jul 23–29    | • Validate multi-objective rewards in Map 3 under Curriculum Mode.<br>• Add necessary sensors/triggers for multi-objective tasks.                                     | • Add UI controls for multi-objective weights:<br>  – Sliders for path\_efficiency, collision\_avoidance, energy\_saving, speed.<br>  – Send weight vector to `/api/train/start`.<br>• Build “Custom Reward Function” uploader: upload Python script, call Ahmad’s validator. | • Implement `multi_objective_reward(state, action, weights)`:<br>  – Weighted sum of `reach_target`, `avoid_crashes`, `save_energy`, and step-wise speed.<br>  – Normalize each component.<br>• Implement `dynamic_mission_reward` to handle mid-mission goal changes.<br>• Write a “reward validator” script that:<br>  1. Loads the user’s uploaded Python file and checks for signature `def custom_reward(state, action) -> float`.<br>  2. Runs a few dummy state/action tests.<br>  3. Returns pass/fail to Jay’s `/api/reward/validate`. |
|  12  | Jul 30–Aug 5 | • Finalize domain randomization in Gazebo (vary wind, lighting, sensor noise).<br>• Create “Sim-to-Real” checklist PDF.                                               | • Build “Sim-to-Real” tutorial page:<br>  – Embed checklist steps.<br>  – Provide “Download Model” button to fetch final ONNX file.<br>  – Offer “Dry-Run in Simulation” feature for final validation.                                                                        | • Complete domain randomization:<br>  – Randomize IMU noise, camera noise, minor force disturbances per episode.<br>  – Confirm models generalize across all three Gazebo worlds.<br>• Export final PPO model to ONNX.<br>• Write `sim_to_real_runner.py` that:<br>  1. Loads ONNX model.  2. Connects to real drone’s ROS 2 topics or staging simulator.  3. Streams telemetry and logs behavior differences.<br>• Deliver “Sim-to-Real” documentation for hardware team.                                                                      |

---

## Key Parameter Categories & Specific Parameters

Below are the **parameter categories** and **specific parameters** you’ll need to define for an indoor-only RL setup.

### 1. Environment & Simulation Parameters

*(Primarily configured by Uma in Gazebo; Jay exposes toggles/fields but does not handle the physics directly.)*

* **Map Dimensions & Boundaries**

  * *Floor plan size (X × Y)*: 10 m × 10 m
  * *Ceiling height*: 3 m
  * *Obstacle density*: 0.1 obs/m²
  * *Obstacle shapes & positions*: e.g. (box, cylinder), coordinates in meters

* **Lighting Conditions**

  * *Ambient light intensity range*: 200–800 lux
  * *Shadow variation*: boolean (on/off) or intensity parameter

* **Physics & Collision**

  * *Gravity*: 9.81 m/s²
  * *Air friction/damping*: drag coefficient 0.1–0.3
  * *Collision restitution (bounciness)*: 0.0 for hard collisions
  * *Floor friction coefficient*: 0.5

* **Wind & External Disturbances** (optional indoors)

  * *Wind gain*: 0 (no wind) or small gusts (0.1–0.3 m/s)
  * *Random force noise magnitude*: ±0.01 N per timestep

* **Sensor Noise / Domain Randomization**

  * *Camera Gaussian noise σ*: e.g. 5 intensity levels
  * *IMU noise floor*: accel σ=0.02 m/s², gyro σ=0.01 rad/s
  * *Depth-sensor noise*: ±0.05 m
  * *Randomization ranges*: e.g. IMU noise ∼ U(0.01, 0.03)

### 2. State / Observation Parameters

*(Ahmad’s RL code reads these; Uma publishes them; Jay may show them as “observation info” in the UI.)*

* **Sensor Frame Rates**

  * *Camera FPS*: 15 fps (front), 10 fps (downward)
  * *IMU update rate*: 100 Hz
  * *SLAM update rate*: 10 Hz

* **Camera Resolution & Field of View**

  * *Resolution (W × H)*: 640 × 480 pixels
  * *FOV horizontal*: 90°
  * *FOV vertical*: 60°

* **SLAM / Localization Outputs**

  * *Map resolution*: 0.05 m per grid cell
  * *Pose noise threshold*: 0.1 m
  * *Scan match tolerance*: 0.05 m

* **Observation Vector Contents**

  * Position (x, y, z) and orientation (quaternion)
  * Linear & angular velocities (IMU data)
  * Depth or point-cloud slice from ToF/LiDAR
  * Front camera image (downsampled or grayscale)
  * Collision flag or distance\_to\_obstacle
  * Battery or energy estimate (optional)

### 3. Action / Control Parameters

*(Ahmad’s RL loop writes these; Uma must accept them in Gazebo; Jay provides UI fields for end-users.)*

* **Velocity & Acceleration Limits**

  * *Max linear velocity (m/s)*: 1.5 m/s
  * *Max linear acceleration (m/s²)*: 2.0 m/s²
  * *Max angular velocity (rad/s)*: π/2 rad/s
  * *Max angular acceleration (rad/s²)*: π rad/s²

* **Control Mode**

  * *Velocity vs. attitude commands*: use `/cmd_vel` (linear & angular velocities)
  * *PID gains for velocity loop*: Kₚ=0.5, Kᵢ=0.1, K𝒹=0.05

* **Action Discretization vs. Continuous**

  * *Discrete example*:
        • 5 bins for linear x/y/z
        • 3 bins for yaw (left, none, right)
  * *Continuous example*:
        • Action vector ∈ ℝ⁴: \[vₓ, vᵧ, v\_z, ω\_yaw], with bounds as above

### 4. Reward Function Parameters

*(Ahmad’s RL code reads these; Jay shows sliders/inputs for user tuning; Uma ensures simulation publishes required state values.)*

* **Distance-to-Goal Component**

  * *Weight₁*: 1.0
  * *Goal tolerance radius*: 0.2 m

* **Collision Penalty**

  * *Collision\_penalty*: –10.0
  * *Near-obstacle penalty scale*: –(1/d²) if d < 0.5 m
  * *Near-miss threshold*: 0.2 m (penalty –1.0)

* **Smoothness / Jerk Penalty**

  * *Weight₂*: 0.3
  * *Max\_lin\_jerk*: 0.5 m/s³
  * *Max\_ang\_jerk*: 0.5 rad/s²

* **Energy / Motor Usage Penalty**

  * *Weight₃*: 0.2
  * *Throttle threshold*: 0.7 (above means penalty)

* **Time / Completion Bonus**

  * *Time\_penalty\_rate*: –0.01 per timestep
  * *Completion\_bonus*: +5

* **Multi-Objective Weights**

  * *w₁ (distance)*, *w₂ (collision)*, *w₃ (energy)*, *w₄ (speed)*: sliders in Researcher mode

### 5. RL Hyperparameters

*(Ahmad sets defaults; Jay provides sliders/fields for user tuning.)*

* **Learning Rate (α)**: 1e-5 → 1e-2 (default 3e-4)
* **Discount Factor (γ)**: 0.90 → 0.999 (default 0.99)
* **Batch Size**: {32, 64, 128, 256}
* **Clip (PPO ε)**: 0.1 → 0.3 (default 0.2)
* **Entropy Coefficient**: 0.0 → 0.1 (default 0.01)
* **Value Loss Coefficient (ϰ\_v)**: 0.5 → 1.0 (default 0.5)
* **Epochs per Update**: 3 → 10 (default 5)
* **Max Gradient Norm**: 0.1 → 1.0 (default 0.5)
* **Replay Buffer Size** (if off-policy): 10k → 100k (default 50k)

### 6. Training & Episode Parameters

*(Ahmad’s code uses these; Jay provides fields.)*

* **Max Episodes**: 1000
* **Max Steps/Episode**: 500 (∆t=0.05 s, total 25 s)
* **Warm‐Up (random actions)**: 1000 steps
* **Evaluation Frequency**: every 50 episodes (evaluate 5 episodes)
* **Checkpoint Frequency**: every 100 episodes

### 7. Curriculum Mode Parameters

*(Researcher only; UI fields generate a JSON list.)*

* **Scenario Sequence**: \["map\_simple", "map\_complex", "map\_multilevel"]
* **Episode thresholds to progress**: average reward ≥ 50 over last 20 episodes
* **Reward thresholds**: \[30, 40, 50] per scenario
* **Reset condition**: 3 crashes in a row

### 8. XAI & AI Coach Parameters

*(Researcher only; relevant to advanced users.)*

* **Saliency overlay threshold**: 0.6 activation
* **Coach plateau length**: 50 episodes with <5 reward improvement
* **Collision rate threshold**: >20% crashes over 20 episodes
* **Max suggestion frequency**: once every 20 episodes

---

## Preset Reward Functions

Below are the full definitions (in Python‐style pseudocode) of each preset reward, grouped by mode.

### Explorer Mode (6 Presets)

1. **`reach_target_reward`**

   ```python
   def reach_target_reward(state, action):
       # state["position"] = (x, y, z)
       # state["goal"]     = (gx, gy, gz)
       # state["max_room_diagonal"] = diagonal length (precomputed)
       dist = euclidean_distance(state["position"], state["goal"])
       # Reward ∈ [0, 1], with 1 at the goal, falls off linearly
       return max(0.0, 1.0 - (dist / state["max_room_diagonal"]))
   ```

2. **`avoid_crashes_reward`**

   ```python
   def avoid_crashes_reward(state, action):
       # state["collision_flag"]    = True/False
       # state["dist_to_obstacle"]  = nearest obstacle distance (m)
       if state["collision_flag"]:
           return -1.0  # heavy penalty for collision
       elif state["dist_to_obstacle"] < 0.2:
           return -0.5  # moderate penalty for very close
       else:
           return 0.0   # no penalty otherwise
   ```

3. **`save_energy_reward`**

   ```python
   def save_energy_reward(state, action):
       # action["throttle"] ∈ [0.0, 1.0]
       throttle = action["throttle"]
       # Reward is high if throttle is low; 1 (best) at 0, 0 (worst) at 1
       return 1.0 - throttle
   ```

4. **`fly_steady_reward`**

   ```python
   def fly_steady_reward(state, action):
       # state["altitude"]          = current z
       # state["target_altitude"]   = desired z
       # state["vertical_velocity"] = current vz
       # state["max_altitude_error"] = e.g. 1.0 m (normalization)
       alt_err = abs(state["altitude"] - state["target_altitude"])
       vz = abs(state["vertical_velocity"])
       # Baseline: 1.0 – (alt_error normalized), then penalize vertical speed
       altitude_component = max(0.0, 1.0 - (alt_err / state["max_altitude_error"]))
       speed_penalty = 0.5 * vz
       return altitude_component - speed_penalty
   ```

5. **`fly_smoothly_reward`**

   ```python
   def fly_smoothly_reward(state, action):
       # state["prev_velocity"] (vx_prev, vy_prev, vz_prev)
       # state["curr_velocity"] (vx, vy, vz)
       # state["prev_angular_velocity"] = ω_prev (scalar yaw rate)
       # state["curr_angular_velocity"] = ω (yaw rate)
       # state["dt"] = timestep duration (e.g. 0.05 s)
       # state["max_lin_jerk"] = e.g. 0.5 m/s³, state["max_ang_jerk"] = 0.5 rad/s²
       lin_jerk = euclidean_distance(state["curr_velocity"], state["prev_velocity"]) / state["dt"]
       ang_diff = abs(state["curr_angular_velocity"] - state["prev_angular_velocity"])
       lin_penalty = min(1.0, lin_jerk / state["max_lin_jerk"] )
       ang_penalty = min(1.0, ang_diff / state["max_ang_jerk"] )
       # Reward ∈ [0, 1], penalizing both linear and angular jerk equally
       return max(0.0, 1.0 - 0.5 * lin_penalty - 0.5 * ang_penalty)
   ```

6. **`be_fast_reward`**

   ```python
   def be_fast_reward(state, action):
       # state["time_elapsed"]       = seconds since episode start
       # state["max_time_allowed"]   = e.g. 30.0 s
       # state["curr_velocity"]      = (vx, vy, vz)
       # state["max_speed"]          = e.g. 1.5 m/s
       speed = euclidean_norm(state["curr_velocity"] )
       if state.get("at_goal", False):
           return 1.0 + (state["max_time_allowed"] - state["time_elapsed"]) / state["max_time_allowed"]
       else:
           # Provide a shaping reward based on forward speed toward goal
           return speed / state["max_speed"]
   ```

### Researcher Mode Additional Presets (3)

7. **`path_efficiency_reward`**

   ```python
   def path_efficiency_reward(state, action):
       # state["distance_traveled"]  = cumulative path length so far  
       # state["straight_line_dist"] = distance from start to goal  
       # state["prev_to_goal_dist"]  = previous step’s distance to goal  
       # state["curr_to_goal_dist"]  = current step’s distance to goal  
       if state.get("at_goal", False):
           eff = state["straight_line_dist"] / max(state["distance_traveled"], 1e-3)
           return eff   # ≥1 if path is longer than straight line; closer to 1 is better
       else:
           # Step-wise reward for moving closer to goal, normalized
           delta = state["prev_to_goal_dist"] - state["curr_to_goal_dist"]
           return delta / state["straight_line_dist"]
   ```

8. **`adaptive_disturbance_reward`**

   ```python
   def adaptive_disturbance_reward(state, action):
       # state["external_force"]    = (fx, fy, fz) on this timestep  
       # action["thrust_vector"]    = (tx, ty, tz) commanded thrust components  
       disturbance_mag = euclidean_norm(state["external_force"])
       comp_mag = projection_magnitude(action["thrust_vector"], state["external_force"])
       # Reward = how well the agent counters the disturbance, minus small penalty for magnitude
       return (comp_mag / (disturbance_mag + 1e-3)) - 0.1 * disturbance_mag
   ```

9. **`multi_objective_reward`**

   ```python
   def multi_objective_reward(state, action, weights):
       # weights = { "reach": w1, "collision": w2, "energy": w3, "speed": w4 }
       w1 = weights.get("reach", 1.0)
       w2 = weights.get("collision", 1.0)
       w3 = weights.get("energy", 1.0)
       w4 = weights.get("speed", 1.0)
       r_reach     = reach_target_reward(state, action)
       r_coll      = avoid_crashes_reward(state, action)
       r_energy    = save_energy_reward(state, action)
       # Step-wise speed component toward goal
       delta = state["prev_to_goal_dist"] - state["curr_to_goal_dist"]
       r_speed     = delta / state["straight_line_dist"] if state["straight_line_dist"] > 0 else 0
       # Weighted sum (ensure each sub-reward is ∈ [0,1] or [–1,1])
       return w1 * r_reach + w2 * r_coll + w3 * r_energy + w4 * r_speed
   ```

---

*This completes the combined 12-week plan, the key parameter definitions, and all preset reward functions in a Markdown-ready format.*
