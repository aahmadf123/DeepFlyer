## Educational RL Drone Platform Overview

This overview describes the entire project at a high level—covering user modes, core features, Gazebo simulation designs, speed profiles, hyperparameter options, reward presets (narrative descriptions), and the implementation roadmap. It focuses on **what** will be done rather than **how** to code each piece.

---

### 1. Introduction

An interactive, web-based platform for teaching reinforcement learning (RL) through indoor drone simulations. Inspired by AWS DeepRacer, the platform guides users from basic concepts to advanced experimentation:

* **Explorer Mode** (ages 11–22): Provides simple, guided missions with preconfigured reward behaviors, intuitive controls, and step-by-step tutorials. Perfect for newcomers to see how trial and error, rewards, and parameters shape a drone’s learning.
* **Researcher Mode** (advanced): Includes all Explorer features plus:

  * Four extra reward behaviors for deeper experimentation.
  * The ability to upload custom reward logic in Python.
  * Full control over hyperparameters via sliders or text fields.
  * Multi-objective optimization tools to combine different goals.
  * Detailed analytics and visualization of training progress.

**Primary Objectives:**

1. **Fundamental RL Education:** Enable beginners to experiment with trial-and-error, observe the effect of reward shaping, and understand the role of hyperparameters.
2. **In-Depth Research:** Allow advanced users to customize observations, actions, reward functions, and algorithms, then analyze results rigorously.
3. **Indoor Sim‑to‑Real Preparation:** Offer a high-fidelity Gazebo + ROS 2 simulation (integrated with PX4/MAVROS), employ domain randomization, and prepare users for eventual transfer to a physical indoor drone.

---

### 2. Gazebo Map Designs

The platform includes three indoor Gazebo worlds of increasing complexity, each targeting different RL skills:

1. **Map 1: Simple Course**
   • A straight, wide corridor (10 m × 10 m floor, 3 m ceiling) with a few static obstacles (boxes or cylinders).
   • Teaches basic waypoint navigation and altitude holding.
   • Variations: gentle wind gusts, a slow-moving obstacle, and an altitude-hold subtask.

2. **Map 2: Multi‑Path Complex**
   • A 10 m × 10 m area divided into three route options: one narrow but short, one wide but long, and one medium with obstacles.
   • Introduces dynamic barriers (gates that open/close) and localized wind zones.
   • Encourages path planning, adaptive route selection, and improved collision avoidance under changing conditions.

3. **Map 3: Multi‑Level Challenging**
   • A 10 m × 10 m base footprint with three vertical levels connected by ramps and floating platforms. Some zones are “no-fly” below certain heights.
   • Features narrow passages, time-based gates (open for limited intervals), moving platforms, and dynamic lighting conditions.
   • Challenges the agent to navigate between levels, land on specific platforms, avoid moving obstacles, and manage altitude changes.

> *Note:* Although each map fits within a 10 m × 10 m × 3 m boundary, the complexity increases by adding vertical elements, dynamic obstacles, and environment variations.

---

### 3. Speed Profiles

Controlling flight speed helps balance exploration and stability. Two sets of profiles address different user needs:

* **Explorer Mode (3 Presets):**
  • **Slow & Safe:** Maximum speed 0.5 m/s, acceleration 1 m/s²—ideal for cautious beginners.
  • **Normal:** Maximum speed 1 m/s, acceleration 2 m/s²—a balanced default.
  • **Fast:** Maximum speed 1.5 m/s, acceleration 3 m/s²—for users seeking quicker progress at some risk.

* **Researcher Mode (Custom Sliders):**
  • Velocity slider from 0.2 m/s to 2.0 m/s.
  • Acceleration slider from 0.5 m/s² to 5.0 m/s².
  • Enables fine-grained experiments exploring trade-offs between speed, stability, and learning efficiency.

---

### 4. Hyperparameter Options

#### 4.1 Explorer Mode Hyperparameters (Predefined Choices)

* **Learning Rate:** Select among 0.0001, 0.0003, or 0.001.
* **Total Episodes:** Choose 100, 500, or 1000 episodes of training.
* **Batch Size:** Select 32, 64, or 128 transitions per training update.

These options simplify tuning for beginners by limiting the range to sensible defaults.

#### 4.2 Researcher Mode Hyperparameters (Full Control)

* **Learning Rate Slider:** Range from 0.00001 to 0.01, allowing fine adjustments.
* **Discount Factor (γ) Slider:** From 0.90 to 0.999, controlling how future rewards are valued.
* **Entropy Coefficient Slider:** From 0 to 0.1, adjusting exploration vs. exploitation.
* **Value Loss Coefficient Slider:** From 0.5 to 1.0, weighting the value-function loss term.
* **Batch Size Selector:** Any power-of-two between 16 and 512.
* **PPO Clip (ε) Slider:** From 0.1 to 0.3, controlling trust-region size.
* **Max Gradient Norm Slider:** From 0.1 to 1.0, setting gradient clipping thresholds.
* **Epochs per Update Choices:** 3, 5, 7, or 10 pass-throughs over each batch.
* **Replay Buffer Size Options:** If using off-policy methods, choose among 10 k, 25 k, 50 k, or 100 k samples.

> *Note:* Researchers can hide individual hyperparameter fields behind a single “difficulty” slider if desired, allowing novices to still benefit from tuned defaults.

---

### 5. Core Parameter Categories & Details

#### 5.1 Environment & Simulation Parameters

* **Map Dimensions & Obstacles:** All maps remain inside a 10 m × 10 m × 3 m volume. Static boxes or cylinders are placed manually; dynamic barriers and platforms move according to simple scripts.
* **Lighting & Shadows:** Indoor lighting levels vary between 200 and 800 lux; shadows can be toggled to simulate different real‐world room conditions.
* **Physics Settings:** Standard Earth gravity (9.81 m/s²), drag coefficient in the range 0.1 – 0.3 for aerodynamic damping, zero restitution on collisions (no bounce), and a floor friction coefficient of 0.5.
* **Disturbances & Randomization:** Optional wind gusts of 0.1 – 0.3 m/s can be enabled in localized zones. Random force noise up to ±0.01 N is applied per timestep to simulate subtle indoor air currents. Sensor noise (Gaussian) is added to camera images (σ = 5 intensity levels), IMU accelerometer readings (σ = 0.02 m/s²) and gyroscope readings (σ = 0.01 rad/s), and depth sensor measurements (±0.05 m) to improve robustness.

#### 5.2 State / Observation Parameters

* **Sensor Update Rates:** Front camera publishes at 15 fps; downward camera at 10 fps; IMU at 100 Hz; SLAM pose updates at 10 Hz.
* **Camera Specifications:** 640 × 480-pixel resolution with a 90° horizontal FOV and 60° vertical FOV.
* **SLAM Configuration:** Occupancy grid resolution of 0.05 m per cell, pose noise tolerance of 0.1 m, and scan-match tolerance of 0.05 m.
* **Observation Vector Contents:** Includes 3D position and orientation (quaternion), linear and angular velocities, a processed depth slice or point-cloud segment, a downsampled front-camera image, a collision flag or nearest-obstacle distance, and an optional battery/energy state estimate.

#### 5.3 Action / Control Parameters

* **Maximum Velocities & Accelerations:** Linear velocity capped at 1.5 m/s, linear acceleration at 2.0 m/s²; angular velocity capped at π/2 rad/s, angular acceleration at π rad/s².
* **Control Mode:** The agent sends velocity commands (`/cmd_vel`) combining linear and angular components. A PID controller (Kₚ = 0.5, Kᵢ = 0.1, K𝒹 = 0.05) ensures smooth tracking.
* **Action-Space Representation:** Explorer uses a small discrete set (e.g., five bins for each linear axis and three for yaw) to simplify decision-making. Researchers can work with a continuous 4D action vector \[vₓ, vᵧ, v\_z, ω\_yaw] bounded by the above limits.

#### 5.4 Reward Function Parameters

* **Distance-to-Goal Component:** Encourages moving toward a specified waypoint. The closer the drone gets, the higher the reward, with full reward at within 0.2 m of the goal.
* **Collision Penalty:** A heavy negative penalty (–10) for any collision. If within 0.5 m of an obstacle, a smaller penalty is applied, increasing as distance decreases; at under 0.2 m, a near-miss penalty (–1) is assigned.
* **Smoothness (Jerk) Penalty:** Penalizes rapid changes in acceleration (jerk). If linear jerk exceeds 0.5 m/s³ or angular jerk exceeds 0.5 rad/s², the reward is reduced proportionally, up to a 30 % penalty weight.
* **Energy (Throttle) Penalty:** Penalizes throttle usage when above 0.7 (70 % power), encouraging energy-efficient flight. A weight of 0.2 is used for this component.
* **Time/Completion Bonus:** A small time-based penalty (–0.01 per timestep) encourages faster completion. Upon reaching the goal, a completion bonus of +5 is awarded.
* **Multi-Objective Weights:** In Researcher Mode, users assign weights to each component—distance, collision avoidance, energy usage, and speed—so they can tune trade-offs dynamically.

#### 5.5 Training & Episode Parameters

* **Maximum Episodes:** 1000 episodes of training are allowed per run, giving ample opportunity for convergence.
* **Steps per Episode:** Each episode lasts up to 500 steps (0.05 s per step, totaling 25 s max). Episodes terminate early if the goal is reached or a crash occurs.
* **Warm-Up Steps:** The first 1000 steps are random actions (no learning) to fill any replay buffer and avoid cold starts.
* **Evaluation Frequency:** Every 50 training episodes, the agent runs 5 evaluation episodes with no exploration noise to track progress.
* **Checkpoint Frequency:** The model is saved every 100 episodes for later analysis or rollback.

#### 5.6 Curriculum Mode Parameters

* **Scenario Sequence:** Training proceeds through three sequential worlds—Map 1 (simple), then Map 2 (complex), then Map 3 (multi-level).
* **Episode Thresholds to Advance:** To progress from one map to the next, the average reward over the last 20 episodes must exceed 50.
* **Reward Thresholds per Map:** During Map 1, the threshold is 30; Map 2 requires 40; Map 3 requires 50.
* **Reset Conditions:** If the drone crashes three consecutive times, the current scenario resets to avoid wasted training time.

#### 5.7 XAI & AI Coach Parameters

* **Saliency Overlay Threshold:** Activations above 0.6 are visualized to show which camera pixels the network attends to.
* **Coach Plateau Length:** If there is <5 % improvement in reward over 50 episodes, the AI Coach provides tips (e.g., adjust learning rate or modify reward weights).
* **Collision Rate Threshold:** If crashes exceed 20 % of episodes over a 20-episode window, the Coach suggests stronger collision penalties or reduced speed.
* **Max Suggestion Frequency:** The Coach offers at most one suggestion every 20 episodes to avoid overwhelming users.

---

### 6. Gazebo Map Designs (Summary)

1. **Map 1: Simple Course**
   • Straight corridor with a few static boxes; target altitude hold at 1.5 m; optional wind gusts.
2. **Map 2: Multi-Path Complex**
   • Three possible routes (short/narrow, long/open, medium/obstacles), dynamic gates, localized wind zones.
3. **Map 3: Multi-Level Challenging**
   • Three stacked levels, moving platforms, timed gates, no-fly zones, variable lighting.

Each world provides the agent with progressively harder tasks, focusing on navigation, path planning, altitude control, and timed interactions.

---

### 7. Speed Profiles (Summary)

* **Explorer Mode:**
  • *Slow & Safe* (0.5 m/s, 1.0 m/s²)
  • *Normal* (1.0 m/s, 2.0 m/s²)
  • *Fast* (1.5 m/s, 3.0 m/s²)
* **Researcher Mode:**
  • *Custom Sliders* allow velocity from 0.2 m/s to 2.0 m/s and acceleration from 0.5 m/s² to 5.0 m/s² for precise experimentation.

---

### 8. Hyperparameter Options (Summary)

* **Explorer Mode:**
  • Learning Rate: {0.0001, 0.0003, 0.001}
  • Episodes: {100, 500, 1000}
  • Batch Size: {32, 64, 128}
* **Researcher Mode:**
  • Learning Rate slider (0.00001–0.01)
  • Discount (γ) slider (0.90–0.999)
  • Entropy Coefficient slider (0–0.1)
  • Value Loss Coefficient slider (0.5–1.0)
  • Batch Size selector (16–512)
  • PPO Clip (ε) slider (0.1–0.3)
  • Max Gradient Norm slider (0.1–1.0)
  • Epochs/Update choices {3, 5, 7, 10}
  • Replay Buffer size options {10 k, 25 k, 50 k, 100 k}

---

### 9. Implementation Roadmap (High-Level)

1. **Weeks 1–2 (Environment & Setup)**

   * Set up ROS 2 & Gazebo.
   * Create a basic URDF with camera, IMU, and collision tags.
   * Scaffold the FastAPI backend and React frontend stubs.
   * Define the first six reward behaviors in a registry.

2. **Weeks 3–4 (Basic Training Loop)**

   * Add motor and velocity command support, verify hover and basic navigation.
   * Connect the frontend “Start Training” button to backend endpoints.
   * Implement a baseline PPO training loop using distance-to-goal rewards.
   * Log training metrics to a file or database.

3. **Weeks 5–6 (Sensor & Reward Extensions)**

   * Integrate PX4/MAVROS for realistic drone flight in Gazebo.
   * Build a simple SLAM pipeline using the downward camera for altitude and positioning.
   * Add energy-efficiency and smoothness reward behaviors, test them in Map 1 and Map 2.
   * Validate collision avoidance in a dynamic obstacle scenario.

4. **Weeks 7–8 (Analytics & XAI)**

   * Develop an evaluation dashboard displaying reward vs. episode curves, crash statistics, and variance.
   * Implement Grad-CAM–based saliency maps to visualize what the agent “looks at” during flight.
   * Add hyperparameter sliders to the UI and an “auto-tune assistant” that suggests adjustments when progress plateaus.

5. **Weeks 9–10 (Curriculum & Scenario Creator)**

   * Complete Map 3 (multi-level) and add RL tutorial overlays explaining concepts in-context.
   * Build a drag-and-drop scenario creator allowing users to place obstacles and define waypoints.
   * Implement curriculum training: automatically progress from Map 1 to Map 3 based on reward thresholds.

6. **Week 11 (Advanced Rewards & Custom Uploader)**

   * Introduce three advanced reward behaviors (path efficiency, adaptive disturbance, multi-objective).
   * Add a UI component for uploading custom Python reward code and validating its format.
   * Enable dynamic weighting of multi-objective components in the UI.

7. **Week 12 (Domain Randomization & Sim‑to‑Real)**

   * Enable sensor noise, random wind, and lighting variations in all Gazebo maps.
   * Finalize model export to ONNX format.
   * Provide a “Sim‑to‑Real” script outline for testing on a physical drone or staging simulator.
   * Publish a concise “Sim‑to‑Real” checklist and tutorial for future hardware integration.

---

### 10. Reward Preset Descriptions (No Code)

#### 10.1 Explorer Mode Presets (6)

1. **Reach the Target:**
   Encourage the drone to move toward a specified waypoint. Rewards increase as the drone gets closer, with maximum reward when within 0.2 m of the goal.

2. **Avoid Crashes:**
   Penalize collisions heavily. If the drone comes within 0.5 m of an obstacle, it receives a small penalty that grows as the distance decreases. A near-miss under 0.2 m incurs a moderate penalty.

3. **Save Energy:**
   Encourage low throttle usage. The less throttle (motor power) the drone applies, the higher the reward—promoting energy-efficient flight for longer missions.

4. **Fly Steady:**
   Reward the drone for maintaining a constant target altitude. Deviations from the target altitude reduce the reward proportionally, and vertical velocity incurs an additional small penalty.

5. **Fly Smoothly:**
   Penalize sudden changes in acceleration (jerk) or rapid yaw rotations. The smoother the velocity and turning profiles, the higher the reward—ideal for inspection or cinematography tasks.

6. **Be Fast:**
   Reward rapid mission completion. Each timestep carries a small negative reward to incentivize speed. When the drone reaches the goal, it receives a completion bonus scaled by how quickly it arrived.

#### 10.2 Researcher Mode Additional Presets (3)

7. **Path Efficiency:**
   Compare the total distance traveled to the straight-line distance between start and goal. Agents that take near-straight paths receive higher rewards; winding or unnecessarily long routes yield lower scores.

8. **Adaptive Disturbance Compensation:**
   Simulate small indoor disturbances (e.g., subtle drafts). Reward the drone for applying thrust that counters these random forces. Effective disturbance compensation leads to higher reward, while letting wind push you off course incurs penalties.

9. **Multi-Objective Optimization:**
   Combine multiple reward components—such as reaching the target, avoiding collisions, saving energy, and maximizing speed—each weighted by a user-defined parameter. This preset allows advanced users to study trade-offs and find Pareto-optimal strategies.

---

*This updated overview now fully covers the platform’s scope—Gazebo map designs, speed profiles, hyperparameter options, reward presets (described narratively), and a high-level implementation roadmap—without any code blocks, ensuring readability and clarity for all stakeholders.*
