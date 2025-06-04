import numpy as np
import pytest
from rl_agent.rewards import (
    reach_target_reward,
    avoid_crashes_reward,
    save_energy_reward,
    fly_steady_reward,
    fly_smoothly_reward,
    be_fast_reward,
    path_efficiency_reward,
    adaptive_disturbance_reward,
    multi_objective_reward,
)

def test_reach_target_reward():
    state = {"position": [0,0,0], "goal": [1,0,0], "max_room_diagonal": np.sqrt(2)}
    r = reach_target_reward(state, {})
    expected = max(0.0, 1.0 - (1/np.sqrt(2)))
    assert pytest.approx(r, rel=1e-6) == expected

def test_avoid_crashes_reward():
    state = {"collision_flag": True, "dist_to_obstacle": 5}
    assert avoid_crashes_reward(state, {}) == -1.0
    state = {"collision_flag": False, "dist_to_obstacle": 0.1}
    assert avoid_crashes_reward(state, {}) == -0.5
    state = {"collision_flag": False, "dist_to_obstacle": 1.0}
    assert avoid_crashes_reward(state, {}) == 0.0

def test_save_energy_reward():
    assert save_energy_reward({}, {"throttle": 0.0}) == 1.0
    assert save_energy_reward({}, {"throttle": 1.0}) == 0.0

def test_fly_steady_reward():
    state = {"altitude": 1.0, "target_altitude": 2.0, "vertical_velocity": 0.5, "max_altitude_error": 1.0}
    r = fly_steady_reward(state, {})
    # altitude_component = 1 - 1/1 = 0, speed_penalty=0.25 => -0.25
    assert pytest.approx(r, rel=1e-6) == -0.25

def test_fly_smoothly_reward():
    state = {"curr_velocity": [1,0,0], "prev_velocity": [0,0,0], "curr_angular_velocity": 0.2, "prev_angular_velocity": 0.0, "dt": 1.0, "max_lin_jerk": 2.0, "max_ang_jerk": 0.5}
    r = fly_smoothly_reward(state, {})
    # lin_jerk=1, lin_penalty=0.5; ang_diff=0.2, ang_penalty=0.4; reward=1 - 0.5*0.5 - 0.5*0.4 = 1 -0.25 -0.2 =0.55
    assert pytest.approx(r, rel=1e-6) == 0.55

def test_be_fast_reward():
    state = {"curr_velocity": [1,0,0], "at_goal": False, "max_speed": 2.0}
    assert be_fast_reward(state, {}) == 0.5
    state = {"curr_velocity": [0,0,0], "at_goal": True, "time_elapsed": 5.0, "max_time_allowed": 10.0}
    r = be_fast_reward(state, {})
    assert pytest.approx(r, rel=1e-6) == 1.5

def test_path_efficiency_reward():
    # not at goal, prev_dist=5, curr_dist=3, straight_line=10
    state = {"distance_traveled": 8.0, "straight_line_dist": 10.0, "prev_to_goal_dist": 5.0, "curr_to_goal_dist": 3.0, "at_goal": False}
    r = path_efficiency_reward(state, {})
    assert pytest.approx(r, rel=1e-6) == (5-3)/10.0
    # at goal
    state["at_goal"] = True
    r2 = path_efficiency_reward(state, {})
    assert pytest.approx(r2, rel=1e-6) == 10.0/8.0

def test_adaptive_disturbance_reward():
    # no external force => reward = -0
    state = {"external_force": [0,0,0]}
    r = adaptive_disturbance_reward(state, {"thrust_vector": [0,0,0]})
    assert pytest.approx(r, abs=1e-6) == 0.0
    # with external force
    state = {"external_force": [1,0,0]}
    r = adaptive_disturbance_reward(state, {"thrust_vector": [1,0,0]})
    # comp_mag=1, disturbance_mag=1 => 1/1 -0.1*1 =0.9
    assert pytest.approx(r, rel=1e-6) == 0.9

def test_multi_objective_reward():
    state = {"curr_to_goal_dist": 3.0, "prev_to_goal_dist": 5.0, "straight_line_dist": 10.0}
    action = {"throttle": 0.0}
    weights = {"reach": 1.0, "collision": 1.0, "energy": 1.0, "speed": 1.0}
    # reach_target_reward approx = max(0,1 - (dist/diag)) diag default=1 => negative =>0
    # avoid_crashes=0, save_energy=1, speed delta=2/10=0.2 => total =0+0+1+0.2=1.2
    # Actually reach_target zero
    r = multi_objective_reward(state, action, weights)
    assert pytest.approx(r, rel=1e-6) == 1.2 