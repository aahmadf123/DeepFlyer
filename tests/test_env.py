import pytest
import gymnasium as gym
from rl_agent.env import make_env


def test_make_gym_env():
    env = make_env('CartPole-v1')
    assert isinstance(env, gym.Env)
    obs, info = env.reset()
    assert isinstance(obs, (int, float, list, tuple)) or hasattr(obs, '__array__')
    env.close()


def test_make_ros_env_fallback(monkeypatch):
    # Simulate ROS not available
    monkeypatch.setitem(__import__('sys').modules, 'rclpy', None)
    env = make_env('ros:Dummy')
    assert isinstance(env, gym.Env)
    obs, info = env.reset()
    assert hasattr(env, 'step')
    env.close() 