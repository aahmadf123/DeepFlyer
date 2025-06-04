import pytest
import numpy as np
import gymnasium as gym
from rl_agent.env.wrappers import CustomRewardWrapper, NormalizeObservation, ActionRescale

class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.state = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    def reset(self, **kwargs):
        return self.state, {}
    def step(self, action):
        self.state = np.clip(self.state + action[:3] if len(action)>=3 else self.state, 0,1)
        return self.state, 0.0, False, False, {'info': True}


def test_custom_reward_wrapper():
    dummy = DummyEnv()
    # reward_fn returns constant 42
    wrapper = CustomRewardWrapper(dummy, lambda s, a: 42.0)
    obs, info = wrapper.reset()
    obs2, reward, term, trunc, info2 = wrapper.step(np.array([0.1, 0.1], dtype=np.float32))
    assert reward == 42.0


def test_normalize_observation():
    env = DummyEnv()
    wrapper = NormalizeObservation(env)
    obs, _ = wrapper.reset()
    # obs normalized to [0,1]
    norm_obs = wrapper.observation(np.array([0.5,0.5,0.5], dtype=np.float32))
    assert np.all(norm_obs >= 0.0) and np.all(norm_obs <= 1.0)


def test_action_rescale():
    env = DummyEnv()
    wrapper = ActionRescale(env, low=-1.0, high=1.0)
    # test action at extremes
    a_low = np.array([-1.0, -1.0], dtype=np.float32)
    a_high = np.array([1.0, 1.0], dtype=np.float32)
    rescaled_low = wrapper.action(a_low)
    rescaled_high = wrapper.action(a_high)
    # should map to env.action_space bounds
    assert np.allclose(rescaled_low, env.action_space.low)
    assert np.allclose(rescaled_high, env.action_space.high) 