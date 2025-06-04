import gymnasium as gym
from typing import Callable, Any, Tuple

class CustomRewardWrapper(gym.Wrapper):
    """Gymnasium wrapper that replaces the environment reward using a user-defined function.

    The wrapper passes the original observation and action to the provided reward_fn and
    returns the resulting value as the reward.
    """

    def __init__(self, env: gym.Env, reward_fn: Callable[[dict, dict], float]):
        super().__init__(env)
        self.reward_fn = reward_fn
        # Track previous observation for reward functions that depend on state transitions
        self._prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        self.last_obs = obs
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Build state dictionary expected by reward_fn; users can customize env->state mapping
        state = {
            "observation": obs,
            "prev_observation": self._prev_obs,
            "info": info,
        }
        self._prev_obs = obs
        self.last_obs = obs

        # Convert action array to dictionary for reward_fn; for now simple mapping
        action_dict = {"raw": action}

        custom_reward = float(self.reward_fn(state, action_dict))
        return obs, custom_reward, terminated, truncated, info

class NormalizeObservation(gym.ObservationWrapper):
    """Normalize Box observations to [0,1]."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("NormalizeObservation only works with Box spaces.")
        self.low = obs_space.low
        self.high = obs_space.high

    def observation(self, obs):
        return (obs - self.low) / (self.high - self.low + 1e-8)

class ActionRescale(gym.ActionWrapper):
    """Rescale actions from [-1,1] to env's action space."""
    def __init__(self, env: gym.Env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        act_space = env.action_space
        if not isinstance(act_space, gym.spaces.Box):
            raise ValueError("ActionRescale only works with Box spaces.")
        self.source_low = low
        self.source_high = high
        self.target_low = act_space.low
        self.target_high = act_space.high

    def action(self, action):
        # scale action from source range to target range
        return self.target_low + (action - self.source_low) * (self.target_high - self.target_low) / (self.source_high - self.source_low)
