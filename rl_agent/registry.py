from typing import Callable, Dict, List

class RewardRegistry:
    _fns: Dict[str, Callable] = {}
    _meta: Dict[str, Dict[str, str]] = {}

    @classmethod
    def register(
        cls,
        id: str,
        fn: Callable,
        label: str,
        description: str
    ):
        """
        Register a reward function.

        Args:
            id: Unique preset identifier (matches user-facing preset_id).
            fn: The Python callable implementing the reward.
            label: Friendly name for UI display.
            description: Short description of the reward.
        """
        cls._fns[id] = fn
        cls._meta[id] = {"label": label, "description": description}

    @classmethod
    def list_presets(cls) -> List[Dict[str, str]]:
        """
        List all registered presets with id and label.
        """
        return [{"id": key, "label": cls._meta[key]["label"]} for key in cls._fns.keys()]

    @classmethod
    def get_fn(cls, id: str) -> Callable:
        """
        Retrieve the reward function by preset id.
        """
        return cls._fns[id]


# Default preset registrations
from rl_agent.rewards import (
    reach_target_reward,
    avoid_crashes_reward,
    save_energy_reward,
    fly_steady_reward,
    fly_smoothly_reward,
    be_fast_reward,
)

RewardRegistry.register(
    id="reach_target",
    fn=reach_target_reward,
    label="Reach the Target",
    description="Reward ∈ [0,1] as you approach the goal."
)
RewardRegistry.register(
    id="collision_avoidance",
    fn=avoid_crashes_reward,
    label="Avoid Crashes",
    description="Penalty for collisions or very close obstacles."
)
RewardRegistry.register(
    id="save_energy",
    fn=save_energy_reward,
    label="Save Energy",
    description="Encourage low throttle usage."
)
RewardRegistry.register(
    id="fly_steady",
    fn=fly_steady_reward,
    label="Fly Steady",
    description="Maintain altitude smoothly with minimal vertical speed."
)
RewardRegistry.register(
    id="fly_smoothly",
    fn=fly_smoothly_reward,
    label="Fly Smoothly",
    description="Penalize sudden changes in velocity and angular rates."
)
RewardRegistry.register(
    id="be_fast",
    fn=be_fast_reward,
    label="Be Fast",
    description="Reward forward speed and time-to-goal bonus."
)
