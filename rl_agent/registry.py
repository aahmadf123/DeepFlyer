"""
Reward registry facade for tests and simple access patterns.

Provides a stable API over the internal registry in `rl_agent.rewards`.
"""

from typing import List, Dict, Callable

from .rewards import REGISTRY


class RewardRegistry:
    @staticmethod
    def list_presets() -> List[Dict[str, str]]:
        presets: List[Dict[str, str]] = []
        for key, meta in REGISTRY.items():
            presets.append({
                "id": key,
                "label": key.replace("_", " ").title(),
                "description": meta.get("description", "")
            })
        return presets

    @staticmethod
    def get_fn(preset_id: str) -> Callable:
        if preset_id not in REGISTRY:
            raise KeyError(preset_id)
        return REGISTRY[preset_id]["fn"]


