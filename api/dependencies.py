from pathlib import Path
from rl_agent.registry import RewardRegistry
from rl_agent.config import EXPLORER_ALGOS, RESEARCHER_ALGOS


def get_reward_registry():
    """Dependency to access the RewardRegistry singleton."""
    return RewardRegistry


def get_algorithms():
    """Dependency to list supported algorithms for Explorer and Researcher modes."""
    return {"explorer": EXPLORER_ALGOS, "researcher": RESEARCHER_ALGOS}


def get_status_dir():
    """Dependency to access the directory where status files are stored."""
    return Path('.')
