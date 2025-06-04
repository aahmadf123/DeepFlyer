import json
from pathlib import Path
from typing import Dict, Any

class AutoTuneAssistant:
    """Auto-tune assistant that analyzes training metrics and suggests hyperparameter adjustments."""

    def __init__(self, log_path: Path, suggestions_path: Path, plateau_episodes: int = 50):
        self.log_path = Path(log_path)
        self.suggestions_path = Path(suggestions_path)
        self.plateau_episodes = plateau_episodes

    def run(self):
        """Analyze the last plateau_episodes and emit tuning suggestions."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Metrics log not found: {self.log_path}")

        # Load recent metrics
        lines = self.log_path.read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]

        # Example: detect if reward hasn't improved
        rewards = [rec.get('episode_reward', 0) for rec in records]
        if len(rewards) < self.plateau_episodes:
            return  # not enough data

        recent = rewards[-self.plateau_episodes:]
        if max(recent) - min(recent) < 1e-3:
            # plateau detected
            suggestions: Dict[str, Any] = {
                "adjust": "Consider increasing learning rate or entropy coefficient"
            }
        else:
            suggestions = {"status": "no_change"}

        self.suggestions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.suggestions_path, 'w') as f:
            json.dump(suggestions, f)
