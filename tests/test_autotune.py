import json
from pathlib import Path
from rl_agent.autotune import AutoTuneAssistant
import pytest

def test_autotune_no_data(tmp_path):
    log_path = tmp_path / 'metrics.jsonl'
    sugg_path = tmp_path / 'suggestions.json'
    at = AutoTuneAssistant(log_path, sugg_path, plateau_episodes=3)
    with pytest.raises(FileNotFoundError):
        at.run()

def test_autotune_plateau(tmp_path):
    # Create a metrics log with constant rewards
    log_path = tmp_path / 'metrics.jsonl'
    lines = []
    for _ in range(5):
        lines.append(json.dumps({'timestamp': 0, 'episode_reward': 1.0, 'episode_length': 10, 'timesteps': _}))
    log_path.write_text('\n'.join(lines))
    sugg_path = tmp_path / 'suggestions.json'
    at = AutoTuneAssistant(log_path, sugg_path, plateau_episodes=3)
    at.run()
    data = json.loads(sugg_path.read_text())
    assert 'adjust' in data or data.get('status') == 'no_change'
