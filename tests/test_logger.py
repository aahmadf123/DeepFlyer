import json
import time
from pathlib import Path
from rl_agent.logger import JSONLinesLogger, EpisodeLog

def test_jsonlines_logger(tmp_path):
    log_file = tmp_path / 'test.log'
    logger = JSONLinesLogger(log_file)
    # Log two entries
    logger.log({'episode_reward': 1.0, 'episode_length': 10, 'timesteps': 10})
    time.sleep(0.01)
    logger.log({'episode_reward': 2.0, 'episode_length': 20, 'timesteps': 20})
    logger.flush()
    content = log_file.read_text().splitlines()
    assert len(content) == 2
    rec0 = json.loads(content[0])
    rec1 = json.loads(content[1])
    assert 'timestamp' in rec0 and rec0['episode_reward'] == 1.0
    assert 'timestamp' in rec1 and rec1['episode_reward'] == 2.0
    logger.close()

def test_episode_log_dataclass():
    el = EpisodeLog(episode=5, episode_reward=3.5, episode_length=50, timesteps=500)
    assert el.episode == 5
    assert isinstance(repr(el), str)
