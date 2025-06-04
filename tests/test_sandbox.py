import pytest
from pathlib import Path
from rl_agent.sandbox import RewardSandbox

def write_temp_file(tmp_path: Path, content: str) -> Path:
    p = tmp_path / 'reward.py'
    p.write_text(content)
    return p

VALID_SCRIPT = '''
 def custom_reward(state, action) -> float:
     return 1.23
'''
INVALID_SCRIPT_NO_FN = '''
 def not_reward(state, action):
     return 0.0
'''
INVALID_SCRIPT_BAD_SIG = '''
 def custom_reward(state):
     return 0.0
'''


def test_sandbox_valid(tmp_path):
    p = write_temp_file(tmp_path, VALID_SCRIPT)
    sb = RewardSandbox(p)
    fn = sb.get_callable()
    ok, msg = sb.test_dummy()
    assert ok and msg == 'pass'


def test_sandbox_no_fn(tmp_path):
    p = write_temp_file(tmp_path, INVALID_SCRIPT_NO_FN)
    with pytest.raises(ValueError):
        RewardSandbox(p)


def test_sandbox_bad_sig(tmp_path):
    p = write_temp_file(tmp_path, INVALID_SCRIPT_BAD_SIG)
    with pytest.raises(ValueError):
        RewardSandbox(p)
