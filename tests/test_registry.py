import pytest
from rl_agent.registry import RewardRegistry


def test_list_presets_nonempty():
    presets = RewardRegistry.list_presets()
    assert isinstance(presets, list)
    assert len(presets) >= 6  # at least the six Explorer presets
    for p in presets:
        assert 'id' in p and 'label' in p


def test_get_fn_exists():
    presets = RewardRegistry.list_presets()
    # pick first preset id
    preset_id = presets[0]['id']
    fn = RewardRegistry.get_fn(preset_id)
    # should be callable
    assert callable(fn)


def test_get_fn_invalid():
    with pytest.raises(KeyError):
        RewardRegistry.get_fn('nonexistent_reward')
