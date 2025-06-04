import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Tuple

REQUIRED_FUNCTION_NAME = "custom_reward"

class RewardSandbox:
    """Loads and validates user-supplied reward functions in an isolated module."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)
        self.module: ModuleType = self._load_module()
        self.reward_fn = getattr(self.module, REQUIRED_FUNCTION_NAME, None)
        self._validate_signature()

    def _load_module(self) -> ModuleType:
        spec = importlib.util.spec_from_file_location("user_reward", self.file_path)
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader  # type: ignore
        assert loader is not None
        loader.exec_module(module)  # type: ignore
        return module

    def _validate_signature(self):
        if self.reward_fn is None or not callable(self.reward_fn):
            raise ValueError(f"File must define a callable `{REQUIRED_FUNCTION_NAME}(state, action)`")
        sig = inspect.signature(self.reward_fn)
        params = list(sig.parameters.values())
        if len(params) != 2:
            raise ValueError("custom_reward must accept exactly two positional arguments (state, action)")

    def get_callable(self):
        """Return the validated reward function."""
        return self.reward_fn

    def test_dummy(self) -> Tuple[bool, str]:
        """Run a quick evaluation on dummy inputs to verify it returns a float."""
        try:
            val = self.reward_fn({}, {})
            if not isinstance(val, (float, int)):
                return False, "custom_reward should return a float value"
        except Exception as e:
            return False, str(e)
        return True, "pass"
