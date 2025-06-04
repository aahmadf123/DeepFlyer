import json
import time
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class EpisodeLog:
    episode: int
    episode_reward: float
    episode_length: int
    timesteps: int

class JSONLinesLogger:
    """A lightweight logger that writes one JSON object per line and supports flushing."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate existing file
        self.log_path.write_text('')
        self._file = self.log_path.open('a')

    def log(self, data: Dict[str, Any]):
        """Append a JSON record with timestamp."""
        record = {"timestamp": time.time(), **data}
        self._file.write(json.dumps(record) + "\n")

    def flush(self):
        """Flush the underlying file buffer to disk."""
        self._file.flush()

    def close(self):
        """Close the log file."""
        self._file.close()
