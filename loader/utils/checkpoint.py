import json
from pathlib import Path


class CheckpointManager:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> int:
        if not self._path.exists():
            return 0
        try:
            return int(json.loads(self._path.read_text()).get("last_committed_line", 0))
        except Exception:
            return 0

    def save(self, line: int) -> None:
        self._path.write_text(json.dumps({"last_committed_line": line}, indent=2))

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
