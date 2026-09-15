"""Reader for the browser client's published frame (`bridge/game-feed.ts`).

The writer renames a temp file into place, so whatever is at `path` is always a
whole JPEG. Everything else here is about refusing to lie: a missing, stale or
undecodable file returns None, and the HUD draws its placeholder. A frozen
frame presented as live is the failure this guards against.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

RUN_DIR = Path(os.environ.get("FLYBRAIN_RUN_DIR", "/tmp/malecns-osrs"))
DEFAULT_PATH = RUN_DIR / "game.jpg"
#: Older than this and the feed is dead, not slow: the writer runs at ~10 fps.
MAX_AGE_S = 3.0


class GameFeed:
    """Decodes the newest frame, and only when the file has actually changed."""

    def __init__(self, path: Path | str = DEFAULT_PATH, max_age_s: float = MAX_AGE_S) -> None:
        self.path = Path(path)
        self.max_age_s = max_age_s
        self._stamp: tuple[float, int] | None = None
        self._frame: np.ndarray | None = None

    def read(self) -> np.ndarray | None:
        """The current frame as BGR, or None if there isn't an honest one."""
        try:
            st = self.path.stat()
        except OSError:
            self._stamp, self._frame = None, None
            return None
        if time.time() - st.st_mtime > self.max_age_s:
            return None
        stamp = (st.st_mtime, st.st_size)
        if stamp != self._stamp:
            self._stamp = stamp
            self._frame = _decode(self.path)
        return self._frame


def _decode(path: Path) -> np.ndarray | None:
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


__all__ = ["DEFAULT_PATH", "MAX_AGE_S", "GameFeed"]
