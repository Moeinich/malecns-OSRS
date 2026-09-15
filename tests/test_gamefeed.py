"""Game-feed reader tests. Hermetic: no browser, no game, only files on disk."""

import os
import time

import cv2
import numpy as np
import pytest

from flybrain.gamefeed import GameFeed


def jpeg(colour=(10, 120, 200), size=(48, 64)) -> bytes:
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = colour
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def publish(path, data: bytes) -> None:
    """What bridge/game-feed.ts does: write a temp file, then rename it in."""
    tmp = path.with_suffix(".jpg.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


@pytest.fixture
def feed_path(tmp_path):
    return tmp_path / "game.jpg"


def test_decodes_a_published_frame(feed_path):
    publish(feed_path, jpeg())
    frame = GameFeed(feed_path).read()
    assert frame is not None
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8


def test_missing_file_is_none(feed_path):
    assert GameFeed(feed_path).read() is None


def test_half_written_file_is_never_torn(feed_path):
    """The temp file exists and is partial; the published path does not yet."""
    data = jpeg()
    tmp = feed_path.with_suffix(".jpg.tmp")
    tmp.write_bytes(data[: len(data) // 2])
    feed = GameFeed(feed_path)
    assert feed.read() is None

    os.replace(tmp, feed_path)
    feed_path.write_bytes(data)
    assert feed.read() is not None


def test_stale_frame_is_none_not_frozen(feed_path):
    publish(feed_path, jpeg())
    feed = GameFeed(feed_path, max_age_s=2.0)
    assert feed.read() is not None

    old = time.time() - 10.0
    os.utime(feed_path, (old, old))
    assert feed.read() is None


def test_corrupt_file_is_none(feed_path):
    feed_path.write_bytes(b"not a jpeg at all")
    assert GameFeed(feed_path).read() is None


def test_unchanged_file_is_not_re_decoded(feed_path, monkeypatch):
    publish(feed_path, jpeg())
    feed = GameFeed(feed_path)
    first = feed.read()
    assert first is not None

    calls = []
    monkeypatch.setattr("flybrain.gamefeed._decode", lambda p: calls.append(p))
    assert feed.read() is first
    assert calls == []

    publish(feed_path, jpeg(colour=(200, 10, 10), size=(32, 32)))
    feed.read()
    assert calls == [feed_path]
