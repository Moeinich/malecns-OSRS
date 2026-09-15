"""Blocking NDJSON client for the Bun sidecar's unix socket.

Synchronous on purpose: the brain is a numpy loop, so `for state in
client.states()` is the whole API and there is no asyncio anywhere below it.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Iterator
from types import TracebackType
from typing import Self

from flybrain.loop.types import (
    PROTOCOL_VERSION,
    Ack,
    Action,
    BridgeError,
    ProtocolError,
    Ready,
    Reward,
    ServerMessage,
    StateUpdate,
    parse_server_message,
)

RECV_SIZE = 65536


def default_socket_path() -> str:
    override = os.environ.get("RS_BRIDGE_SOCKET")
    if override:
        return override
    return f"/tmp/malecns-osrs/{os.environ.get('RS_BOT_USERNAME', 'flybot01')}.sock"


class BridgeClient:
    """One connection to the sidecar, reconnecting if the sidecar outlives us."""

    def __init__(
        self,
        socket_path: str | None = None,
        role: str = "brain",
        *,
        reconnect: bool = True,
        backoff_initial: float = 0.1,
        backoff_max: float = 5.0,
    ) -> None:
        self.socket_path = socket_path or default_socket_path()
        self.role = role
        self.reconnect = reconnect
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max

        self.ready: Ready | None = None
        self.last_reward: Reward | None = None
        self.last_ack: Ack | None = None
        self.dropped_game_ticks = 0
        self.overruns = 0
        self.reconnects = 0

        self._sock: socket.socket | None = None
        self._buffer = b""
        self._cmd_id = 0
        self._deadline_at: float | None = None

    # -------------------------------------------------- connection

    def connect(self) -> None:
        self.close()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        self._sock = sock
        self._buffer = b""
        self._deadline_at = None
        self._send({"t": "hello", "protocol": PROTOCOL_VERSION, "role": self.role})

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -------------------------------------------------- sending

    def send_cmd(self, revision: int, action: Action) -> int:
        """Echo `revision` so the sidecar can reject a command aimed at a past tick."""
        self._note_deadline()
        self._cmd_id += 1
        self._send(
            {
                "t": "cmd",
                "cmdId": self._cmd_id,
                "revision": revision,
                "action": action.to_dict(),
            }
        )
        return self._cmd_id

    def send_noop(self, revision: int) -> None:
        self._note_deadline()
        self._send({"t": "noop", "revision": revision})

    def _note_deadline(self) -> None:
        if self._deadline_at is not None and time.monotonic() > self._deadline_at:
            self.overruns += 1
        self._deadline_at = None

    def _send(self, msg: dict[str, object]) -> None:
        if self._sock is None:
            raise ConnectionError("not connected to the sidecar")
        self._sock.sendall((json.dumps(msg) + "\n").encode())

    # -------------------------------------------------- receiving

    def states(self) -> Iterator[StateUpdate]:
        """Yield every state message, reconnecting with backoff if the socket drops."""
        backoff = self.backoff_initial
        while True:
            if self._sock is None:
                self.connect()
            try:
                for msg in self._messages():
                    backoff = self.backoff_initial
                    if isinstance(msg, StateUpdate):
                        self.dropped_game_ticks += msg.dropped_since_last
                        self._deadline_at = time.monotonic() + msg.deadline_ms / 1000.0
                        yield msg
            except (ConnectionError, OSError) as exc:
                if not self.reconnect:
                    raise ConnectionError(f"sidecar connection lost: {exc}") from exc
            self.close()
            if not self.reconnect:
                return
            self.reconnects += 1
            time.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max)

    def _messages(self) -> Iterator[ServerMessage]:
        """Every message until the sidecar closes; side effects applied here."""
        while True:
            line = self._read_line()
            if line is None:
                return
            msg = parse_server_message(_decode(line))
            if isinstance(msg, Ready):
                self.ready = msg
            elif isinstance(msg, Reward):
                self.last_reward = msg
            elif isinstance(msg, Ack):
                self.last_ack = msg
            elif isinstance(msg, BridgeError):
                raise ProtocolError(f"sidecar error: {msg.message}")
            yield msg

    def _read_line(self) -> bytes | None:
        """Buffer until a newline: a state message exceeds one `recv`, and a
        partial read that is parsed as a whole message is the truncation bug."""
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line, self._buffer = self._buffer[:newline], self._buffer[newline + 1 :]
                return line
            if self._sock is None:
                raise ConnectionError("not connected to the sidecar")
            chunk = self._sock.recv(RECV_SIZE)
            if not chunk:
                if self._buffer:
                    raise ProtocolError("sidecar closed mid-line")
                return None
            self._buffer += chunk


def _decode(line: bytes) -> dict[str, object]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"malformed line: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProtocolError(f"expected a JSON object, got {type(raw).__name__}")
    return raw
