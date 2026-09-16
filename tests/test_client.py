"""BridgeClient against a fake sidecar: a real unix socket, no game."""

from __future__ import annotations

import json
import socket
import tempfile
import threading

import pytest

from flybrain.loop import BridgeClient, ProtocolError, Walk


class FakeSidecar:
    """A listening unix socket that speaks NDJSON the way the sidecar does."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(path)
        self.listener.listen(4)
        self.conn: socket.socket | None = None
        self._buffer = b""

    def accept(self) -> None:
        self.conn, _ = self.listener.accept()
        self._buffer = b""

    def recv_line(self) -> dict:
        assert self.conn is not None
        while b"\n" not in self._buffer:
            chunk = self.conn.recv(4096)
            assert chunk, "client closed while we waited for a line"
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def send_raw(self, payload: bytes) -> None:
        assert self.conn is not None
        self.conn.sendall(payload)

    def send(self, msg: dict) -> None:
        self.send_raw((json.dumps(msg) + "\n").encode())

    def drip(self, payload: bytes, chunk: int) -> threading.Thread:
        """Feed a message in deliberately fragmented writes, from a thread."""

        def run() -> None:
            for i in range(0, len(payload), chunk):
                self.send_raw(payload[i : i + chunk])

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def close_conn(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def close(self) -> None:
        self.close_conn()
        self.listener.close()


READY = {
    "t": "ready",
    "protocol": 1,
    "username": "flybot01",
    "mode": "control",
    "role": "brain",
    "tickMs": 400,
    "pathfindingWarm": True,
}


def npc(index: int) -> dict:
    return {
        "id": 3266,
        "index": index,
        "name": f"Chicken {index}",
        "combatLevel": 1,
        "x": 3230 + index,
        "z": 3295,
        "size": 1,
        "distance": 4,
        "hp": 3,
        "maxHp": 3,
        "inCombat": False,
        "targetIndex": -1,
        "reachable": True,
        "options": ["Attack"],
    }


def state_msg(revision: int, npc_count: int = 1) -> dict:
    return {
        "t": "state",
        "revision": revision,
        "tick": 88000 + revision,
        "droppedSinceLast": 2,
        "deadlineMs": 240,
        "tickMs": 400,
        "observedTickMs": None,
        "state": {
            "tick": 88000 + revision,
            "inGame": True,
            "modalOpen": False,
            "player": {
                "name": "flybot01",
                "combatLevel": 3,
                "hp": 10,
                "maxHp": 10,
                "x": 3222,
                "z": 3218,
                "level": 0,
                "runEnergy": 100,
                "animId": -1,
                "inCombat": False,
                "targetIndex": -1,
                "targetType": "none",
                "isDead": False,
                "lifeId": 1,
            },
            "npcs": [npc(i) for i in range(npc_count)],
            "groundItems": [
                {
                    "id": 526,
                    "name": "Bones",
                    "count": 1,
                    "x": 3223,
                    "z": 3219,
                    "distance": 1,
                    "reachable": True,
                }
            ],
            "locs": [
                {
                    "id": 1276,
                    "name": "Tree",
                    "x": 3220,
                    "z": 3220,
                    "distance": 3,
                    "options": ["Chop"],
                }
            ],
            "inventory": [{"slot": 0, "id": 2309, "name": "Bread", "count": 1}],
            "skills": {"Attack": 0, "Hitpoints": 1154},
            "opRejectedCount": 0,
        },
    }


@pytest.fixture
def sidecar():
    # Not tmp_path: pytest's paths overflow the 104-byte AF_UNIX limit.
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        server = FakeSidecar(f"{tmp}/b.sock")
        yield server
        server.close()


@pytest.fixture
def client(sidecar):
    client = BridgeClient(sidecar.path, backoff_initial=0.01, backoff_max=0.01)
    with client:
        sidecar.accept()
        yield client


def test_hello_handshake(sidecar, client):
    assert sidecar.recv_line() == {"t": "hello", "protocol": 1, "role": "brain"}
    sidecar.send(READY)
    sidecar.send(state_msg(1))
    state = next(client.states())
    assert client.ready is not None
    assert client.ready.username == "flybot01"
    assert state.revision == 1
    assert client.dropped_game_ticks == 2


def test_large_state_arrives_fragmented(sidecar, client):
    sidecar.recv_line()
    payload = (json.dumps(state_msg(7, npc_count=120)) + "\n").encode()
    assert len(payload) > 8192
    thread = sidecar.drip(payload, chunk=1000)
    state = next(client.states())
    thread.join(5)
    assert len(state.state.npcs) == 120
    assert state.state.npcs[-1].name == "Chicken 119"
    assert state.state.npcs[-1].x == 3230 + 119


def test_two_messages_in_one_packet(sidecar, client):
    sidecar.recv_line()
    packet = b"".join((json.dumps(m) + "\n").encode() for m in (state_msg(1), state_msg(2)))
    sidecar.send_raw(packet)
    states = client.states()
    assert next(states).revision == 1
    assert next(states).revision == 2


def test_malformed_line_raises(sidecar, client):
    sidecar.recv_line()
    sidecar.send_raw(b"{not json}\n")
    with pytest.raises(ProtocolError):
        next(client.states())


def test_unknown_message_type_raises(sidecar, client):
    sidecar.recv_line()
    sidecar.send_raw(b'{"t":"surprise"}\n')
    with pytest.raises(ProtocolError):
        next(client.states())


def test_renamed_field_raises(sidecar, client):
    sidecar.recv_line()
    broken = state_msg(1)
    broken["state"]["player"]["hitpoints"] = broken["state"]["player"].pop("hp")
    sidecar.send(broken)
    with pytest.raises(ProtocolError):
        next(client.states())


def test_send_cmd_echoes_revision(sidecar, client):
    sidecar.recv_line()
    sidecar.send(state_msg(42))
    state = next(client.states())
    cmd_id = client.send_cmd(state.revision, Walk(3222, 3218, running=True))
    sent = sidecar.recv_line()
    assert sent == {
        "t": "cmd",
        "cmdId": cmd_id,
        "revision": 42,
        "action": {"kind": "walk", "x": 3222, "z": 3218, "running": True},
    }
    client.send_noop(state.revision)
    assert sidecar.recv_line() == {"t": "noop", "revision": 42}


def test_reconnects_after_server_closes(sidecar, client):
    sidecar.recv_line()
    sidecar.send(state_msg(1))
    states = client.states()
    assert next(states).revision == 1

    sidecar.close_conn()
    reconnected = threading.Thread(target=sidecar.accept, daemon=True)
    reconnected.start()
    next_state = None

    def pump() -> None:
        nonlocal next_state
        next_state = next(states)

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()
    reconnected.join(5)
    assert sidecar.recv_line()["t"] == "hello"
    sidecar.send(state_msg(2))
    pump_thread.join(5)
    assert next_state is not None
    assert next_state.revision == 2
    assert client.reconnects == 1


def test_send_reset_and_wait_for_the_tile_it_landed_on(sidecar, client):
    sidecar.recv_line()
    cmd_id = client.send_reset(3222, 3218)
    assert sidecar.recv_line() == {"t": "reset", "cmdId": cmd_id, "x": 3222, "z": 3218}

    # A state in front of the ack must not be mistaken for it, and an ack for
    # some other command must not end the wait either.
    sidecar.send(state_msg(1))
    sidecar.send(
        {
            "t": "ack",
            "cmdId": cmd_id + 1,
            "ok": True,
            "phase": "completion",
            "opRejectedDelta": 0,
            "message": "other",
        }
    )
    sidecar.send(
        {
            "t": "ack",
            "cmdId": cmd_id,
            "ok": True,
            "phase": "reset",
            "opRejectedDelta": 0,
            "message": "Arrived",
            "x": 3222,
            "z": 3218,
        }
    )
    assert client.wait_reset(cmd_id, timeout_s=5) == (True, 3222, 3218)

    sidecar.send(state_msg(2))
    assert next(client.states()).revision == 2


def test_a_failed_reset_ack_carries_where_the_bot_actually_is(sidecar, client):
    sidecar.recv_line()
    cmd_id = client.send_reset(3222, 3218)
    sidecar.recv_line()
    sidecar.send(
        {
            "t": "ack",
            "cmdId": cmd_id,
            "ok": False,
            "phase": "reset",
            "opRejectedDelta": 0,
            "message": "Pathfinding failed",
            "x": 3300,
            "z": 3190,
        }
    )
    assert client.wait_reset(cmd_id, timeout_s=5) == (False, 3300, 3190)
    assert client.last_ack is not None
    assert client.last_ack.phase == "reset"
