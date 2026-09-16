"""Supervisor tests. Hermetic: trivial python children, never the real stack."""

import io
import os
import signal
import socket
import subprocess
import sys
import time
import uuid

import pytest

from flybrain import app
from flybrain.app import (
    LONG,
    SHORT,
    Service,
    ServiceFailed,
    Supervisor,
    bot_save_path,
    group_alive,
    main,
    probe,
)

# Spawns a grandchild, prints its pid, then sleeps. The grandchild is the point:
# it is what `kill <pid>` on the leader would leave orphaned.
FORKER = (
    "import subprocess,sys,time;"
    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
    "print('grandchild',p.pid,flush=True);"
    "print('READY',flush=True);"
    "time.sleep(60)"
)


# A pgrep pattern is a regex, so the marker has to be free of regex punctuation.
# Unique per process so a leftover from a prior run can never be mistaken for this one's.
def stranger_marker() -> str:
    return f"flybrain_test_stranger_{os.getpid()}_{uuid.uuid4().hex}"


def sleeper(tmp_path, name="sleeper", tier=LONG, code=FORKER, ready="READY"):
    return Service(
        name=name,
        tier=tier,
        argv=[sys.executable, "-c", code],
        cwd=tmp_path,
        ready=ready,
        ready_timeout=10.0,
    )


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def sup(tmp_path):
    supervisor = Supervisor([], run_dir=tmp_path)
    yield supervisor
    supervisor.stop(grace=0.5)


def test_teardown_leaves_no_survivors(tmp_path, sup):
    service = sleeper(tmp_path)
    sup.services = [service]
    assert sup.start(service) == "started"

    leader = sup.procs["sleeper"].pid
    grandchild = int(
        next(
            line
            for line in service.log_path(tmp_path).read_text().splitlines()
            if "grandchild" in line
        ).split()[1]
    )
    assert alive(leader) and alive(grandchild)
    assert os.getpgid(leader) == leader, "child must lead its own process group"

    sup.stop()

    assert wait_gone(leader), "process group leader survived teardown"
    assert wait_gone(grandchild), "grandchild orphaned by teardown"
    assert not group_alive(leader)


def test_teardown_is_idempotent(tmp_path, sup):
    service = sleeper(tmp_path)
    sup.services = [service]
    sup.start(service)
    sup.stop()
    sup.stop()
    sup.stop()
    assert not sup.state_path.exists() or sup._load_state() == {}


def test_nonzero_exit_is_reported(tmp_path, sup):
    service = sleeper(
        tmp_path, code="print('READY',flush=True);import sys;sys.exit(3)", ready="READY"
    )
    sup.services = [service]
    sup.start(service)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not (dead := sup.deaths()):
        time.sleep(0.05)
    assert dead, "a child that exited was never reported"
    assert "exit 3" in dead[0]
    assert str(service.log_path(tmp_path)) in dead[0]
    assert sup.deaths() == [], "a death must only be reported once"


def test_failure_to_become_ready_names_the_log(tmp_path, sup):
    service = sleeper(tmp_path, code="print('nope',flush=True);import time;time.sleep(30)")
    object.__setattr__(service, "ready_timeout", 1.0)
    sup.services = [service]
    with pytest.raises(ServiceFailed) as exc:
        sup.start(service)
    assert str(service.log_path(tmp_path)) in str(exc.value)
    sup.stop(grace=0.5)


def test_listening_port_is_attached_not_double_started(tmp_path, sup):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        service = Service(
            name="listener",
            tier=SHORT,
            argv=[sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            port=port,
        )
        sup.services = [service]
        assert probe(service) is not None
        assert sup.start(service) == "attached"
        assert "listener" not in sup.procs, "attached service must not be spawned"
        assert sup._load_state() == {}, "attached service must not be recorded as ours"
    finally:
        listener.close()


def test_down_leaves_processes_it_did_not_start_alone(tmp_path, sup):
    marker = stranger_marker()
    stranger_code = f"import time; {marker} = 1; time.sleep(30)"
    stranger = subprocess.Popen([sys.executable, "-c", stranger_code], start_new_session=True)
    try:
        service = Service(
            name="stranger",
            tier=SHORT,
            argv=[sys.executable, "-c", stranger_code],
            cwd=tmp_path,
            pattern=marker,
        )
        sup.services = [service]
        assert sup.start(service) == "attached"
        sup.stop()
        assert alive(stranger.pid), "supervisor killed a stack it did not start"
    finally:
        stranger.kill()
        stranger.wait()


def test_adopt_takes_ownership_of_a_running_stack(tmp_path, sup):
    marker = stranger_marker()
    stranger_code = f"import time; {marker} = 1; time.sleep(30)"
    stranger = subprocess.Popen([sys.executable, "-c", stranger_code], start_new_session=True)
    try:
        service = Service(
            name="stranger",
            tier=SHORT,
            argv=[sys.executable, "-c", stranger_code],
            cwd=tmp_path,
            pattern=marker,
        )
        sup.services = [service]
        assert sup.start(service, adopt=True) == "adopted"
        sup.stop(grace=1.0)
        # wait() reaps it; until then it is a zombie and still answers signal 0.
        assert stranger.wait(timeout=5) == -signal.SIGTERM, "--adopt did not take ownership"
        assert not alive(stranger.pid)
    finally:
        if alive(stranger.pid):
            stranger.kill()
            stranger.wait()


def test_restart_short_does_not_touch_the_long_tier(tmp_path, sup):
    long_service = sleeper(tmp_path, name="long", tier=LONG)
    short_service = sleeper(tmp_path, name="short", tier=SHORT)
    sup.services = [long_service, short_service]
    sup.up()

    long_pid = sup.procs["long"].pid
    short_pid = sup.procs["short"].pid
    sup.restart_short()

    assert alive(long_pid), "restart-short killed the long tier"
    assert wait_gone(short_pid), "restart-short did not replace the short tier"
    assert sup.procs["short"].pid != short_pid


def test_status_reports_every_service_and_its_log(tmp_path, sup):
    sup.services = [sleeper(tmp_path, name="a"), sleeper(tmp_path, name="b", tier=SHORT)]
    out = sup.status()
    assert "a" in out and "b" in out
    assert str(sup.services[0].log_path(tmp_path)) in out


# --- reset-bot ----------------------------------------------------------


@pytest.fixture
def reset_stack(tmp_path, sup):
    """A running client + sidecar, a save file, and a captured output stream."""
    client = sleeper(tmp_path, name="lite", tier=LONG)
    sidecar = sleeper(tmp_path, name="sidecar", tier=SHORT)
    sup.services = [client, sidecar]
    sup.up()
    sup.out = io.StringIO()
    save = tmp_path / "flybot01.sav"
    save.write_bytes(b"x" * 862)
    return sup, client, sidecar, save


def logged_out(*_args, **_kwargs):
    return False


def order_of(out: str, *lines: str) -> list[int]:
    said = out.splitlines()
    return [said.index(line) for line in lines]


def test_reset_bot_cycles_the_client_inside_the_sidecar(reset_stack, monkeypatch):
    sup, client, sidecar, save = reset_stack
    monkeypatch.setattr(app, "gateway_in_game", logged_out)
    sup.reset_bot("flybot01", client, sidecar, save, settle=0.0)

    stop_sidecar, stop_client, start_client, start_sidecar = order_of(
        sup.out.getvalue(),
        "sidecar: stopped",
        "lite: stopped",
        "lite: started",
        "sidecar: started",
    )
    assert stop_sidecar < stop_client, "the controller must go down before the client"
    assert start_client < start_sidecar, "the sidecar must log in to a client that exists"
    assert not save.exists()


def test_reset_bot_deletes_the_save_only_once_logged_out(reset_stack, monkeypatch):
    sup, client, sidecar, save = reset_stack
    seen = []

    def still_in_game(*_args, **_kwargs):
        seen.append(save.exists())
        return len(seen) < 3

    monkeypatch.setattr(app, "gateway_in_game", still_in_game)
    sup.reset_bot("flybot01", client, sidecar, save, settle=0.0)

    assert seen == [True, True, True], "the save was touched while the bot was still in game"
    assert not save.exists()


def test_reset_bot_refuses_to_delete_a_save_it_cannot_prove_is_idle(reset_stack, monkeypatch):
    sup, client, sidecar, save = reset_stack
    monkeypatch.setattr(app, "gateway_in_game", lambda *a, **k: True)

    with pytest.raises(ServiceFailed) as exc:
        sup.reset_bot("flybot01", client, sidecar, save, timeout=0.5, settle=0.0)

    assert save.read_bytes() == b"x" * 862, "a bot that never logged out lost its save"
    assert "still in game" in str(exc.value)
    assert "sidecar stopped" in str(exc.value), "a half-reset must say what it did"
    assert "sidecar" not in sup._load_state(), "the stopped services were silently restarted"


def test_reset_bot_survives_a_missing_save(reset_stack, monkeypatch):
    sup, client, sidecar, save = reset_stack
    save.unlink()
    monkeypatch.setattr(app, "gateway_in_game", logged_out)
    sup.reset_bot("flybot01", client, sidecar, save, settle=0.0)
    assert "already fresh" in sup.out.getvalue()


def test_reset_bot_parses_with_bot_and_client(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        Supervisor,
        "reset_bot",
        lambda self, bot, client, sidecar, save: seen.update(
            bot=bot, client=client.name, sidecar=sidecar.name, save=save
        ),
    )
    assert main(["reset-bot", "--bot", "flybot09", "--client", "browser"]) == 0
    assert seen == {
        "bot": "flybot09",
        "client": "browser",
        "sidecar": "sidecar",
        "save": bot_save_path("flybot09"),
    }
