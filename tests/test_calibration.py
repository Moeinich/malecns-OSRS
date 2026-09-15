from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.connectome.loader import Connectome
from flybrain.engine.calibrate import Acceptance, summarize_rates
from flybrain.engine.calibration import (
    UNCALIBRATED,
    Calibration,
    Fingerprint,
    StaleCalibration,
)
from flybrain.engine.calibration import load as load_calibration


def _connectome(n: int = 6, scale: float = 1.0) -> Connectome:
    W = sp.csc_matrix(np.eye(n, k=1, dtype=np.float32) * np.float32(scale))
    return Connectome(
        W=W,
        body_ids=np.arange(n, dtype=np.int64),
        populations={"L1": np.arange(2), "DNp01": np.arange(2, 4)},
        provenance={"dataset": "test v0"},
    )


def _calibration(connectome: Connectome) -> Calibration:
    rates = summarize_rates(np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32), max_hz=50.0)
    return Calibration(
        gain=2.0334,
        spontaneous_noise_std=0.5,
        i_max=24.0,
        b=2.0,
        tau_w=150.0,
        acceptance=Acceptance(),
        rates=rates,
        measure_steps=8000,
        drive="sensory_drive over L1/L2/L3/Tm1/Mi1, amplitude 24.0",
        connectome=Fingerprint.of(connectome),
    )


def test_round_trips_through_json(tmp_path):
    c = _calibration(_connectome())
    path = c.save(tmp_path / "calibration_v1.json")

    back = load_calibration(path)
    assert back is not None
    assert back.gain == c.gain
    assert back.acceptance == c.acceptance
    assert back.connectome == c.connectome
    assert back.drive == c.drive
    assert back.measure_steps == 8000
    assert back.rates.percentiles_hz == c.rates.percentiles_hz
    assert back.rates.mean_hz == c.rates.mean_hz
    np.testing.assert_array_equal(back.rates.histogram, c.rates.histogram)
    np.testing.assert_array_equal(back.rates.bin_edges_hz, c.rates.bin_edges_hz)
    assert json.loads(path.read_text())["format"] == 1


def test_apply_returns_a_new_matrix_and_leaves_the_original_alone():
    connectome = _connectome()
    original = connectome.W.data.copy()
    scaled = _calibration(connectome).apply(connectome.W)

    assert scaled is not connectome.W
    assert scaled.data is not connectome.W.data
    np.testing.assert_array_equal(connectome.W.data, original)
    np.testing.assert_allclose(scaled.data, original * np.float32(2.0334), rtol=1e-6)

    scaled.data[:] = 0.0
    np.testing.assert_array_equal(connectome.W.data, original)


def test_a_different_network_is_a_hard_error(tmp_path):
    path = _calibration(_connectome()).save(tmp_path / "calibration_v1.json")

    with pytest.raises(StaleCalibration, match="different network"):
        load_calibration(path, _connectome(scale=3.0))

    with pytest.raises(StaleCalibration):
        load_calibration(path, _connectome(n=8))


def test_a_missing_artifact_is_none_not_an_exception(tmp_path):
    assert load_calibration(tmp_path / "nothing.json") is None
    assert UNCALIBRATED.gain == 1.0
    assert not UNCALIBRATED.calibrated
    assert UNCALIBRATED.engine_kwargs() == {}
    assert "UNCALIBRATED" in UNCALIBRATED.describe()


def test_engine_kwargs_and_encode_params_carry_the_values_through():
    c = _calibration(_connectome())
    assert c.engine_kwargs() == {"spontaneous_noise_std": 0.5, "b": 2.0, "tau_w": 150.0}
    assert c.encode_params().i_max == 24.0
    # Unset fields stay unset rather than overriding the engine's own defaults.
    assert Calibration(gain=1.0, b=3.0).engine_kwargs() == {"b": 3.0}
    assert "gain 2.0334" in c.describe()
    assert "1.0-5.0 Hz" in c.describe()


def test_the_live_path_builds_its_engine_and_encoder_from_the_calibration(tmp_path, monkeypatch):
    """The whole point: a calibrated value has to reach the engine and the encoder."""
    from flybrain.engine.lif import LIFEngine
    from flybrain.loop import run as run_module

    connectome = _connectome()
    path = _calibration(connectome).save(tmp_path / "calibration_v1.json")
    seen: dict[str, object] = {}

    def fake_engine(W, **kw):
        seen["W"] = W
        seen["kwargs"] = kw
        return LIFEngine(W, **kw)

    def fake_encoder(c, params=None):
        seen["encode_params"] = params
        return lambda frame: np.zeros(c.n, dtype=np.float32)

    monkeypatch.setattr(run_module, "load", lambda _p: connectome)
    monkeypatch.setattr(run_module, "LIFEngine", fake_engine)
    monkeypatch.setattr(run_module, "default_encoder", fake_encoder)
    monkeypatch.setattr(run_module.Agent, "run", lambda self, ticks=None: iter(()))
    monkeypatch.setattr(run_module.CollisionGrid, "load", classmethod(lambda cls, p: None))
    monkeypatch.setattr(run_module.MotorIndex, "from_connectome", lambda c: None)
    monkeypatch.setattr(run_module.BridgeClient, "__enter__", lambda self: self)
    monkeypatch.setattr(run_module.BridgeClient, "__exit__", lambda self, *a: False)

    assert run_module.main(["--calibration", str(path), "--ticks", "0"]) == 0

    assert seen["kwargs"] == {
        "seed": 0,
        "spontaneous_noise_std": 0.5,
        "b": 2.0,
        "tau_w": 150.0,
    }
    assert seen["encode_params"].i_max == 24.0
    np.testing.assert_allclose(seen["W"].data, connectome.W.data * np.float32(2.0334), rtol=1e-6)


def test_a_run_without_an_artifact_says_so_and_runs_raw(tmp_path, monkeypatch, capsys):
    from flybrain.loop import run as run_module

    connectome = _connectome()
    seen: dict[str, object] = {}

    monkeypatch.setattr(run_module, "load", lambda _p: connectome)
    monkeypatch.setattr(
        run_module, "LIFEngine", lambda W, **kw: seen.update(W=W, kwargs=kw) or _Stub()
    )
    monkeypatch.setattr(run_module, "default_encoder", lambda c, params=None: None)
    monkeypatch.setattr(run_module.Agent, "run", lambda self, ticks=None: iter(()))
    monkeypatch.setattr(run_module.CollisionGrid, "load", classmethod(lambda cls, p: None))
    monkeypatch.setattr(run_module.MotorIndex, "from_connectome", lambda c: None)
    monkeypatch.setattr(run_module.BridgeClient, "__enter__", lambda self: self)
    monkeypatch.setattr(run_module.BridgeClient, "__exit__", lambda self, *a: False)

    assert run_module.main(["--calibration", str(tmp_path / "gone.json"), "--ticks", "0"]) == 0

    err = capsys.readouterr().err
    assert "UNCALIBRATED: running at gain 1.0, the brain will be silent" in err
    assert "calibration     UNCALIBRATED" in err
    assert seen["kwargs"] == {"seed": 0}
    np.testing.assert_array_equal(seen["W"].data, connectome.W.data)


class _Stub:
    dt_ms = 1.0
