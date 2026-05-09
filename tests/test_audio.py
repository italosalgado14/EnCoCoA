from __future__ import annotations

import importlib
import sys
import types

import pytest


def _stub_sounddevice() -> types.ModuleType:
    """Build a minimal sounddevice stub so the audio module imports without PortAudio."""
    sd = types.ModuleType("sounddevice")
    sd.query_devices = lambda: []  # type: ignore[attr-defined]
    sd.default = types.SimpleNamespace(device=(None, None))  # type: ignore[attr-defined]
    sd.InputStream = object  # type: ignore[attr-defined]
    return sd


@pytest.fixture
def audio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _stub_sounddevice())
    sys.modules.pop("encocoa.audio", None)
    return importlib.import_module("encocoa.audio")


def test_format_hms(audio) -> None:
    assert audio._format_hms(0) == "0:00"
    assert audio._format_hms(5) == "0:05"
    assert audio._format_hms(65) == "1:05"
    assert audio._format_hms(3661) == "1:01:01"
    assert audio._format_hms(-3) == "0:00"


def test_level_bar_silence_clamps_low(audio) -> None:
    bar = audio._level_bar(0.0, width=10)
    assert bar.startswith("[")
    assert "----------" in bar
    assert "dBFS" in bar


def test_level_bar_full_scale(audio) -> None:
    bar = audio._level_bar(1.0, width=10)
    assert "##########" in bar
    assert "0.0 dBFS" in bar


def test_rms_int16_zero(audio) -> None:
    import numpy as np

    assert audio._rms_int16(np.zeros(0, dtype=np.int16)) == 0.0
    assert audio._rms_int16(np.zeros(100, dtype=np.int16)) == 0.0


def test_rms_int16_full_scale(audio) -> None:
    import numpy as np

    chunk = np.full(1000, 32767, dtype=np.int16)
    assert audio._rms_int16(chunk) == pytest.approx(1.0, rel=1e-3)


def test_list_input_devices_filters_outputs(
    audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = [
        {"name": "Mic A", "max_input_channels": 2, "default_samplerate": 48000.0},
        {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
        {"name": "Mic B", "max_input_channels": 1, "default_samplerate": 16000.0},
    ]
    monkeypatch.setattr(audio.sd, "query_devices", lambda: fake)
    devs = audio.list_input_devices()
    assert [d.name for d in devs] == ["Mic A", "Mic B"]
    assert devs[0].index == 0
    assert devs[1].index == 2


class _IndexablePair:
    """Mimics sounddevice._InputOutputPair: indexable like a 2-tuple but not a tuple."""

    def __init__(self, a, b) -> None:
        self._items = (a, b)

    def __getitem__(self, i):
        return self._items[i]


def test_default_input_index_handles_indexable_pair(
    audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audio.sd, "default", types.SimpleNamespace(device=_IndexablePair(3, 5)))
    assert audio.default_input_index() == 3


def test_default_input_index_no_default(
    audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audio.sd, "default", types.SimpleNamespace(device=_IndexablePair(-1, -1)))
    assert audio.default_input_index() is None


def test_default_input_index_plain_int(
    audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audio.sd, "default", types.SimpleNamespace(device=7))
    assert audio.default_input_index() == 7


def test_resolve_input_device_passes_through_int_and_none(
    audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert audio.resolve_input_device(None) is None
    assert audio.resolve_input_device("") is None
    assert audio.resolve_input_device(3) == 3
    assert audio.resolve_input_device("5") == 5


def test_resolve_input_device_substring_match(
    audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = [
        {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 48000.0},
        {"name": "Built-in", "max_input_channels": 2, "default_samplerate": 48000.0},
    ]
    monkeypatch.setattr(audio.sd, "query_devices", lambda: fake)
    assert audio.resolve_input_device("usb") == 0
    assert audio.resolve_input_device("Built") == 1


def test_resolve_input_device_no_match_raises(
    audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = [
        {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 48000.0},
    ]
    monkeypatch.setattr(audio.sd, "query_devices", lambda: fake)
    with pytest.raises(ValueError, match="No input device matches"):
        audio.resolve_input_device("nope")


def test_resolve_input_device_ambiguous_match_raises(
    audio, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = [
        {"name": "Mic A", "max_input_channels": 1, "default_samplerate": 48000.0},
        {"name": "Mic B", "max_input_channels": 1, "default_samplerate": 48000.0},
    ]
    monkeypatch.setattr(audio.sd, "query_devices", lambda: fake)
    with pytest.raises(ValueError, match="Multiple input devices"):
        audio.resolve_input_device("mic")


def test_print_input_devices_marks_default(
    audio, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = [
        {"name": "Mic A", "max_input_channels": 1, "default_samplerate": 48000.0},
        {"name": "Mic B", "max_input_channels": 1, "default_samplerate": 16000.0},
    ]
    monkeypatch.setattr(audio.sd, "query_devices", lambda: fake)
    monkeypatch.setattr(audio.sd, "default", types.SimpleNamespace(device=(1, 2)))
    audio.print_input_devices()
    out = capsys.readouterr().out
    assert "Mic A" in out
    assert "Mic B" in out
    assert "[default]" in out
    # default is index 1 (Mic B)
    mic_b_line = next(line for line in out.splitlines() if "Mic B" in line)
    assert "[default]" in mic_b_line
