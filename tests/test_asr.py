from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


class _FakeRawSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    def __init__(self, duration: float, language: str = "en", language_probability: float = 0.99) -> None:
        self.duration = duration
        self.language = language
        self.language_probability = language_probability


class _FakeWhisperModel:
    instances: list["_FakeWhisperModel"] = []

    def __init__(self, model_name: str, device: str, compute_type: str, download_root: str) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        _FakeWhisperModel.instances.append(self)

    def transcribe(self, audio_path: str, language=None, beam_size: int = 5):
        self.last_call = {"audio_path": audio_path, "language": language, "beam_size": beam_size}
        segments = [
            _FakeRawSegment(0.0, 1.5, " hello world "),
            _FakeRawSegment(1.5, 4.0, " how are you today"),
            _FakeRawSegment(4.0, 6.2, " I am fine thanks"),
        ]
        return iter(segments), _FakeInfo(duration=6.2, language=language or "en")


@pytest.fixture
def asr(monkeypatch: pytest.MonkeyPatch):
    _FakeWhisperModel.instances.clear()
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    sys.modules.pop("encocoa.asr", None)
    import importlib

    return importlib.import_module("encocoa.asr")


def test_transcribe_returns_segments_and_stats(asr, tmp_path: Path) -> None:
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"\0\0")  # not a real WAV; the fake model doesn't parse it

    segs, stats = asr.transcribe(
        wav_path=wav,
        model_name="tiny.en",
        model_dir=tmp_path / "models",
        language="en",
        device="cpu",
        compute_type="int8",
        progress_stream=None,
    )

    assert [s.text for s in segs] == ["hello world", "how are you today", "I am fine thanks"]
    assert stats.audio_duration == pytest.approx(6.2)
    assert stats.segment_count == 3
    assert stats.word_count == 2 + 4 + 4
    assert stats.char_count == sum(len(s.text) for s in segs)
    assert stats.model == "tiny.en"
    assert stats.real_time_factor > 0


def test_transcribe_passes_through_model_args(asr, tmp_path: Path) -> None:
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"\0")
    asr.transcribe(
        wav_path=wav,
        model_name="base.en",
        model_dir=tmp_path / "cache",
        language="en",
        device="cpu",
        compute_type="int8",
        beam_size=3,
        progress_stream=None,
    )
    inst = _FakeWhisperModel.instances[-1]
    assert inst.model_name == "base.en"
    assert inst.device == "cpu"
    assert inst.compute_type == "int8"
    assert Path(inst.download_root) == (tmp_path / "cache").resolve() or Path(inst.download_root) == tmp_path / "cache"
    assert inst.last_call["beam_size"] == 3
    assert inst.last_call["language"] == "en"


def test_transcribe_missing_file_raises(asr, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        asr.transcribe(
            wav_path=tmp_path / "nope.wav",
            model_dir=tmp_path / "models",
            progress_stream=None,
        )


def test_save_transcript_writes_expected_schema(asr, tmp_path: Path) -> None:
    segs = [
        asr.TranscriptSegment(start=0.0, end=1.0, text="hi"),
        asr.TranscriptSegment(start=1.0, end=2.5, text="there"),
    ]
    out = tmp_path / "out" / "session.transcript.json"
    asr.save_transcript(segs, out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == [
        {"start": 0.0, "end": 1.0, "text": "hi"},
        {"start": 1.0, "end": 2.5, "text": "there"},
    ]


def test_format_stats_contains_key_fields(asr) -> None:
    stats = asr.TranscriptionStats(
        audio_duration=120.0,
        processing_time=30.0,
        language="en",
        language_probability=0.99,
        model="small.en",
        segment_count=10,
        word_count=200,
        char_count=1000,
    )
    out = asr.format_stats(stats)
    for token in ("model=small.en", "audio=120.0s", "proc=30.0s", "rtf=4.00", "words=200", "chars=1000"):
        assert token in out
