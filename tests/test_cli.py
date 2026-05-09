from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from encocoa import __version__
from encocoa.cli import build_parser, main


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for sub in ("record", "process", "coach", "run"):
        assert sub in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_errors() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0


def test_run_help_lists_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--out-dir",
        "--name",
        "--duration",
        "--device",
        "--model",
        "--diarizer",
        "--num-speakers",
        "--llm",
        "--no-coach",
        "--no-open",
    ):
        assert flag in out


def test_record_help_lists_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["record", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--duration", "--out", "--device", "--list-devices", "--samplerate", "--vad-trim"):
        assert flag in out


def test_record_list_devices(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa

    fake = types.ModuleType("encocoa.audio")

    def _print_input_devices() -> None:
        print("Available input devices:\n  [ 0] Fake Mic  (1 ch, 48000 Hz)  [default]")

    fake.print_input_devices = _print_input_devices  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "encocoa.audio", fake)
    monkeypatch.setattr(encocoa, "audio", fake, raising=False)

    rc = main(["record", "--list-devices"])
    assert rc == 0
    assert "Fake Mic" in capsys.readouterr().out


def test_record_vad_trim_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["record", "--vad-trim"])
    assert rc == 2
    assert "vad-trim" in capsys.readouterr().err


def test_record_invalid_duration(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["record", "--duration", "0"])
    assert rc == 2
    assert "duration" in capsys.readouterr().err


def test_process_help_lists_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["process", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--model",
        "--out-dir",
        "--language",
        "--device",
        "--compute-type",
        "--beam-size",
        "--model-dir",
        "--diarizer",
        "--num-speakers",
        "--hf-token",
        "--no-diarize",
    ):
        assert flag in out


def test_process_missing_wav_returns_error(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["process", str(tmp_path / "missing.wav")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


class _FakeASRSeg:
    def __init__(self, s: float, e: float, t: str) -> None:
        self.start, self.end, self.text = s, e, t

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


def _stub_wav_utils() -> types.ModuleType:
    """Stub wav_utils so process/run tests don't have to write valid WAV files."""
    fake = types.ModuleType("encocoa.wav_utils")
    fake.SILENT_THRESHOLD_DBFS = -50.0  # type: ignore[attr-defined]
    fake.peak_dbfs = lambda _path: 0.0  # type: ignore[attr-defined]
    fake.is_effectively_silent = lambda _path, **_: False  # type: ignore[attr-defined]
    return fake


def _install_wav_utils_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    import encocoa

    fake = _stub_wav_utils()
    monkeypatch.setitem(sys.modules, "encocoa.wav_utils", fake)
    monkeypatch.setattr(encocoa, "wav_utils", fake, raising=False)


def _stub_asr_module(tmp_path, captured: dict) -> types.ModuleType:
    fake = types.ModuleType("encocoa.asr")
    fake.DEFAULT_CACHE_DIR = tmp_path / "models"  # type: ignore[attr-defined]

    def _transcribe(**kwargs):
        captured.update(kwargs)
        segs = [_FakeASRSeg(0.0, 1.0, "hi"), _FakeASRSeg(1.0, 2.0, "there")]
        return segs, object()

    def _save_transcript(segs, out_path):
        from pathlib import Path as _P

        p = _P(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[]\n", encoding="utf-8")
        captured["saved_transcript_to"] = p
        return p

    fake.transcribe = _transcribe  # type: ignore[attr-defined]
    fake.save_transcript = _save_transcript  # type: ignore[attr-defined]
    fake.format_stats = lambda s: "STATS-LINE"  # type: ignore[attr-defined]
    fake.TranscriptSegment = _FakeASRSeg  # type: ignore[attr-defined]
    return fake


def test_process_no_diarize_writes_only_transcript(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa

    captured: dict = {}
    fake_asr = _stub_asr_module(tmp_path, captured)
    monkeypatch.setitem(sys.modules, "encocoa.asr", fake_asr)
    monkeypatch.setattr(encocoa, "asr", fake_asr, raising=False)
    _install_wav_utils_stub(monkeypatch)

    wav = tmp_path / "session.wav"
    wav.write_bytes(b"\0")

    rc = main(["process", str(wav), "--model", "tiny.en", "--language", "en", "--no-diarize"])
    captured_io = capsys.readouterr()

    assert rc == 0
    assert captured["model_name"] == "tiny.en"
    assert captured["language"] == "en"
    assert captured["wav_path"] == wav
    assert captured["saved_transcript_to"] == tmp_path / "session.transcript.json"
    # No diarization/dialog/markdown files should exist
    assert not (tmp_path / "session.diarization.json").exists()
    assert not (tmp_path / "session.dialog.json").exists()
    assert not (tmp_path / "session.transcript.md").exists()
    assert "transcript-only" in captured_io.out


def test_process_full_pipeline_writes_all_outputs(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa

    captured: dict = {}
    fake_asr = _stub_asr_module(tmp_path, captured)

    fake_diarize = types.ModuleType("encocoa.diarize")

    class _DSeg:
        def __init__(self, s, e, sp):
            self.start, self.end, self.speaker = s, e, sp

        def to_dict(self):
            return {"start": self.start, "end": self.end, "speaker": self.speaker}

    def _diarize_simple(wav_path, num_speakers=2):
        captured["diarize_called_with"] = (wav_path, num_speakers)
        return [_DSeg(0.0, 1.0, "A"), _DSeg(1.0, 2.0, "B")]

    def _save_diarization(segs, out_path):
        from pathlib import Path as _P

        p = _P(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[]\n", encoding="utf-8")
        return p

    fake_diarize.DiarizationSegment = _DSeg  # type: ignore[attr-defined]
    fake_diarize.diarize_simple = _diarize_simple  # type: ignore[attr-defined]
    fake_diarize.diarize_pyannote = lambda **_: (_ for _ in ()).throw(AssertionError("should not call pyannote"))  # type: ignore[attr-defined]
    fake_diarize.save_diarization = _save_diarization  # type: ignore[attr-defined]
    fake_diarize.diarization_summary = lambda segs: {"A": 1.0, "B": 1.0}  # type: ignore[attr-defined]

    fake_merge = types.ModuleType("encocoa.merge")

    class _Turn:
        def __init__(self, s, e, sp, t):
            self.start, self.end, self.speaker, self.text = s, e, sp, t

        def to_dict(self):
            return {"start": self.start, "end": self.end, "speaker": self.speaker, "text": self.text}

    def _merge(asr_segs, diar_segs):
        captured["merge_inputs"] = (len(asr_segs), len(diar_segs))
        return [_Turn(0.0, 1.0, "A", "hi"), _Turn(1.0, 2.0, "B", "there")]

    def _coalesce(turns):
        return turns

    def _save_dialog(turns, out_path):
        from pathlib import Path as _P

        p = _P(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[]\n", encoding="utf-8")
        return p

    fake_merge.DialogTurn = _Turn  # type: ignore[attr-defined]
    fake_merge.merge = _merge  # type: ignore[attr-defined]
    fake_merge.coalesce_consecutive = _coalesce  # type: ignore[attr-defined]
    fake_merge.save_dialog = _save_dialog  # type: ignore[attr-defined]

    fake_report = types.ModuleType("encocoa.report")

    def _save_md(turns, out_path, *, title=None, names=None):
        from pathlib import Path as _P

        p = _P(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {title or 'x'}\n", encoding="utf-8")
        captured["md_title"] = title
        captured["md_names"] = names
        return p

    fake_report.save_markdown = _save_md  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "encocoa.asr", fake_asr)
    monkeypatch.setitem(sys.modules, "encocoa.diarize", fake_diarize)
    monkeypatch.setitem(sys.modules, "encocoa.merge", fake_merge)
    monkeypatch.setitem(sys.modules, "encocoa.report", fake_report)
    monkeypatch.setattr(encocoa, "asr", fake_asr, raising=False)
    monkeypatch.setattr(encocoa, "diarize", fake_diarize, raising=False)
    monkeypatch.setattr(encocoa, "merge", fake_merge, raising=False)
    monkeypatch.setattr(encocoa, "report", fake_report, raising=False)
    _install_wav_utils_stub(monkeypatch)

    wav = tmp_path / "session.wav"
    wav.write_bytes(b"\0")
    out_dir = tmp_path / "out"

    rc = main(
        [
            "process",
            str(wav),
            "--model",
            "tiny.en",
            "--diarizer",
            "simple",
            "--out-dir",
            str(out_dir),
        ]
    )
    captured_io = capsys.readouterr()

    assert rc == 0
    assert (out_dir / "session.transcript.json").exists()
    assert (out_dir / "session.diarization.json").exists()
    assert (out_dir / "session.dialog.json").exists()
    assert (out_dir / "session.transcript.md").exists()
    assert captured["diarize_called_with"][1] == 2  # default num_speakers
    assert captured["merge_inputs"] == (2, 2)
    assert captured["md_title"] == "session"
    assert "done" in captured_io.out


def test_process_pyannote_runtime_error_is_reported(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa

    captured: dict = {}
    fake_asr = _stub_asr_module(tmp_path, captured)

    fake_diarize = types.ModuleType("encocoa.diarize")

    def _diarize_pyannote(**kwargs):
        raise RuntimeError("token missing")

    fake_diarize.diarize_pyannote = _diarize_pyannote  # type: ignore[attr-defined]
    fake_diarize.diarize_simple = lambda **_: []  # type: ignore[attr-defined]

    fake_merge = types.ModuleType("encocoa.merge")
    fake_report = types.ModuleType("encocoa.report")

    monkeypatch.setitem(sys.modules, "encocoa.asr", fake_asr)
    monkeypatch.setitem(sys.modules, "encocoa.diarize", fake_diarize)
    monkeypatch.setitem(sys.modules, "encocoa.merge", fake_merge)
    monkeypatch.setitem(sys.modules, "encocoa.report", fake_report)
    monkeypatch.setattr(encocoa, "asr", fake_asr, raising=False)
    monkeypatch.setattr(encocoa, "diarize", fake_diarize, raising=False)
    monkeypatch.setattr(encocoa, "merge", fake_merge, raising=False)
    monkeypatch.setattr(encocoa, "report", fake_report, raising=False)
    _install_wav_utils_stub(monkeypatch)

    wav = tmp_path / "session.wav"
    wav.write_bytes(b"\0")

    rc = main(["process", str(wav), "--diarizer", "pyannote"])
    err = capsys.readouterr().err
    assert rc == 4
    assert "diarization failed" in err
    assert "token missing" in err


def test_parser_builds() -> None:
    assert build_parser().prog == "encocoa"


def test_coach_help_lists_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["coach", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("dialog", "--out", "--language", "--llm", "--ollama-model", "--ollama-host"):
        assert flag in out


def test_coach_missing_dialog_returns_error(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["coach", str(tmp_path / "missing.dialog.json")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def _stub_coach_module(captured: dict) -> types.ModuleType:
    fake = types.ModuleType("encocoa.coach")

    class _Corr:
        def __init__(self, rule_id="R", category="C", message="m"):
            self.rule_id = rule_id
            self.category = category
            self.message = message
            self.replacement = None
            self.offset = 0
            self.length = 0
            self.context = ""

        def to_dict(self):
            return {"rule_id": self.rule_id, "category": self.category, "message": self.message}

    class _Rep:
        def __init__(self, speaker, text, corrections=None, rewrite=None):
            self.start = 0.0
            self.end = 1.0
            self.speaker = speaker
            self.text = text
            self.corrections = corrections or []
            self.rewrite = rewrite

    class _Stat:
        def __init__(self, speaker, correction_count):
            self.speaker = speaker
            self.turn_count = 1
            self.word_count = 2
            self.correction_count = correction_count
            self.top_categories = []
            self.top_rules = []

    def _load_dialog(path):
        captured["loaded"] = path
        return [{"start": 0.0, "end": 1.0, "speaker": "A", "text": "hi"}]

    def _check_dialog(turns, *, language="en-US", **_):
        captured["checked_with_language"] = language
        captured["checked_turns"] = len(turns)
        return [_Rep("A", "hi", corrections=[_Corr()])]

    def _add_llm_rewrites(reports, *, model, host, **_):
        captured["llm_called"] = (model, host)
        return [_Rep(r.speaker, r.text, corrections=r.corrections, rewrite="hi.") for r in reports]

    def _per_speaker_stats(reports):
        return [_Stat("A", correction_count=sum(len(r.corrections) for r in reports))]

    def _save_report(reports, stats, out_path, *, title=None, names=None):
        from pathlib import Path as _P

        p = _P(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {title or 'x'}\n", encoding="utf-8")
        captured["saved_to"] = p
        captured["saved_title"] = title
        captured["saved_names"] = names
        return p

    fake.load_dialog = _load_dialog  # type: ignore[attr-defined]
    fake.check_dialog = _check_dialog  # type: ignore[attr-defined]
    fake.add_llm_rewrites = _add_llm_rewrites  # type: ignore[attr-defined]
    fake.per_speaker_stats = _per_speaker_stats  # type: ignore[attr-defined]
    fake.save_report = _save_report  # type: ignore[attr-defined]
    return fake


def test_coach_writes_report_default_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa

    captured: dict = {}
    fake = _stub_coach_module(captured)
    monkeypatch.setitem(sys.modules, "encocoa.coach", fake)
    monkeypatch.setattr(encocoa, "coach", fake, raising=False)

    dialog = tmp_path / "session.dialog.json"
    dialog.write_text('[{"start":0,"end":1,"speaker":"A","text":"hi"}]', encoding="utf-8")

    rc = main(["coach", str(dialog)])
    out = capsys.readouterr().out

    assert rc == 0
    assert captured["loaded"] == dialog
    assert captured["checked_with_language"] == "en-US"
    assert "llm_called" not in captured
    assert captured["saved_to"] == tmp_path / "session.report.md"
    assert captured["saved_title"] == "session"
    assert "1 correction(s)" in out


def test_coach_with_llm_flag_calls_ollama(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import encocoa

    captured: dict = {}
    fake = _stub_coach_module(captured)
    monkeypatch.setitem(sys.modules, "encocoa.coach", fake)
    monkeypatch.setattr(encocoa, "coach", fake, raising=False)

    dialog = tmp_path / "x.dialog.json"
    dialog.write_text('[{"start":0,"end":1,"speaker":"A","text":"hi"}]', encoding="utf-8")
    out_path = tmp_path / "custom.md"

    rc = main(
        [
            "coach",
            str(dialog),
            "--out",
            str(out_path),
            "--language",
            "en-GB",
            "--llm",
            "--ollama-model",
            "llama3.2:3b",
            "--ollama-host",
            "http://example:1234",
        ]
    )

    assert rc == 0
    assert captured["checked_with_language"] == "en-GB"
    assert captured["llm_called"] == ("llama3.2:3b", "http://example:1234")
    assert captured["saved_to"] == out_path


def test_coach_languagetool_unavailable_returns_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa

    captured: dict = {}
    fake = _stub_coach_module(captured)

    def _check_dialog(turns, **_):
        raise RuntimeError("no java runtime")

    fake.check_dialog = _check_dialog  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "encocoa.coach", fake)
    monkeypatch.setattr(encocoa, "coach", fake, raising=False)

    dialog = tmp_path / "x.dialog.json"
    dialog.write_text('[{"start":0,"end":1,"speaker":"A","text":"hi"}]', encoding="utf-8")

    rc = main(["coach", str(dialog)])
    err = capsys.readouterr().err

    assert rc == 4
    assert "LanguageTool unavailable" in err
    assert "no java runtime" in err


def test_coach_invalid_json_returns_error(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    dialog = tmp_path / "bad.dialog.json"
    dialog.write_text("{not json", encoding="utf-8")
    rc = main(["coach", str(dialog)])
    assert rc == 2
    assert "could not parse dialog" in capsys.readouterr().err


def _stub_audio_module(captured: dict) -> types.ModuleType:
    fake = types.ModuleType("encocoa.audio")

    def _record_wav(out_path, duration, samplerate=16000, device=None, **_):
        from pathlib import Path as _P

        p = _P(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0")
        captured["recorded"] = {
            "out_path": p,
            "duration": duration,
            "samplerate": samplerate,
            "device": device,
        }
        return p

    fake.record_wav = _record_wav  # type: ignore[attr-defined]
    fake.print_input_devices = lambda: None  # type: ignore[attr-defined]
    fake.resolve_input_device = lambda spec: spec if isinstance(spec, int) else None  # type: ignore[attr-defined]
    return fake


def test_run_end_to_end_with_stubs(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa
    from encocoa import cli as cli_module

    captured: dict = {}

    fake_audio = _stub_audio_module(captured)
    fake_asr = _stub_asr_module(tmp_path, captured)

    fake_diarize = types.ModuleType("encocoa.diarize")

    class _DSeg:
        def __init__(self, s, e, sp):
            self.start, self.end, self.speaker = s, e, sp

        def to_dict(self):
            return {"start": self.start, "end": self.end, "speaker": self.speaker}

    fake_diarize.diarize_simple = lambda **kw: [_DSeg(0.0, 1.0, "A"), _DSeg(1.0, 2.0, "B")]  # type: ignore[attr-defined]
    fake_diarize.diarize_pyannote = lambda **_: (_ for _ in ()).throw(AssertionError("simple expected"))  # type: ignore[attr-defined]
    fake_diarize.save_diarization = lambda segs, p: Path(p).write_text("[]\n", encoding="utf-8")  # type: ignore[attr-defined]
    fake_diarize.diarization_summary = lambda segs: {"A": 1.0, "B": 1.0}  # type: ignore[attr-defined]

    fake_merge = types.ModuleType("encocoa.merge")

    class _Turn:
        def __init__(self, s, e, sp, t):
            self.start, self.end, self.speaker, self.text = s, e, sp, t

        def to_dict(self):
            return {"start": self.start, "end": self.end, "speaker": self.speaker, "text": self.text}

    fake_merge.merge = lambda a, d: [_Turn(0.0, 1.0, "A", "hi"), _Turn(1.0, 2.0, "B", "there")]  # type: ignore[attr-defined]
    fake_merge.coalesce_consecutive = lambda turns: turns  # type: ignore[attr-defined]
    fake_merge.save_dialog = lambda turns, p: Path(p).write_text(
        '[{"start":0,"end":1,"speaker":"A","text":"hi"}]', encoding="utf-8"
    )  # type: ignore[attr-defined]

    fake_report = types.ModuleType("encocoa.report")
    fake_report.save_markdown = lambda turns, p, *, title=None, names=None: Path(p).write_text(  # type: ignore[attr-defined]
        f"# {title or 'x'}\n", encoding="utf-8"
    )

    fake_coach = _stub_coach_module(captured)

    monkeypatch.setitem(sys.modules, "encocoa.audio", fake_audio)
    monkeypatch.setitem(sys.modules, "encocoa.asr", fake_asr)
    monkeypatch.setitem(sys.modules, "encocoa.diarize", fake_diarize)
    monkeypatch.setitem(sys.modules, "encocoa.merge", fake_merge)
    monkeypatch.setitem(sys.modules, "encocoa.report", fake_report)
    monkeypatch.setitem(sys.modules, "encocoa.coach", fake_coach)
    monkeypatch.setattr(encocoa, "audio", fake_audio, raising=False)
    monkeypatch.setattr(encocoa, "asr", fake_asr, raising=False)
    monkeypatch.setattr(encocoa, "diarize", fake_diarize, raising=False)
    monkeypatch.setattr(encocoa, "merge", fake_merge, raising=False)
    monkeypatch.setattr(encocoa, "report", fake_report, raising=False)
    monkeypatch.setattr(encocoa, "coach", fake_coach, raising=False)
    _install_wav_utils_stub(monkeypatch)

    opened: list[Path] = []
    monkeypatch.setattr(cli_module, "_open_path", lambda p: opened.append(p))

    rc = main(
        [
            "run",
            "--out-dir",
            str(tmp_path),
            "--name",
            "demo",
            "--duration",
            "1",
            "--model",
            "tiny.en",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert (tmp_path / "demo.wav").exists()
    assert (tmp_path / "demo.transcript.json").exists()
    assert (tmp_path / "demo.diarization.json").exists()
    assert (tmp_path / "demo.dialog.json").exists()
    assert (tmp_path / "demo.transcript.md").exists()
    assert (tmp_path / "demo.report.md").exists()
    assert captured["recorded"]["duration"] == 1
    assert captured["recorded"]["out_path"] == tmp_path / "demo.wav"
    assert captured["model_name"] == "tiny.en"
    assert opened == [tmp_path / "demo.report.md"]
    assert "Final report" in out
    assert str(tmp_path / "demo.report.md") in out


def test_run_no_coach_stops_after_transcript(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa
    from encocoa import cli as cli_module

    captured: dict = {}

    fake_audio = _stub_audio_module(captured)
    fake_asr = _stub_asr_module(tmp_path, captured)

    fake_diarize = types.ModuleType("encocoa.diarize")

    class _DSeg:
        def __init__(self, s, e, sp):
            self.start, self.end, self.speaker = s, e, sp

        def to_dict(self):
            return {"start": self.start, "end": self.end, "speaker": self.speaker}

    fake_diarize.diarize_simple = lambda **_: [_DSeg(0.0, 1.0, "A")]  # type: ignore[attr-defined]
    fake_diarize.diarize_pyannote = lambda **_: []  # type: ignore[attr-defined]
    fake_diarize.save_diarization = lambda segs, p: Path(p).write_text("[]\n", encoding="utf-8")  # type: ignore[attr-defined]
    fake_diarize.diarization_summary = lambda segs: {"A": 1.0}  # type: ignore[attr-defined]

    fake_merge = types.ModuleType("encocoa.merge")
    fake_merge.merge = lambda a, d: []  # type: ignore[attr-defined]
    fake_merge.coalesce_consecutive = lambda t: t  # type: ignore[attr-defined]
    fake_merge.save_dialog = lambda turns, p: Path(p).write_text("[]\n", encoding="utf-8")  # type: ignore[attr-defined]

    fake_report = types.ModuleType("encocoa.report")
    fake_report.save_markdown = lambda turns, p, *, title=None, names=None: Path(p).write_text(  # type: ignore[attr-defined]
        f"# {title or 'x'}\n", encoding="utf-8"
    )

    def _coach_should_not_run(*a, **kw):
        raise AssertionError("coach must not be called when --no-coach is set")

    fake_coach = types.ModuleType("encocoa.coach")
    fake_coach.load_dialog = _coach_should_not_run  # type: ignore[attr-defined]
    fake_coach.check_dialog = _coach_should_not_run  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "encocoa.audio", fake_audio)
    monkeypatch.setitem(sys.modules, "encocoa.asr", fake_asr)
    monkeypatch.setitem(sys.modules, "encocoa.diarize", fake_diarize)
    monkeypatch.setitem(sys.modules, "encocoa.merge", fake_merge)
    monkeypatch.setitem(sys.modules, "encocoa.report", fake_report)
    monkeypatch.setitem(sys.modules, "encocoa.coach", fake_coach)
    monkeypatch.setattr(encocoa, "audio", fake_audio, raising=False)
    monkeypatch.setattr(encocoa, "asr", fake_asr, raising=False)
    monkeypatch.setattr(encocoa, "diarize", fake_diarize, raising=False)
    monkeypatch.setattr(encocoa, "merge", fake_merge, raising=False)
    monkeypatch.setattr(encocoa, "report", fake_report, raising=False)
    monkeypatch.setattr(encocoa, "coach", fake_coach, raising=False)
    _install_wav_utils_stub(monkeypatch)

    opened: list[Path] = []
    monkeypatch.setattr(cli_module, "_open_path", lambda p: opened.append(p))

    rc = main(
        [
            "run",
            "--out-dir",
            str(tmp_path),
            "--name",
            "x",
            "--duration",
            "1",
            "--no-coach",
            "--no-open",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert (tmp_path / "x.transcript.md").exists()
    assert not (tmp_path / "x.report.md").exists()
    assert opened == []
    assert "Final report" in out
    assert str(tmp_path / "x.transcript.md") in out


def test_run_invalid_duration_returns_error(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad --duration in the record stage should bubble up and abort run."""
    rc = main(
        [
            "run",
            "--out-dir",
            str(tmp_path),
            "--duration",
            "0",
            "--no-open",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "duration" in err


def test_parse_names_helper() -> None:
    from encocoa.cli import _parse_names

    assert _parse_names(None) is None
    assert _parse_names("") is None
    assert _parse_names("   ") is None
    assert _parse_names("Italo") == {"A": "Italo"}
    assert _parse_names("Italo,Maria") == {"A": "Italo", "B": "Maria"}
    assert _parse_names(" Italo , Maria , Pedro ") == {
        "A": "Italo",
        "B": "Maria",
        "C": "Pedro",
    }
    # Empty fragments are skipped (they don't shift later names).
    assert _parse_names("Italo,,Maria") == {"A": "Italo", "B": "Maria"}


def test_process_names_passed_to_save_markdown(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import encocoa

    captured: dict = {}
    fake_asr = _stub_asr_module(tmp_path, captured)

    fake_diarize = types.ModuleType("encocoa.diarize")

    class _DSeg:
        def __init__(self, s, e, sp):
            self.start, self.end, self.speaker = s, e, sp

        def to_dict(self):
            return {"start": self.start, "end": self.end, "speaker": self.speaker}

    fake_diarize.diarize_simple = lambda **_: [_DSeg(0.0, 1.0, "A"), _DSeg(1.0, 2.0, "B")]  # type: ignore[attr-defined]
    fake_diarize.diarize_pyannote = lambda **_: []  # type: ignore[attr-defined]
    fake_diarize.save_diarization = lambda segs, p: Path(p).write_text("[]\n", encoding="utf-8")  # type: ignore[attr-defined]
    fake_diarize.diarization_summary = lambda segs: {"A": 1.0, "B": 1.0}  # type: ignore[attr-defined]

    fake_merge = types.ModuleType("encocoa.merge")

    class _Turn:
        def __init__(self, s, e, sp, t):
            self.start, self.end, self.speaker, self.text = s, e, sp, t

        def to_dict(self):
            return {"start": self.start, "end": self.end, "speaker": self.speaker, "text": self.text}

    fake_merge.merge = lambda a, d: [_Turn(0.0, 1.0, "A", "hi"), _Turn(1.0, 2.0, "B", "yo")]  # type: ignore[attr-defined]
    fake_merge.coalesce_consecutive = lambda t: t  # type: ignore[attr-defined]
    fake_merge.save_dialog = lambda turns, p: Path(p).write_text("[]\n", encoding="utf-8")  # type: ignore[attr-defined]

    fake_report = types.ModuleType("encocoa.report")

    def _save_md(turns, p, *, title=None, names=None):
        Path(p).write_text("ok\n", encoding="utf-8")
        captured["names_passed"] = names
        return Path(p)

    fake_report.save_markdown = _save_md  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "encocoa.asr", fake_asr)
    monkeypatch.setitem(sys.modules, "encocoa.diarize", fake_diarize)
    monkeypatch.setitem(sys.modules, "encocoa.merge", fake_merge)
    monkeypatch.setitem(sys.modules, "encocoa.report", fake_report)
    monkeypatch.setattr(encocoa, "asr", fake_asr, raising=False)
    monkeypatch.setattr(encocoa, "diarize", fake_diarize, raising=False)
    monkeypatch.setattr(encocoa, "merge", fake_merge, raising=False)
    monkeypatch.setattr(encocoa, "report", fake_report, raising=False)
    _install_wav_utils_stub(monkeypatch)

    wav = tmp_path / "session.wav"
    wav.write_bytes(b"\0")

    rc = main(["process", str(wav), "--names", "Italo,Maria"])
    assert rc == 0
    assert captured["names_passed"] == {"A": "Italo", "B": "Maria"}


def test_coach_names_flag_passed_to_save_report(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import encocoa

    captured: dict = {}
    fake = _stub_coach_module(captured)
    monkeypatch.setitem(sys.modules, "encocoa.coach", fake)
    monkeypatch.setattr(encocoa, "coach", fake, raising=False)

    dialog = tmp_path / "x.dialog.json"
    dialog.write_text('[{"start":0,"end":1,"speaker":"A","text":"hi"}]', encoding="utf-8")

    rc = main(["coach", str(dialog), "--names", "Italo,Maria"])
    assert rc == 0
    assert captured["saved_names"] == {"A": "Italo", "B": "Maria"}


def test_process_silent_audio_emits_warning(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa

    captured: dict = {}
    fake_asr = _stub_asr_module(tmp_path, captured)
    monkeypatch.setitem(sys.modules, "encocoa.asr", fake_asr)
    monkeypatch.setattr(encocoa, "asr", fake_asr, raising=False)

    # Stub wav_utils to report effective silence.
    fake_wav = types.ModuleType("encocoa.wav_utils")
    fake_wav.SILENT_THRESHOLD_DBFS = -50.0  # type: ignore[attr-defined]
    fake_wav.peak_dbfs = lambda _p: -80.0  # type: ignore[attr-defined]
    fake_wav.is_effectively_silent = lambda _p, **_: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "encocoa.wav_utils", fake_wav)
    monkeypatch.setattr(encocoa, "wav_utils", fake_wav, raising=False)

    wav = tmp_path / "session.wav"
    wav.write_bytes(b"\0")

    rc = main(["process", str(wav), "--no-diarize"])
    err = capsys.readouterr().err

    assert rc == 0  # warning, not error
    assert "effectively silent" in err
    assert "-80.0 dBFS" in err


def test_record_resolves_named_device(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--device 'mic' should be resolved to an integer index via audio.resolve_input_device."""
    import encocoa

    captured: dict = {}
    fake = _stub_audio_module(captured)

    def _resolve(spec):
        captured["resolve_called_with"] = spec
        return 7

    fake.resolve_input_device = _resolve  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "encocoa.audio", fake)
    monkeypatch.setattr(encocoa, "audio", fake, raising=False)

    rc = main(
        [
            "record",
            "--device",
            "USB Mic",
            "--duration",
            "1",
            "--out",
            str(tmp_path / "out.wav"),
        ]
    )
    assert rc == 0
    assert captured["resolve_called_with"] == "USB Mic"
    assert captured["recorded"]["device"] == 7


def test_record_unknown_named_device_returns_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa

    captured: dict = {}
    fake = _stub_audio_module(captured)

    def _resolve(_spec):
        raise ValueError("No input device matches 'nonexistent'. Available: (none)")

    fake.resolve_input_device = _resolve  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "encocoa.audio", fake)
    monkeypatch.setattr(encocoa, "audio", fake, raising=False)

    rc = main(
        [
            "record",
            "--device",
            "nonexistent",
            "--duration",
            "1",
            "--out",
            str(tmp_path / "out.wav"),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "nonexistent" in err


def test_record_mic_busy_returns_friendly_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import encocoa

    captured: dict = {}
    fake = _stub_audio_module(captured)

    def _busy(*_a, **_kw):
        raise OSError("Device unavailable")

    fake.record_wav = _busy  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "encocoa.audio", fake)
    monkeypatch.setattr(encocoa, "audio", fake, raising=False)

    rc = main(["record", "--duration", "1", "--out", str(tmp_path / "out.wav")])
    err = capsys.readouterr().err
    assert rc == 4
    assert "could not open audio input" in err
    assert "Device unavailable" in err
