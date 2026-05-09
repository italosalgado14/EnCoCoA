from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from . import __version__


def _parse_names(spec: str | None) -> dict[str, str] | None:
    """Parse a comma-separated --names value into a {canonical → display} map.

    `--names "Italo,Maria"` → `{"A": "Italo", "B": "Maria"}`. Empty fragments
    are skipped. Returns None if no names are provided.
    """
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return None
    return {chr(ord("A") + i): name for i, name in enumerate(parts)}


def _cmd_record(args: argparse.Namespace) -> int:
    try:
        from . import audio
    except OSError as e:
        # sounddevice raises OSError at import time when PortAudio is missing.
        print(
            f"encocoa record: audio backend unavailable ({e}).\n"
            "Install the PortAudio system library, e.g.:\n"
            "  Debian/Ubuntu:  sudo apt install libportaudio2\n"
            "  Fedora:         sudo dnf install portaudio\n"
            "  macOS (brew):   brew install portaudio",
            file=sys.stderr,
        )
        return 3

    if args.list_devices:
        audio.print_input_devices()
        return 0

    if args.vad_trim:
        print(
            "encocoa record: --vad-trim is not implemented yet "
            "(planned alongside diarization in a later stage).",
            file=sys.stderr,
        )
        return 2

    if args.duration <= 0:
        print("encocoa record: --duration must be > 0.", file=sys.stderr)
        return 2

    try:
        device_index = audio.resolve_input_device(args.device)
    except ValueError as e:
        print(f"encocoa record: {e}", file=sys.stderr)
        return 2

    try:
        audio.record_wav(
            out_path=args.out,
            duration=args.duration,
            samplerate=args.samplerate,
            device=device_index,
        )
    except OSError as e:
        # PortAudio raises sounddevice.PortAudioError (a subclass of OSError)
        # when the mic is busy, the device is invalid, or the sample rate is
        # unsupported.
        print(
            f"encocoa record: could not open audio input ({e}). "
            "Is the microphone in use by another app, or is the device index/name correct?",
            file=sys.stderr,
        )
        return 4
    print(f"encocoa record: wrote {args.out}")
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    from . import asr

    wav = Path(args.wav)
    if not wav.exists():
        print(f"encocoa process: input WAV not found: {wav}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else wav.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = wav.stem
    transcript_path = out_dir / f"{stem}.transcript.json"
    diarization_path = out_dir / f"{stem}.diarization.json"
    dialog_path = out_dir / f"{stem}.dialog.json"
    md_path = out_dir / f"{stem}.transcript.md"

    print(
        f"encocoa process: model={args.model}  device={args.device}  "
        f"compute={args.compute_type}  cache={asr.DEFAULT_CACHE_DIR}",
        file=sys.stderr,
    )

    from . import wav_utils

    try:
        peak = wav_utils.peak_dbfs(wav)
    except (OSError, RuntimeError) as e:
        print(f"encocoa process: could not read {wav}: {e}", file=sys.stderr)
        return 2
    if peak < wav_utils.SILENT_THRESHOLD_DBFS:
        peak_str = "-inf" if peak == float("-inf") else f"{peak:.1f}"
        print(
            f"encocoa process: warning — input WAV peaks at {peak_str} dBFS "
            f"(threshold {wav_utils.SILENT_THRESHOLD_DBFS:.1f} dBFS); "
            "audio is effectively silent, ASR is likely to produce nothing.",
            file=sys.stderr,
        )

    language = None if args.language.lower() in ("auto", "none", "") else args.language
    asr_segments, stats = asr.transcribe(
        wav_path=wav,
        model_name=args.model,
        model_dir=args.model_dir or asr.DEFAULT_CACHE_DIR,
        language=language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
    )
    asr.save_transcript(asr_segments, transcript_path)
    print(f"encocoa process: wrote {transcript_path}", file=sys.stderr)
    print(f"encocoa process: {asr.format_stats(stats)}", file=sys.stderr)

    if args.no_diarize:
        print(
            "encocoa process: --no-diarize set; skipping diarization, merge, and report.",
            file=sys.stderr,
        )
        print(f"encocoa process: done (transcript-only). Output: {transcript_path}")
        return 0

    from . import diarize as dz
    from . import merge as mg
    from . import report as rp

    print(
        f"encocoa process: diarization (backend={args.diarizer}, num_speakers={args.num_speakers}) ...",
        file=sys.stderr,
    )
    try:
        if args.diarizer == "pyannote":
            diar_segments = dz.diarize_pyannote(
                wav_path=wav,
                num_speakers=args.num_speakers,
                hf_token=args.hf_token,
                device=args.device,
            )
        else:
            diar_segments = dz.diarize_simple(
                wav_path=wav,
                num_speakers=args.num_speakers,
            )
    except RuntimeError as e:
        print(f"encocoa process: diarization failed: {e}", file=sys.stderr)
        return 4

    dz.save_diarization(diar_segments, diarization_path)
    summary = dz.diarization_summary(diar_segments)
    summary_str = ", ".join(f"{k}={v:.1f}s" for k, v in sorted(summary.items()))
    print(
        f"encocoa process: wrote {diarization_path} "
        f"({len(diar_segments)} segments; {summary_str})",
        file=sys.stderr,
    )

    turns = mg.coalesce_consecutive(mg.merge(asr_segments, diar_segments))
    mg.save_dialog(turns, dialog_path)
    print(
        f"encocoa process: wrote {dialog_path} ({len(turns)} dialog turns)",
        file=sys.stderr,
    )

    names = _parse_names(getattr(args, "names", None))
    rp.save_markdown(turns, md_path, title=stem, names=names)
    print(f"encocoa process: wrote {md_path}", file=sys.stderr)

    print(f"encocoa process: done. Outputs in {out_dir}/")
    return 0


def _cmd_coach(args: argparse.Namespace) -> int:
    from . import coach as ch

    dialog_path = Path(args.dialog)
    if not dialog_path.exists():
        print(f"encocoa coach: input dialog not found: {dialog_path}", file=sys.stderr)
        return 2

    stem = dialog_path.stem
    if stem.endswith(".dialog"):
        stem = stem[: -len(".dialog")]
    out_path = Path(args.out) if args.out else dialog_path.parent / f"{stem}.report.md"

    try:
        turns = ch.load_dialog(dialog_path)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"encocoa coach: could not parse dialog: {e}", file=sys.stderr)
        return 2

    print(
        f"encocoa coach: running LanguageTool on {len(turns)} turn(s) "
        f"(language={args.language}) ...",
        file=sys.stderr,
    )
    try:
        reports = ch.check_dialog(turns, language=args.language)
    except RuntimeError as e:
        print(f"encocoa coach: LanguageTool unavailable: {e}", file=sys.stderr)
        return 4

    if args.llm:
        print(
            f"encocoa coach: requesting Ollama rewrites "
            f"(model={args.ollama_model}, host={args.ollama_host}) ...",
            file=sys.stderr,
        )
        reports = ch.add_llm_rewrites(
            reports, model=args.ollama_model, host=args.ollama_host
        )

    stats = ch.per_speaker_stats(reports)
    names = _parse_names(getattr(args, "names", None))
    ch.save_report(reports, stats, out_path, title=stem, names=names)

    total_corrections = sum(s.correction_count for s in stats)
    print(
        f"encocoa coach: wrote {out_path} "
        f"({total_corrections} correction(s) across {len(stats)} speaker(s))"
    )
    return 0


def _open_path(path: Path) -> None:
    """Best-effort: open a file with the OS's default handler. Failures are non-fatal."""
    p = str(path)
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(
                ["open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        elif system == "Windows":
            os.startfile(p)  # type: ignore[attr-defined]
        else:
            opener = shutil.which("xdg-open")
            if opener is None:
                print(
                    f"encocoa run: install xdg-open to auto-open files. Final report: {p}",
                    file=sys.stderr,
                )
                return
            subprocess.Popen(
                [opener, p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except OSError as e:
        print(f"encocoa run: could not auto-open {p}: {e}", file=sys.stderr)


def _cmd_run(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.name or f"session-{timestamp}"
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = out_dir / f"{name}.wav"
    dialog_path = out_dir / f"{name}.dialog.json"
    transcript_md_path = out_dir / f"{name}.transcript.md"
    report_md_path = out_dir / f"{name}.report.md"

    print(f"encocoa run: session '{name}' → {out_dir}/", file=sys.stderr)

    rec_args = argparse.Namespace(
        duration=args.duration,
        out=wav_path,
        device=args.device,
        samplerate=args.samplerate,
        list_devices=False,
        vad_trim=False,
    )
    rc = _cmd_record(rec_args)
    if rc != 0:
        return rc

    proc_args = argparse.Namespace(
        wav=wav_path,
        model=args.model,
        out_dir=out_dir,
        language=args.language,
        device=args.asr_device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        model_dir=None,
        diarizer=args.diarizer,
        num_speakers=args.num_speakers,
        hf_token=args.hf_token,
        no_diarize=False,
        names=args.names,
    )
    rc = _cmd_process(proc_args)
    if rc != 0:
        return rc

    if args.no_coach:
        print(
            "encocoa run: --no-coach set; stopping after transcript.", file=sys.stderr
        )
        final_artifact = transcript_md_path
    else:
        coach_args = argparse.Namespace(
            dialog=dialog_path,
            out=report_md_path,
            language=args.coach_language,
            llm=args.llm,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            names=args.names,
        )
        rc = _cmd_coach(coach_args)
        if rc != 0:
            return rc
        final_artifact = report_md_path

    if not args.no_open:
        _open_path(final_artifact)

    print(f"encocoa run: done. Final report: {final_artifact}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="encocoa",
        description=(
            "EnCoCoA — local English conversation coach for two speakers. "
            "Record a conversation, transcribe it, separate the speakers, "
            "and produce correction suggestions."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        required=True,
    )

    p_record = subparsers.add_parser(
        "record",
        help="Capture audio from the microphone to a WAV file.",
    )
    p_record.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Recording length in seconds (default: 600 = 10 minutes).",
    )
    p_record.add_argument(
        "--out",
        type=Path,
        default=Path("session.wav"),
        help="Output WAV file path (default: ./session.wav).",
    )
    p_record.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Input device — integer index or a case-insensitive name substring "
            "(see --list-devices). Default: system default input."
        ),
    )
    p_record.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000).",
    )
    p_record.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit.",
    )
    p_record.add_argument(
        "--vad-trim",
        action="store_true",
        help="Trim leading/trailing silence with VAD (deferred to a later stage).",
    )
    p_record.set_defaults(func=_cmd_record)

    p_process = subparsers.add_parser(
        "process",
        help="Transcribe and diarize a WAV file; produce dialog JSON and a Markdown transcript.",
    )
    p_process.add_argument(
        "wav",
        type=Path,
        help="Input WAV file to process.",
    )
    p_process.add_argument(
        "--model",
        default="small.en",
        help="faster-whisper model name (e.g. tiny.en, base.en, small.en, medium.en). Default: small.en.",
    )
    p_process.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: same directory as the input WAV).",
    )
    p_process.add_argument(
        "--language",
        default="en",
        help="Language code (default: en). Pass 'auto' to let the model detect.",
    )
    p_process.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Inference device (default: cpu).",
    )
    p_process.add_argument(
        "--compute-type",
        default="int8",
        help="ctranslate2 compute type (default: int8 — fast on CPU; use float16 on GPU).",
    )
    p_process.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Decoding beam size (default: 5).",
    )
    p_process.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=f"Local directory for ASR model files (default: {Path.home() / '.cache' / 'encocoa' / 'models'}).",
    )
    p_process.add_argument(
        "--diarizer",
        choices=("simple", "pyannote"),
        default="simple",
        help=(
            "Diarization backend. "
            "'simple' = Resemblyzer + KMeans (no HF token, default). "
            "'pyannote' = pyannote.audio 3.1 (more accurate; needs HF token + `uv add pyannote.audio`)."
        ),
    )
    p_process.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Expected number of speakers (default: 2).",
    )
    p_process.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for the pyannote backend (or set HF_TOKEN env var).",
    )
    p_process.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip diarization and merge; produce only the ASR transcript JSON.",
    )
    p_process.add_argument(
        "--names",
        default=None,
        help=(
            "Comma-separated display names for the speakers, in pipeline order "
            '(e.g. "Italo,Maria"). Replaces "Speaker A"/"Speaker B" in the '
            "Markdown transcript."
        ),
    )
    p_process.set_defaults(func=_cmd_process)

    p_coach = subparsers.add_parser(
        "coach",
        help="Generate English-correction suggestions from a dialog JSON.",
    )
    p_coach.add_argument(
        "dialog",
        type=Path,
        help="Input dialog JSON (produced by `encocoa process`).",
    )
    p_coach.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Markdown report path (default: <dialog-stem>.report.md beside the input).",
    )
    p_coach.add_argument(
        "--language",
        default="en-US",
        help="LanguageTool language code (default: en-US).",
    )
    p_coach.add_argument(
        "--llm",
        action="store_true",
        help="Add a phrasing rewrite per utterance via a local Ollama server (off by default).",
    )
    p_coach.add_argument(
        "--ollama-model",
        default="phi3:mini",
        help="Ollama model used when --llm is set (default: phi3:mini).",
    )
    p_coach.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    p_coach.add_argument(
        "--names",
        default=None,
        help=(
            'Comma-separated display names for the speakers (e.g. "Italo,Maria"). '
            "Replaces \"Speaker A\"/\"Speaker B\" in the report headings."
        ),
    )
    p_coach.set_defaults(func=_cmd_coach)

    p_run = subparsers.add_parser(
        "run",
        help="End-to-end: record, process, and coach in one command.",
        description=(
            "Record a conversation, transcribe + diarize it, generate corrections, "
            "and (by default) open the final Markdown report. Designed to work with "
            "no flags on a fresh machine."
        ),
    )
    p_run.add_argument(
        "--out-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for session artifacts (default: current directory).",
    )
    p_run.add_argument(
        "--name",
        default=None,
        help="Basename for session files (default: session-YYYYMMDD-HHMMSS).",
    )
    p_run.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Recording length in seconds (default: 600 = 10 minutes).",
    )
    p_run.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Input device — integer index or a case-insensitive name substring "
            "(use `encocoa record --list-devices` to see options)."
        ),
    )
    p_run.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Recording sample rate in Hz (default: 16000).",
    )
    p_run.add_argument(
        "--model",
        default="small.en",
        help="faster-whisper model name (default: small.en).",
    )
    p_run.add_argument(
        "--language",
        default="en",
        help="ASR language code (default: en; pass 'auto' to detect).",
    )
    p_run.add_argument(
        "--asr-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="ASR inference device (default: cpu).",
    )
    p_run.add_argument(
        "--compute-type",
        default="int8",
        help="ctranslate2 compute type (default: int8).",
    )
    p_run.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="ASR decoding beam size (default: 5).",
    )
    p_run.add_argument(
        "--diarizer",
        choices=("simple", "pyannote"),
        default="simple",
        help="Diarization backend (default: simple — no HF token required).",
    )
    p_run.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Expected number of speakers (default: 2).",
    )
    p_run.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (required only when --diarizer=pyannote).",
    )
    p_run.add_argument(
        "--coach-language",
        default="en-US",
        help="LanguageTool language code for the coach (default: en-US).",
    )
    p_run.add_argument(
        "--llm",
        action="store_true",
        help="Add per-utterance phrasing rewrites via a local Ollama server.",
    )
    p_run.add_argument(
        "--ollama-model",
        default="phi3:mini",
        help="Ollama model used when --llm is set (default: phi3:mini).",
    )
    p_run.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    p_run.add_argument(
        "--names",
        default=None,
        help=(
            'Comma-separated display names for the speakers (e.g. "Italo,Maria"). '
            "Used in both the transcript and the corrections report."
        ),
    )
    p_run.add_argument(
        "--no-coach",
        action="store_true",
        help="Stop after transcription/diarization; skip the coach step.",
    )
    p_run.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the final report in the OS default viewer.",
    )
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
