# EnCoCoA — English Conversation Coach Assistant

A local, lightweight tool for practicing English with a partner. Record a ~10-minute conversation between **two speakers** from a single microphone, then get a transcript and post-hoc grammar/usage feedback to help you improve.

No cloud APIs by default — everything runs on your machine.

## Pipeline

`record → diarize (2 speakers) → transcribe → merge → coach → Markdown report`

Built on:

- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) for ASR (`base.en` / `small.en`)
- [`Resemblyzer`](https://github.com/resemble-ai/Resemblyzer) + KMeans (`k=2`) for speaker diarization
- [`language-tool-python`](https://github.com/jxmorris12/language_tool_python) for grammar suggestions
- `sounddevice` / `soundfile` for capture

## Requirements

- Python ≥ 3.11
- [`uv`](https://github.com/astral-sh/uv) for environment and dependency management
- PortAudio (for microphone capture)
  - Debian/Ubuntu: `sudo apt install libportaudio2`
  - Fedora: `sudo dnf install portaudio`
  - macOS: `brew install portaudio`

## Install

```bash
uv sync
```

## Usage

The CLI exposes four subcommands:

```bash
uv run encocoa record  --out conv.wav --duration 600
uv run encocoa process --in  conv.wav --out conv.transcript.json --names "Alice,Bob"
uv run encocoa coach   --in  conv.transcript.json --out conv.report.md
uv run encocoa run     --out-dir ./session                       # record + process + coach
```

List available input devices:

```bash
uv run encocoa record --list-devices
```

## Project layout

```
src/encocoa/
  audio.py     # microphone capture
  diarize.py   # 2-speaker diarization
  asr.py       # faster-whisper transcription
  merge.py     # align ASR words to speaker turns
  coach.py     # grammar / usage suggestions
  report.py    # Markdown report
  cli.py       # encocoa CLI
tests/
PLAN.md        # staged implementation plan
```

## Status

Stage 1 of the staged plan in `PLAN.md` is complete. See that file for the full roadmap.

## License

See [`LICENSE`](LICENSE).