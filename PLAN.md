# EnCoCoA — Implementation Plan

**EnCoCoA**: English Conversation Coach Assistant.
A local, lightweight desktop app that records a 2-person English conversation from a single microphone, separates the two speakers, transcribes the dialogue, and produces correction suggestions to improve the speakers' English.

---

## 1. Goals & Constraints

**Functional goals**
- Capture a single microphone stream containing two speakers.
- Automatically attribute each utterance to **Speaker A** or **Speaker B** (speaker diarization, fixed `n=2`).
- Transcribe the conversation to text.
- After the session ends (~10 min target), analyze each speaker's utterances and produce grammar / vocabulary / phrasing suggestions.
- Persist the conversation and the corrections to a file (Markdown).

**Non-functional constraints (from `INSTRUCTIONS.md`)**
- **Local-first**: everything must run offline on the user's machine.
- **Cheap**: prefer free, open-source models over cloud APIs.
- **Lightweight**: prefer small models / CPU-friendly inference. GPU is a bonus, not a requirement.
- Cloud / external APIs are **only a last resort** if a local component proves unworkable.

**Explicit non-goals (for v1)**
- No real-time UI of who is speaking — batch processing after the recording is fine.
- No multi-language support — English only.
- No mobile / web — desktop CLI first.
- No cloud sync, accounts, or multi-user storage.

---

## 2. High-level Architecture

```
┌──────────────┐   wav    ┌────────────────┐   segments   ┌────────────────┐
│ Audio capture├─────────►│  Diarization   ├─────────────►│      ASR       │
│   (mic)      │  16kHz   │  (2 speakers)  │ (start,end,  │  (whisper)     │
└──────────────┘  mono    └────────────────┘  speaker)    └───────┬────────┘
                                                                  │ text per segment
                                                                  ▼
                                                         ┌────────────────┐
                                                         │   Aggregator   │
                                                         │ (merge & sort) │
                                                         └───────┬────────┘
                                                                 │
                                              ┌──────────────────┴──────────────────┐
                                              ▼                                     ▼
                                     ┌────────────────┐                    ┌─────────────────┐
                                     │  Transcript.md │                    │ English Coach   │
                                     │  (raw dialog)  │                    │ (corrections)   │
                                     └────────────────┘                    └────────┬────────┘
                                                                                    ▼
                                                                            ┌────────────────┐
                                                                            │  Report.md     │
                                                                            │  (suggestions) │
                                                                            └────────────────┘
```

The pipeline is **batch**: record → process → report. Real-time streaming is out of scope for v1; it complicates diarization and adds little value for a 10-minute practice session.

---

## 3. Technology Choices (local-first)

For each component, the **primary choice** is the lightest credible option; **fallback** is what to swap in if the primary is too inaccurate.

| Component             | Primary                                                           | Fallback                                          | Notes                                                                 |
|-----------------------|-------------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------------------------------|
| Language / runtime    | **Python 3.11+**                                                  | —                                                 | Best ML ecosystem; keeps deps small.                                  |
| Audio capture         | **`sounddevice`** + `soundfile`                                   | `pyaudio`                                         | Cross-platform, simple, writes WAV directly.                          |
| ASR (speech → text)   | **`faster-whisper`** with `small.en` or `base.en`                 | `whisper.cpp` for very low-end machines           | CTranslate2 backend, ~4× faster than openai-whisper, runs on CPU.     |
| Speaker diarization   | **`pyannote.audio` 3.x** (`speaker-diarization-3.1`)              | Embeddings (`Resemblyzer`) + KMeans `k=2`         | Pyannote needs a free HF token (one-time, local inference after).     |
| Grammar correction    | **`language_tool_python`** (LanguageTool, fully offline)          | Local LLM via **Ollama** (`phi3:mini` / `llama3.2:3b`) for richer rewrites | LanguageTool catches mechanical errors cheaply; LLM gives style/phrasing advice. |
| Voice activity        | **`silero-vad`**                                                  | Whisper's built-in VAD                            | Trims dead air before diarization to save compute.                    |
| Packaging             | **`uv`** (manages venv, lockfile, install, scripts)               | `pipx` for end-user install                       | Single tool for env + deps; reproducible via `uv.lock`.               |

**Why batch instead of streaming?** Pyannote and accurate diarization need the whole utterance in context. Streaming diarization exists but is much less accurate, and 10 minutes of audio processes in well under a minute on a modern CPU.

**Why Whisper `*.en` variants?** English-only models are ~30% smaller and slightly more accurate on English than the multilingual ones. `base.en` ≈ 74M params, `small.en` ≈ 244M; both run on CPU.

**Why pyannote vs. clustering embeddings?** Pyannote 3.1 handles overlapping speech and is much more accurate than naive KMeans on Resemblyzer embeddings. The fallback exists so users who refuse the HF gated-model click-through still have a working pipeline.

---

## 4. Staged Implementation

Each stage produces a runnable artifact. We do not move to stage *N+1* until stage *N* works end-to-end.

### Stage 0 — Project skeleton  *(½ day)*
- Initialize a `uv` project: `pyproject.toml`, `.python-version` (≥ 3.11), `uv.lock`, `.gitignore`, `src/encocoa/`, `tests/`.
- `uv` manages the virtual environment and dependencies; no `requirements.txt` and no manual `venv` activation.
- Add a single `encocoa` CLI entry point (stdlib `argparse`, no deps yet) with subcommands: `record`, `process`, `coach`, `run` (= record + process + coach). Subcommands are stubs that print "not yet implemented".
- **Exit criterion**: `uv run encocoa --help` prints usage and lists the four subcommands.

### Stage 1 — Audio capture  *(½ day)*
- `encocoa record --duration 600 --out session.wav` records 16 kHz mono WAV from the default input device.
- Add `--device` to list/pick microphones, `--vad-trim` flag (off by default).
- Visual progress in the terminal (elapsed time, simple level meter).
- **Exit criterion**: a clean WAV file is produced and plays back correctly.

### Stage 2 — ASR transcription  *(1 day)*
- `encocoa process session.wav --model small.en` runs faster-whisper and writes `session.transcript.json` with `[ {start, end, text}, ... ]`.
- Cache models under `~/.cache/encocoa/models`.
- Print word/char count and timing benchmark on completion.
- **Exit criterion**: a 2-minute hand-recorded sample transcribes with reasonable accuracy on CPU in < ~30 s.

### Stage 3 — Diarization (2 speakers)  *(1–2 days)*
- Integrate `pyannote.audio` with `num_speakers=2`.
- Document the one-time HF token setup in README.
- Produce `session.diarization.json`: `[ {start, end, speaker: "A"|"B"}, ... ]`.
- **Merge step**: align ASR segments with diarization segments by maximum time overlap, output `session.dialog.json`: `[ {start, end, speaker, text}, ... ]`.
- Render `session.transcript.md`:
  ```markdown
  **Speaker A** (00:00–00:04): Hi, how are you today?
  **Speaker B** (00:05–00:09): I am good, and you?
  ```
- Build the **Resemblyzer + KMeans fallback** behind `--diarizer simple` for users who skip pyannote.
- **Exit criterion**: on a 2-min two-person sample, ≥ 90% of words are attributed to the correct speaker.

### Stage 4 — English coach (corrections)  *(1–2 days)*
- `encocoa coach session.dialog.json --out session.report.md`.
- **Pass 1 — mechanical (LanguageTool)**: per utterance, report grammar/spelling/punctuation issues with a suggested fix and the rule name.
- **Pass 2 — phrasing (optional, behind `--llm` flag)**: send each utterance to a local Ollama model with a tight prompt: *"Rewrite this spoken English sentence to sound natural and grammatical. Keep the meaning. Return only the rewrite."* Show original vs. rewrite as a diff.
- Produce per-speaker stats: word count, top recurring errors, suggested focus areas (e.g. "verb tense agreement: 4 occurrences").
- **Exit criterion**: report contains corrections for at least mechanical errors with no LLM dependency required.

### Stage 5 — `encocoa run` end-to-end  *(½ day)*
- One command: records 10 min, processes, diarizes, transcribes, coaches, opens the final `report.md`.
- Sensible defaults so a new user can succeed without flags.
- **Exit criterion**: `encocoa run` on a fresh machine produces a usable report.

### Stage 6 — Polish & robustness  *(ongoing — round 1 done)*
- ✅ **Speaker labeling:** `--names "Italo,Maria"` on `process`/`coach`/`run` replaces `Speaker A/B` in both Markdown outputs. Names map to canonical labels in pipeline order; unmapped speakers fall back to `Speaker X`.
- ✅ **Named device picker:** `--device` on `record`/`run` accepts an integer index *or* a case-insensitive name substring (e.g. `--device "USB Mic"`). Ambiguous and unknown matches raise a friendly error.
- ✅ **Silent-audio warning:** `process` checks WAV peak amplitude before ASR and warns when the input is below `-50 dBFS`.
- ✅ **Friendlier mic-busy message:** `record` catches PortAudio `OSError` and prints "could not open audio input — is the microphone in use by another app …" with exit code 4.
- ⏳ **Speaker enrollment:** record 10 s of each voice up front → use embeddings to assign **stable** labels across sessions. Deferred to Stage 6 round 2.
- ⏳ **Config TOML:** `~/.config/encocoa/config.toml` for default flag values. Deferred to round 2.
- Optional Stage 7: minimal local Tk/PySide GUI — only after the CLI is solid.

---

## 5. Repository Layout (proposed)

```
EnCoCoA/
├── INSTRUCTIONS.md
├── PLAN.md                    # this file
├── pyproject.toml             # project + deps (managed by uv)
├── uv.lock                    # reproducible resolution
├── .python-version            # pinned interpreter version
├── .gitignore
├── src/
│   └── encocoa/
│       ├── __init__.py
│       ├── cli.py             # argparse entry point
│       ├── audio.py           # Stage 1: capture
│       ├── asr.py             # Stage 2: faster-whisper wrapper
│       ├── diarize.py         # Stage 3: pyannote + simple fallback
│       ├── merge.py           # Stage 3: align ASR + diarization
│       ├── coach.py           # Stage 4: LanguageTool + optional LLM
│       └── report.py          # Markdown rendering
└── tests/
    └── data/                  # short sample wavs for regression
```

---

## 6. Hardware Expectations & Fallbacks

- **Baseline target**: laptop with 4 CPU cores, 8 GB RAM, no GPU. Should process a 10-min session in < 2 min wall time using `base.en` + pyannote on CPU.
- **Low-end fallback**: drop to `tiny.en` and the simple diarizer; trades accuracy for speed.
- **GPU available**: auto-detect CUDA in faster-whisper and pyannote; expect ~5× speedup.
- **If pyannote is unacceptable** (license click-through, model size): the simple Resemblyzer + KMeans path is the offline-only escape hatch. We measure both on a labeled 2-speaker sample and document the gap.

---

## 7. Risks & Open Questions

| Risk                                                                 | Mitigation                                                                          |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Diarization accuracy drops on noisy/overlapping speech.              | Measure on a real 10-min sample early (end of Stage 3); document expected WER/DER.  |
| Pyannote model gating (HF) annoys users.                             | Keep the simple-diarizer fallback first-class.                                      |
| LanguageTool runs a Java process — heavy on cold start.              | Start it once per `coach` invocation; document JVM as an optional dep.              |
| Local LLM (Ollama) is too heavy on baseline hardware.                | Make `--llm` opt-in. Default coach uses LanguageTool only.                          |
| Microphone quality varies wildly; cheap mics produce poor diarization.| Document recommended setup; add an `--audio-check` command to gauge SNR.            |
| Real-time desire creeps back in.                                     | Explicitly out of scope in v1. Revisit only after Stage 6.                          |

**Open questions to resolve before Stage 1**
1. OS targets — Linux only first, or Linux + macOS + Windows from day one? (Affects audio backend testing.)
2. Are we OK with a one-time HF token setup for the better diarizer, or do we want a strictly zero-account install?
3. Does the user want speaker **identity** (Italo vs. Maria) persisted across sessions, or is per-session A/B enough for v1?

---

## 8. Definition of Done (v1)

- `encocoa run` records a 10-minute session and produces a Markdown report containing:
  - The full dialog with each turn attributed to Speaker A or B and timestamps.
  - Per-utterance mechanical corrections (LanguageTool).
  - Per-speaker summary statistics and top recurring error types.
- Everything works offline on a CPU-only laptop (with the `--llm` flag clearly marked as optional).
- A short README explains: install, mic setup, one-time model download, and the three commands (`record`, `process`, `coach`) plus the unified `run`.

---

## 9. Suggested Next Step

Confirm the three open questions in §7, then begin **Stage 0 + Stage 1** in a single small PR: project skeleton + working `encocoa record`. This gets a real WAV on disk fast and lets us validate the audio path before pulling in any ML dependencies.
