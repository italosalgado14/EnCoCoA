# EnCoCoA — Technical Documentation

**EnCoCoA** = English Conversation Coach Assistant. A local, lightweight CLI
that records a 2-speaker English practice conversation, attributes each
utterance to *Speaker A* or *Speaker B*, transcribes the dialogue, and
produces correction suggestions to help both speakers improve their English.

This document describes the **current technical design** of the pipeline,
the rationale behind each component choice, and the **implementation status**
across the seven planned stages.

---

## 1. Design constraints

EnCoCoA's design is dominated by three constraints, in priority order:

1. **Local-first.** The whole pipeline must run offline on the user's
   laptop. No cloud APIs except as a last resort (none are used today).
2. **Lightweight.** Prefer small, CPU-friendly models. Avoid heavy
   dependencies when a leaner alternative gives acceptable accuracy.
3. **Cheap.** No paid services. Open-source only. The one optional
   third-party touchpoint is HuggingFace, and only for the gated pyannote
   model — which is itself optional.

These constraints guide the choices below.

---

## 2. Pipeline architecture

```
┌──────────────┐   wav    ┌────────────────┐
│ Audio capture├─────────►│   ASR          │ faster-whisper (small.en/tiny.en)
│   (mic)      │  16 kHz  │  segments      │ → session.transcript.json
└──────────────┘  mono    └────────┬───────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │  Diarization   │ Resemblyzer + KMeans (default)
                          │  (2 speakers)  │ pyannote.audio 3.1 (optional)
                          └────────┬───────┘ → session.diarization.json
                                   │
                                   ▼
                          ┌────────────────┐
                          │ Merge (overlap)│ → session.dialog.json
                          └────────┬───────┘
                                   ▼
                          ┌────────────────┐
                          │  Report (md)   │ → session.transcript.md
                          └────────┬───────┘
                                   ▼
                          ┌────────────────┐
                          │  English Coach │ LanguageTool (mechanical)
                          │  (corrections) │ Ollama rewrite (optional, --llm)
                          └────────────────┘ → session.report.md
```

The pipeline is **batch**, not streaming. The recording finishes first; then
ASR, diarization, merge, and report run sequentially. Streaming diarization
exists in pyannote, but it is significantly less accurate, and a 10-minute
practice session processes in well under a minute on CPU. Real-time
attribution adds complexity without meaningful user value for this use case.

A single `encocoa run` command chains all of the above stages with sensible
defaults; the per-stage subcommands (`record`, `process`, `coach`) remain
available for debugging or for re-running an individual stage against an
existing artifact on disk.

### File contracts between stages

Each stage writes a JSON or Markdown artifact next to the input WAV (or in
`--out-dir`). Stages can be re-run independently because the contract is on
disk, not in memory.

| File                          | Schema                                          | Producer       |
|-------------------------------|-------------------------------------------------|----------------|
| `<stem>.transcript.json`      | `[{start, end, text}]`                          | ASR            |
| `<stem>.diarization.json`     | `[{start, end, speaker}]`                       | Diarization    |
| `<stem>.dialog.json`          | `[{start, end, speaker, text}]`                 | Merge          |
| `<stem>.transcript.md`        | Markdown — `**Speaker A** (mm:ss–mm:ss): text`  | Report         |
| `<stem>.report.md`            | Markdown corrections + per-speaker stats        | Coach          |

`start`/`end` are seconds (float, rounded to 3 decimals). `speaker` is a
canonical label (`A`, `B`, …) assigned by **first appearance in time** so
that a given session is reproducible regardless of which raw cluster IDs
the underlying model emits.

---

## 3. Component decisions

### 3.1 Language and packaging — Python 3.12 + uv

**Decision:** Python ≥ 3.11, dependencies managed by `uv`.

**Why:** Python has the strongest local-ML ecosystem (faster-whisper,
Resemblyzer, pyannote, language_tool_python, …). `uv` replaces the
traditional `venv` + `pip` + `requirements.txt` triad with a single tool
that produces a reproducible `uv.lock`. Faster than pip and with first-class
script entry points.

**How it is used:** `pyproject.toml` is the source of truth; dependencies
are added via `uv add <pkg>`. The dev environment (`pytest`) lives in a
separate group. The `encocoa` console script is wired in
`[project.scripts]`.

### 3.2 Audio capture — `sounddevice` + `soundfile`

**Decision:** `sounddevice.InputStream` with a callback that pushes
`int16` numpy chunks into a `queue.Queue`; `soundfile.SoundFile` writes
those chunks to a 16 kHz mono PCM-16 WAV.

**Why:**
- `sounddevice` is the lightest cross-platform mic-capture binding for
  Python; the alternative `pyaudio` has the same PortAudio dependency and
  a less ergonomic API.
- 16 kHz mono PCM-16 is the canonical input format for both
  faster-whisper and Resemblyzer. Recording natively at this rate avoids a
  later resample step.
- A streaming write (callback → queue → `SoundFile.write`) means a
  Ctrl+C interrupt still flushes a valid WAV to disk; nothing is lost.

**System dependency:** PortAudio (`libportaudio2`). The CLI catches the
`OSError` raised at import time and prints a per-platform install hint
(apt / dnf / brew) instead of a traceback.

**Trade-offs we accepted:**
- The `--vad-trim` flag is exposed but **not yet implemented**. Adding
  silero-vad would pull torch as a hard runtime dep on the recorder
  itself, which is wasteful for users who don't use VAD trim. The flag
  will become functional once VAD is wired in, likely in Stage 6.

### 3.3 ASR — `faster-whisper` with `small.en` (default)

**Decision:** `faster-whisper` (CTranslate2 backend) with the `small.en`
model by default; `--compute-type int8`; `--device cpu`.

**Why:**
- CTranslate2 is roughly **4× faster** than the openai-whisper Python
  package on the same model, with comparable accuracy. It also
  produces lower memory pressure, which matters on the 8 GB-RAM
  baseline target.
- `*.en` models (English-only) are ~30% smaller than the multilingual
  variants and *slightly more accurate* on English audio. EnCoCoA is
  English-only by mandate, so we never need multilingual.
- `int8` quantization makes the model fit comfortably on a CPU and
  recovers most of the latency lost by not having a GPU. On a 4-core
  CPU, `small.en` at int8 typically transcribes audio at **3–5× real
  time**.
- `download_root` is set to `~/.cache/encocoa/models/`, which keeps
  EnCoCoA's models out of the user's HuggingFace cache and makes it
  trivial to wipe state by deleting one directory.

**Streamed iteration:** `model.transcribe()` returns a generator. We
iterate it inline with a progress line that updates per percent of
audio duration consumed — the user sees text appear instead of waiting
silently.

**Trade-offs we accepted:**
- A first-time run downloads ~145 MB (int8 quantized `small.en`). This
  is the price of being local; subsequent runs are zero-network.
- Beam size 5 is a good default; lower beam sizes are faster but
  noticeably worse on disfluencies common in conversational practice.

### 3.4 Diarization — Resemblyzer + KMeans (default), pyannote optional

**Decision:** Two backends behind `--diarizer {simple,pyannote}`.

The **default** is `simple`: Resemblyzer voice embeddings + scikit-learn
KMeans clustering with `n_clusters = num_speakers` (= 2 by default).
Resemblyzer produces a partial embedding per ~0.77 s window with 50 %
overlap. Each window is assigned a cluster label, the labels are
canonicalized to `A`/`B` by **time of first appearance**, and consecutive
same-speaker windows are coalesced (gap ≤ 0.2 s).

The **opt-in** backend is `pyannote.audio` 3.1's
`speaker-diarization-3.1` pipeline — the state of the art for
2-speaker diarization with overlap support. We deliberately do not
install `pyannote.audio` by default; users who want it run
`uv add pyannote.audio` and pass an HF token (gated model).

**Why this split:**
- Plan §3 lists pyannote as the *primary* and Resemblyzer as the
  *fallback*. Two real-world frictions made us flip the default:
  1. **Install footprint.** pyannote.audio pulls torch + torchaudio +
     speechbrain + lightning + …, ~3 GB of dependencies. That
     contradicts the project's "lightweight" mandate for users who
     might be perfectly served by the simple backend.
  2. **HF gating.** pyannote's model requires accepting terms on
     HuggingFace and providing a token. That is friction on a
     first-time install and a non-zero account/privacy concern.
- Resemblyzer + KMeans **is** less accurate, especially on overlapping
  speech and short turns. But for two speakers in a clean recording
  (one mic, two voices, alternating), it is adequate to pass the
  Stage 3 exit criterion. Users who hit accuracy issues can opt into
  pyannote with one `uv add` and one env var.
- Both backends share the same `DiarizationSegment` schema and the same
  `_label_by_first_appearance` canonicalization, so the rest of the
  pipeline is backend-agnostic.

**Trade-offs we accepted:**
- The simple backend still requires torch (Resemblyzer is a torch
  model). torch on CPU is heavy, but unavoidable for any modern
  speaker-embedding network. There is no torch-free diarization path
  with comparable quality.
- Resemblyzer's windows can overlap each other in time. We keep them as
  produced; the merge step uses *maximum overlap*, which behaves
  correctly under window overlap.

### 3.5 Merge — maximum-overlap alignment

**Decision:** For each ASR segment, compute the temporal overlap with
every diarization segment and assign the speaker whose segment has the
**largest** overlap. If no diarization segment overlaps the ASR
segment, fall back to the **temporally closest** diarization segment by
midpoint distance. After alignment, **coalesce consecutive same-speaker
turns** into one turn (joining the text with a space).

**Why:**
- Word-level alignment (assigning each word to a speaker) is more
  accurate but requires word-level timestamps, which are heavier to
  produce and prone to noise on short function words. Segment-level
  alignment is much simpler and matches Stage 3's exit criterion of
  ≥ 90 % word attribution accuracy on a clean 2-min sample.
- The closest-by-midpoint fallback only fires when ASR and diarization
  disagree about whether a region contains speech (rare, but happens
  near the start/end of an utterance). Returning *something* sensible
  is better than dropping the segment.
- Coalescing consecutive same-speaker turns produces a far more readable
  Markdown report — a single "Speaker A: ..." block instead of three
  half-sentence fragments.

### 3.6 Report — Markdown by hand

**Decision:** Render Markdown directly with f-strings; no templating
engine.

**Why:** The output format is fixed and small (one line per turn plus
an optional title). Pulling in Jinja2 or similar would be over-engineering.
Timestamps are formatted as `MM:SS` (or `HH:MM:SS` past one hour).

### 3.7 English coach — LanguageTool + optional Ollama

**Decision:** Two-pass coaching behind `encocoa coach <dialog.json>`.

- **Pass 1 (always-on, offline) — LanguageTool** via the
  `language_tool_python` binding. A single `LanguageTool` instance is
  opened for the whole batch — JVM startup is the expensive part — and
  every utterance's text is passed through `tool.check()`. Each match
  becomes a `Correction` (rule id, category, message, best replacement,
  offset/length, and a short context window).
- **Pass 2 (opt-in) — local LLM rewrite** via `--llm`. Sends each
  utterance to a local Ollama server with a tight prompt
  (*"Rewrite this spoken English sentence to sound natural and
  grammatical. Keep the meaning. Return only the rewrite."*) and stores
  the response as `UtteranceReport.rewrite`. The HTTP call uses stdlib
  `urllib`; no extra dependency is added. Any network or JSON failure
  yields `None` rather than crashing the run.

**Aggregation:** `per_speaker_stats` produces a `SpeakerStats` per
speaker — turn count, word count, total corrections, and the top-N
LanguageTool **categories** and **rule ids** by frequency. These feed
the per-speaker summary section in the Markdown report.

**Why this split:**
- LanguageTool is the cheapest credible source of mechanical-error
  feedback: rule-based, fully offline, and fast enough that the JVM
  startup dominates wall time. It catches the bulk of practical-English
  mistakes (subject-verb agreement, capitalization, *interesting* vs
  *interested*, *me* vs *I*) without any model download.
- A local LLM is the right tool for *phrasing* feedback ("rewrite this
  to sound natural") but is heavy: Ollama needs a separate runtime, GBs
  of model weights, and ~tens of seconds per utterance on CPU. Making
  it opt-in keeps the default `coach` invocation single-process and
  small.
- Match-attribute compatibility: `language_tool_python` v3+ exposes
  snake-case attributes (`rule_id`, `error_length`); older versions
  used camel-case. The extractor reads both so we are not pinned to
  one major version of the library.

**System dependency:** LanguageTool ships as a Java JAR
(~259 MB) which `language_tool_python` downloads on first use into the
user cache. A Java runtime (JRE 17+) must be installed; the CLI
catches the startup error and prints an apt/dnf hint.

**Trade-offs we accepted:**
- The 259 MB JAR download is one-time and not avoidable if we want
  rule-based grammar checking offline. Users who already accepted the
  ~145 MB ASR download tend to find this acceptable.
- LanguageTool's per-rule output is verbose. The report format keeps
  one bullet per match with rule-id + category + message + suggested
  fix; deeper triage is left to the user for now.
- The Ollama rewrite is per-utterance, not per-speaker. That keeps the
  prompt short and the failure isolated, at the cost of not exploiting
  cross-utterance context (e.g. consistent tense across a turn).

### 3.8 End-to-end orchestrator — `encocoa run`

**Decision:** A thin wrapper that builds three `argparse.Namespace` objects
(one each for `record`, `process`, `coach`) from a single set of run-level
flags and calls `_cmd_record`, `_cmd_process`, `_cmd_coach` in sequence.
After the coach step, the final `<stem>.report.md` is opened with the
OS default handler (`xdg-open` / `open` / `os.startfile`) unless
`--no-open` is set.

**Why this composition (not a separate pipeline module):**
- The per-stage subcommand handlers already encapsulate every behavior we
  want — argument validation, error codes, progress printing, default
  paths. Reusing them keeps `run` and the per-stage commands strictly in
  sync; there is exactly one place where each stage's behavior lives.
- All session artifacts share a common stem (`session-YYYYMMDD-HHMMSS` by
  default, or `--name`) and land in `--out-dir` (default: current
  directory). This means `run` produces exactly the same files a user
  would get from running the three commands by hand, in the same layout,
  with predictable names.
- `--no-coach` lets the user stop after the transcript step (useful when
  Java is unavailable). `--no-open` suppresses the auto-open (useful in
  CI or over SSH). Both flags are escape hatches; the default
  zero-flag invocation does the full pipeline and shows the report.

**Trade-offs we accepted:**
- The `run` subparser repeats most of the per-stage flags (model,
  diarizer, language, …) so users do not have to learn three different
  command surfaces. This duplicates the help text but keeps the mental
  model "one command, all knobs reachable from there".
- ASR cache directory is not exposed as a `run` flag — the default
  (`~/.cache/encocoa/models/`) is correct for almost everyone, and power
  users who need to override it should use `encocoa process` directly.
- Auto-open is best-effort: if `xdg-open` is missing on Linux, the path
  to the report is printed instead of aborting the run.

### 3.9 Polish features — `--names`, named device picker, silent-audio warning

**Decisions (Stage 6 round 1):**

- **`--names "Italo,Maria"`** is parsed by `_parse_names` in `cli.py` into a
  `{canonical → display}` map keyed by canonical labels in pipeline order
  (`{"A": "Italo", "B": "Maria"}`). The map is plumbed through to
  `report.save_markdown` and `coach.save_report` and consumed by a single
  helper, `report.speaker_label(speaker, names)`, which both modules call.
  The dialog JSON (`<stem>.dialog.json`) intentionally still stores
  canonical `A`/`B` so the same artifact can be re-rendered later with
  different display names.
- **Named device picker** — `audio.resolve_input_device(spec)` accepts
  `None` (system default), an integer (passed through), an all-digit
  string (parsed to int), or a name substring (case-insensitive). Multiple
  matches raise `ValueError` with the candidate list rather than guessing,
  and an unknown name prints all available devices in the error message.
- **Silent-audio warning** — `wav_utils.peak_dbfs` (a tiny module that
  depends only on `numpy` + `soundfile`, not on PortAudio) computes the
  WAV peak in dBFS. `_cmd_process` calls it before ASR and prints a
  warning when the peak is below `SILENT_THRESHOLD_DBFS = -50.0`. The
  warning does not abort processing — the user might still want to see
  what (if anything) ASR produces.
- **Mic-busy error** — `_cmd_record` catches `OSError` (the parent class
  of `sounddevice.PortAudioError`) raised by `record_wav` and prints a
  message naming the typical cause ("microphone in use by another app")
  with exit code 4 instead of letting the traceback escape.

**Why a separate `wav_utils` module:** the silent-audio check needs to run
inside `process`, which does **not** import `audio` (and therefore does
not require PortAudio). `wav_utils` keeps the dependency footprint tight:
it imports only `numpy` and `soundfile`, both already-required deps, so
machines that lack a working microphone stack can still run `process`
on existing WAVs.

### 3.10 What is **not** present yet

| Component                | Planned in stage | Notes                                          |
|--------------------------|------------------|------------------------------------------------|
| Per-speaker enrollment       | Stage 6 round 2 | Stable Italo/Maria labels across sessions.   |
| Config TOML                  | Stage 6 round 2 | `~/.config/encocoa/config.toml` default flags. |
| GUI (Tk/PySide)              | Stage 7 (opt)| Only after CLI is solid.                       |

---

## 4. Implementation status by stage

The plan in `PLAN.md` lays out seven stages. Status as of the most
recent commit:

### Stage 0 — Project skeleton  ✅ done

`uv` project initialized with `pyproject.toml`, `uv.lock`,
`.python-version` (3.12), `.gitignore`, `src/encocoa/`, and `tests/`.
The `encocoa` CLI is wired with subcommands `record`, `process`,
`coach`, and `run` via `argparse`.

**Verifies:** `uv run encocoa --help` lists all four subcommands.

### Stage 1 — Audio capture  ✅ done

`src/encocoa/audio.py` implements `list_input_devices`,
`print_input_devices`, and `record_wav` (16 kHz mono int16 WAV via
`sounddevice` + `soundfile`). The CLI exposes:

- `encocoa record --duration <s> --out <path>`
- `--device <id>`, `--list-devices`, `--samplerate <hz>`
- `--vad-trim` (flag declared, deferred to a later stage)

A live progress line shows elapsed/total time and a dBFS level meter.
Ctrl+C flushes the partial recording. PortAudio errors are caught at
import time and translated into per-platform install hints.

**Verifies:** `uv run encocoa record --list-devices` prints the system's
input-capable devices with the default marked. `uv run encocoa record
--duration 5 --out test.wav` writes a 5-second WAV that plays back
correctly.

### Stage 2 — ASR transcription  ✅ done

`src/encocoa/asr.py` wraps `faster-whisper` behind a small dataclass
API (`TranscriptSegment`, `TranscriptionStats`). The CLI exposes
`encocoa process <wav>` with `--model`, `--language`, `--device`,
`--compute-type`, `--beam-size`, and `--model-dir`. Models cache under
`~/.cache/encocoa/models/`. After completion the CLI prints a stats
line: `model=… audio=… proc=… rtf=… segments=… words=… chars=…
lang=…`.

**Verifies:** A 1-second silent WAV transcribes to `[]` in ~0.2 s
with `tiny.en`, confirming the full ASR path works. The Stage 2 exit
criterion (a 2-minute hand-recorded sample transcribes accurately on
CPU in < ~30 s) is left for the user to confirm with real audio.

### Stage 3 — Diarization (2 speakers)  ✅ done

Three new modules:

- `src/encocoa/diarize.py` — both `diarize_simple` (Resemblyzer +
  KMeans, default) and `diarize_pyannote` (opt-in, requires HF token
  and `uv add pyannote.audio`). Cluster labels are canonicalized to
  `A`/`B`/… by time of first appearance and adjacent same-speaker
  windows are coalesced.
- `src/encocoa/merge.py` — `merge` aligns ASR segments to
  diarization segments by maximum temporal overlap, with a
  closest-by-midpoint fallback when no overlap exists.
  `coalesce_consecutive` joins adjacent same-speaker turns.
- `src/encocoa/report.py` — `render_markdown` and `save_markdown`
  produce the `**Speaker A** (mm:ss–mm:ss): text` format.

The `process` subcommand now runs the **full pipeline** by default
(ASR → diarize → merge → render). New flags:

- `--diarizer {simple,pyannote}` (default `simple`)
- `--num-speakers <n>` (default `2`)
- `--hf-token <token>` (or `HF_TOKEN` env var; pyannote backend only)
- `--no-diarize` (skip diarization, ASR only — useful for debugging)
- `--out-dir <dir>` replaces the old `--out` flag; all four output
  files are derived from the WAV stem.

**Verifies:** Smoke run on a 4-second noise WAV produces all four
output files; pipeline does not crash on empty ASR results. Unit tests
cover label canonicalization, coalescence, max-overlap assignment,
the closest-by-midpoint fallback, JSON schemas, and Markdown
formatting. The Stage 3 exit criterion (≥ 90 % words attributed to the
correct speaker on a real 2-min sample) is left for the user to
confirm with real audio.

### Stage 4 — English coach (corrections)  ✅ done

`src/encocoa/coach.py` wraps `language_tool_python` behind a small
dataclass API (`Correction`, `UtteranceReport`, `SpeakerStats`) and
adds an injectable Ollama client (`_ollama_generate` / stdlib
`urllib`) for the optional rewrite pass. The CLI exposes:

- `encocoa coach <dialog.json>`
- `--out <path>` (default: `<stem>.report.md` beside the input)
- `--language <code>` (LanguageTool language, default `en-US`)
- `--llm` (opt-in Ollama rewrite pass)
- `--ollama-model <name>` (default `phi3:mini`)
- `--ollama-host <url>` (default `http://localhost:11434`)

A single `LanguageTool` instance is opened per `coach` invocation
and reused across all utterances. Match-attribute reading tolerates
both snake-case (v3+) and camel-case (legacy) Match objects. Per-
speaker statistics (`per_speaker_stats`) aggregate turn count, word
count, total corrections, and top-N categories and rule ids. The
report is rendered with `render_report`/`save_report` and contains
two sections: a per-speaker summary table-style block and the
dialogue with inline corrections.

Exit codes from the `coach` subcommand: `0` success, `2` missing or
malformed dialog JSON, `4` LanguageTool unavailable (no Java
runtime, JAR download failed, etc.).

**Verifies:** End-to-end smoke on a synthetic 4-turn 2-speaker
dialog with deliberate mistakes produced 5 mechanical corrections
across 2 speakers — `BASE_FORM`, `HE_VERB_AGR`,
`MORFOLOGIK_RULE_EN_US`, `CONFUSION_OF_ME_I`, `BE_INTEREST_IN` —
plus a per-speaker summary with top categories and rules. The
Stage 4 exit criterion ("report contains corrections for at least
mechanical errors with no LLM dependency required") is met.

### Stage 5 — `encocoa run` end-to-end  ✅ done

`_cmd_run` (in `cli.py`) sequences `record` → `process` → `coach` and
opens the final `<stem>.report.md` with the OS default handler. The CLI
exposes:

- `encocoa run` (no flags required — defaults to a 10-minute recording,
  `small.en` ASR, `simple` diarizer, `en-US` coach, auto-open on)
- Session naming: `--out-dir <dir>` (default: cwd), `--name <stem>`
  (default: `session-YYYYMMDD-HHMMSS`)
- All per-stage knobs surfaced at the run level: `--duration`,
  `--device`, `--samplerate`, `--model`, `--language`, `--asr-device`,
  `--compute-type`, `--beam-size`, `--diarizer`, `--num-speakers`,
  `--hf-token`, `--coach-language`, `--llm`, `--ollama-model`,
  `--ollama-host`
- Escape hatches: `--no-coach` stops after the transcript stage;
  `--no-open` suppresses the auto-open

Exit codes are the union of the per-stage codes — a record failure
returns the record code, a process failure returns the process code,
etc. The first non-zero code aborts the run.

**Verifies:** Two end-to-end tests in `tests/test_cli.py` stub every
underlying module (`audio`, `asr`, `diarize`, `merge`, `report`,
`coach`) and assert that the full run writes `wav`, `transcript.json`,
`diarization.json`, `dialog.json`, `transcript.md`, and `report.md` in
`--out-dir`, and that `_open_path` is invoked with the report. A
third test exercises `--no-coach` and `--no-open` together. The full
suite of 75 tests runs in under a second. The Stage 5 exit criterion
("`encocoa run` on a fresh machine produces a usable report") is left
for live verification — it requires a working microphone, a one-time
ASR model download, and a JRE for the coach.

### Stage 6 — Polish & robustness  🔄 in progress (round 1 done)

**Round 1 — done:**

- `--names "Italo,Maria"` on `process`, `coach`, and `run` replaces
  `Speaker A`/`Speaker B` in both Markdown outputs. Implemented as a
  `_parse_names` helper in `cli.py` plus a single
  `report.speaker_label` rendering helper shared by `report.py` and
  `coach.py`. The canonical labels (`A`, `B`, …) remain in
  `dialog.json` so the same artifact can be re-rendered later.
- `--device` on `record` and `run` accepts a name substring as well
  as an integer index (`audio.resolve_input_device`). Unknown or
  ambiguous matches raise `ValueError` with the candidate list, and
  the CLI surfaces a friendly message with exit code 2.
- Silent-audio warning in `process`: a new lightweight
  `src/encocoa/wav_utils.py` module (only depends on `numpy` +
  `soundfile`, no PortAudio) computes WAV peak dBFS; the CLI warns
  when the peak is below `-50 dBFS` but does not abort.
- Mic-busy error in `record`: `OSError` (parent of
  `sounddevice.PortAudioError`) is caught and rendered as "could not
  open audio input — is the microphone in use by another app …",
  with exit code 4.

**Verifies:** Twelve new tests cover `_parse_names`, `--names`
end-to-end through `process` and `coach`, name-string device
resolution and friendly errors for unknown/ambiguous/busy devices,
the silent-audio warning, plus six unit tests for
`audio.resolve_input_device` and seven for `wav_utils`. Suite is now
**94 passing in ~0.2 s**.

**Round 2 — deferred:** speaker enrollment (stable labels across
sessions), `~/.config/encocoa/config.toml` for default flags.

### Stage 7 — Optional GUI  ⏳ pending

Tk/PySide front-end. Only after the CLI is solid.

---

## 5. Repository layout

```
EnCoCoA/
├── INSTRUCTIONS.md          # original problem statement (input)
├── PLAN.md                  # 7-stage plan (frozen design)
├── documentation.md         # this file (living technical doc)
├── pyproject.toml           # project + deps (managed by uv)
├── uv.lock                  # reproducible resolution
├── .python-version          # 3.12
├── .gitignore
├── src/
│   └── encocoa/
│       ├── __init__.py      # __version__
│       ├── cli.py           # argparse entry point, all subcommands
│       ├── audio.py         # Stage 1
│       ├── asr.py           # Stage 2
│       ├── diarize.py       # Stage 3
│       ├── merge.py         # Stage 3
│       ├── report.py        # Stage 3 (+ speaker_label helper, Stage 6.1)
│       ├── coach.py         # Stage 4
│       └── wav_utils.py     # Stage 6.1: peak dBFS / silent-audio detection
└── tests/
    ├── test_cli.py
    ├── test_audio.py
    ├── test_asr.py
    ├── test_diarize.py
    ├── test_merge.py
    ├── test_report.py
    ├── test_coach.py
    └── test_wav_utils.py
```

`encocoa run` is implemented in `cli.py` as a thin orchestrator over the
existing subcommand handlers; it adds no new module.

The repo-level `.gitignore` excludes Python caches, virtual envs,
build artifacts, editor state, local secrets (`.env*`), and the
EnCoCoA runtime artifacts (`*.wav`, `*.transcript.json`,
`*.transcript.md`, `*.diarization.json`, `*.dialog.json`,
`*.report.md`, plus `sessions/` and `models/`). Local Claude Code
state (`.claude/settings.local.json`) is ignored, but project-level
Claude content under `.claude/` (slash commands, agents, shared
settings) is tracked. Test fixtures committed under `tests/data/` are
re-allowed via a negation pattern so regression WAVs are not blocked
by `*.wav`.

---

## 6. Quickstart

```bash
# one-time
sudo apt install libportaudio2 default-jre   # PortAudio for the recorder, JRE for the coach
uv sync                                       # install all Python deps

# simplest path — record, process, coach, open the report
uv run encocoa run
# produces session-YYYYMMDD-HHMMSS.{wav,transcript.json,diarization.json,
#          dialog.json,transcript.md,report.md} in the current directory
# and opens the report with the OS default Markdown viewer

# common variants
uv run encocoa run --duration 120 --no-open                  # 2-minute session, do not auto-open
uv run encocoa run --out-dir ./sessions --name lesson-01     # custom output location and stem
uv run encocoa run --no-coach                                # stop after the transcript step
uv run encocoa run --names "Italo,Maria"                     # use real names instead of Speaker A/B
uv run encocoa run --device "USB Mic"                        # pick the input device by name substring
```

If you prefer to run the stages individually (useful for debugging or for
re-running just one stage against an existing artifact):

```bash
uv run encocoa record --duration 600 --out session.wav
uv run encocoa process session.wav --model small.en
# produces session.transcript.json, session.diarization.json,
#          session.dialog.json, session.transcript.md

uv run encocoa coach session.dialog.json
# produces session.report.md (mechanical corrections + per-speaker stats)
```

To add an LLM phrasing pass (requires a running local Ollama server):

```bash
ollama serve &                                       # in another shell
ollama pull phi3:mini                                # one-time, ~2 GB
uv run encocoa coach session.dialog.json --llm
```

To use the more accurate (but heavier) pyannote backend:

```bash
uv add pyannote.audio
# Accept terms at https://hf.co/pyannote/speaker-diarization-3.1
export HF_TOKEN=hf_xxx
uv run encocoa process session.wav --diarizer pyannote
```

---

## 7. Testing

Run the suite with:

```bash
uv run pytest -q
```

The tests stub heavy dependencies (`faster_whisper`, `sounddevice`,
`resemblyzer`, `sklearn.cluster`, `pyannote.audio`,
`language_tool_python`, Ollama HTTP) so the suite runs in under a
second and never hits the network or the JVM. Integration runs against
real audio and a real LanguageTool JAR are intentionally manual — they
are the exit criteria for each stage.

---

## 8. Known limitations

- **Single-microphone recording.** EnCoCoA assumes both speakers share
  one microphone. Diarization quality depends heavily on the recording
  setup; cheap headsets and noisy rooms hurt the simple backend the
  most.
- **Two speakers, fixed.** The clustering uses `n_clusters =
  num_speakers`. Three-way conversations work in principle by passing
  `--num-speakers 3`, but the rest of the pipeline (and the report
  format) is tuned for two.
- **English only.** The `*.en` faster-whisper models do not
  transcribe other languages. The `--language auto` option exists for
  experiments but is not part of the design target.
- **First-run downloads.** ASR models (~145 MB for `small.en` at int8),
  diarization embeddings, and the LanguageTool JAR (~259 MB) download
  on first use. After that everything is local.
- **Torch is unavoidable.** Both the simple and the pyannote diarizer
  depend on torch. There is no acceptable torch-free path for
  speaker-embedding-based diarization at this quality level.
- **JRE required for the coach.** `language_tool_python` runs the
  LanguageTool server in-process via JPype; a Java runtime (JRE 17+)
  must be installed. The CLI prints an apt/dnf hint when it cannot
  start the JVM. This is an unavoidable consequence of using
  LanguageTool offline.
- **Ollama rewrite is opt-in and best-effort.** When `--llm` is set,
  failures to reach the Ollama server (down, wrong port, model not
  pulled) silently produce `rewrite=None` for that utterance rather
  than aborting the whole report. The Markdown only renders a
  "Rewrite:" line when the rewrite is non-empty and differs from the
  original.
