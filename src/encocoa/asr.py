from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_MODEL = "small.en"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "encocoa" / "models"


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
        }


@dataclass(frozen=True)
class TranscriptionStats:
    audio_duration: float
    processing_time: float
    language: str
    language_probability: float
    model: str
    segment_count: int
    word_count: int
    char_count: int

    @property
    def real_time_factor(self) -> float:
        if self.processing_time <= 0:
            return float("inf")
        return self.audio_duration / self.processing_time

    def to_dict(self) -> dict:
        d = asdict(self)
        d["real_time_factor"] = round(self.real_time_factor, 3)
        return d


def _iter_with_progress(
    segments: Iterable, total_duration: float, stream
) -> Iterator[TranscriptSegment]:
    last_pct = -1
    for s in segments:
        seg = TranscriptSegment(
            start=float(s.start),
            end=float(s.end),
            text=(s.text or "").strip(),
        )
        if total_duration > 0 and stream is not None:
            pct = int(min(100.0, (seg.end / total_duration) * 100.0))
            if pct != last_pct:
                print(
                    f"\r[ASR] {pct:3d}%  {seg.start:7.2f}–{seg.end:7.2f}s  "
                    f"{seg.text[:60]}",
                    end="",
                    flush=True,
                    file=stream,
                )
                last_pct = pct
        yield seg


def transcribe(
    wav_path: Path | str,
    model_name: str = DEFAULT_MODEL,
    *,
    model_dir: Path | str = DEFAULT_CACHE_DIR,
    language: str | None = "en",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    progress_stream=None,
) -> tuple[list[TranscriptSegment], TranscriptionStats]:
    """Transcribe a WAV file with faster-whisper, returning segments and stats."""
    from faster_whisper import WhisperModel  # local import: heavy backend

    wav = Path(wav_path)
    if not wav.exists():
        raise FileNotFoundError(f"audio file not found: {wav}")

    cache = Path(model_dir)
    cache.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(cache),
    )

    start_t = time.monotonic()
    segments_iter, info = model.transcribe(
        str(wav),
        language=language,
        beam_size=beam_size,
    )
    audio_duration = float(getattr(info, "duration", 0.0) or 0.0)

    out_stream = progress_stream if progress_stream is not None else sys.stdout
    segments = list(_iter_with_progress(segments_iter, audio_duration, out_stream))
    if progress_stream is not None or sys.stdout.isatty():
        print(file=out_stream)

    elapsed = time.monotonic() - start_t

    word_count = sum(len(seg.text.split()) for seg in segments)
    char_count = sum(len(seg.text) for seg in segments)

    stats = TranscriptionStats(
        audio_duration=audio_duration,
        processing_time=elapsed,
        language=str(getattr(info, "language", language or "")),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        model=model_name,
        segment_count=len(segments),
        word_count=word_count,
        char_count=char_count,
    )
    return segments, stats


def save_transcript(segments: Iterable[TranscriptSegment], out_path: Path | str) -> Path:
    out = Path(out_path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    payload = [s.to_dict() for s in segments]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def format_stats(stats: TranscriptionStats) -> str:
    rtf = stats.real_time_factor
    rtf_s = f"{rtf:.2f}×" if rtf != float("inf") else "∞"
    return (
        f"model={stats.model}  "
        f"audio={stats.audio_duration:.1f}s  "
        f"proc={stats.processing_time:.1f}s  "
        f"rtf={rtf_s}  "
        f"segments={stats.segment_count}  "
        f"words={stats.word_count}  "
        f"chars={stats.char_count}  "
        f"lang={stats.language} ({stats.language_probability:.2f})"
    )
