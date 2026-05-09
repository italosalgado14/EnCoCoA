from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .asr import TranscriptSegment
from .diarize import DiarizationSegment


@dataclass(frozen=True)
class DialogTurn:
    start: float
    end: float
    speaker: str
    text: str

    def to_dict(self) -> dict:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "speaker": self.speaker,
            "text": self.text,
        }


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(
    asr_segment: TranscriptSegment,
    diar_segments: list[DiarizationSegment],
    *,
    fallback: str = "A",
) -> str:
    """Pick the speaker whose diarization span has the largest overlap with `asr_segment`.

    If no diarization segment overlaps, fall back to the temporally closest
    diarization segment (by midpoint distance). If diarization is empty, return
    `fallback`.
    """
    if not diar_segments:
        return fallback
    best_overlap = 0.0
    best_speaker: str | None = None
    for d in diar_segments:
        ov = _overlap(asr_segment.start, asr_segment.end, d.start, d.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = d.speaker
    if best_speaker is not None:
        return best_speaker
    mid = 0.5 * (asr_segment.start + asr_segment.end)
    closest = min(
        diar_segments,
        key=lambda d: abs(0.5 * (d.start + d.end) - mid),
    )
    return closest.speaker


def merge(
    asr_segments: list[TranscriptSegment],
    diar_segments: list[DiarizationSegment],
) -> list[DialogTurn]:
    """Attribute each ASR segment to a speaker via maximum-overlap alignment."""
    return [
        DialogTurn(
            start=s.start,
            end=s.end,
            speaker=assign_speaker(s, diar_segments),
            text=s.text,
        )
        for s in asr_segments
    ]


def coalesce_consecutive(turns: list[DialogTurn]) -> list[DialogTurn]:
    """Join adjacent turns by the same speaker into a single turn."""
    if not turns:
        return []
    merged = [turns[0]]
    for t in turns[1:]:
        last = merged[-1]
        if t.speaker == last.speaker:
            joined_text = (last.text + " " + t.text).strip() if t.text else last.text
            merged[-1] = DialogTurn(
                start=last.start,
                end=max(last.end, t.end),
                speaker=last.speaker,
                text=joined_text,
            )
        else:
            merged.append(t)
    return merged


def save_dialog(turns: Iterable[DialogTurn], out_path: Path | str) -> Path:
    out = Path(out_path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([t.to_dict() for t in turns], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out
