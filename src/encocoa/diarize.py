from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_NUM_SPEAKERS = 2
RESEMBLYZER_INTERNAL_SR = 16000  # Resemblyzer always works at 16 kHz internally.


@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str  # canonical labels: "A", "B", "C", ...

    def to_dict(self) -> dict:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "speaker": self.speaker,
        }


def _label_by_first_appearance(
    items: list[tuple[float, float, object]],
) -> list[DiarizationSegment]:
    """Map raw cluster IDs to A, B, C... in the order they first appear in time."""
    mapping: dict[object, str] = {}
    next_idx = 0
    out: list[DiarizationSegment] = []
    for start, end, raw_id in sorted(items, key=lambda t: (t[0], t[1])):
        if raw_id not in mapping:
            mapping[raw_id] = chr(ord("A") + next_idx)
            next_idx += 1
        out.append(DiarizationSegment(start=float(start), end=float(end), speaker=mapping[raw_id]))
    return out


def _coalesce(
    segments: list[DiarizationSegment], gap_tolerance: float = 0.2
) -> list[DiarizationSegment]:
    """Merge consecutive segments that share a speaker and are separated by at most `gap_tolerance` seconds."""
    if not segments:
        return []
    merged = [segments[0]]
    for s in segments[1:]:
        last = merged[-1]
        if s.speaker == last.speaker and (s.start - last.end) <= gap_tolerance:
            merged[-1] = DiarizationSegment(
                start=last.start, end=max(last.end, s.end), speaker=last.speaker
            )
        else:
            merged.append(s)
    return merged


def diarize_pyannote(
    wav_path: Path | str,
    num_speakers: int = DEFAULT_NUM_SPEAKERS,
    *,
    hf_token: str | None = None,
    device: str = "cpu",
) -> list[DiarizationSegment]:
    """Run pyannote.audio's speaker-diarization-3.1 pipeline.

    Requires `pyannote.audio` and a HuggingFace token (the model is gated).
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError(
            "pyannote.audio is not installed. Either install it "
            "(`uv add pyannote.audio`) or use --diarizer simple."
        ) from e

    token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "pyannote requires a HuggingFace token. "
            "Accept the model terms at https://hf.co/pyannote/speaker-diarization-3.1, "
            "then pass --hf-token or set HF_TOKEN. "
            "Alternative: --diarizer simple (no token required)."
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    if device != "cpu":
        import torch

        pipeline.to(torch.device(device))

    annotation = pipeline(str(wav_path), num_speakers=num_speakers)
    raw: list[tuple[float, float, object]] = [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    return _coalesce(_label_by_first_appearance(raw))


def diarize_simple(
    wav_path: Path | str,
    num_speakers: int = DEFAULT_NUM_SPEAKERS,
) -> list[DiarizationSegment]:
    """Lightweight fallback: Resemblyzer voice embeddings + KMeans clustering.

    No HuggingFace gating. Less accurate than pyannote but works fully offline
    after the first model download.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as e:
        raise RuntimeError(
            "Resemblyzer is not installed. Install with `uv add resemblyzer scikit-learn`."
        ) from e
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise RuntimeError(
            "scikit-learn is not installed. Install with `uv add scikit-learn`."
        ) from e

    wav = preprocess_wav(str(wav_path))  # 16 kHz mono float32
    encoder = VoiceEncoder(verbose=False)

    _mean, partial_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True)
    if len(partial_embeds) == 0:
        return []

    intervals: list[tuple[float, float]] = [
        (sl.start / RESEMBLYZER_INTERNAL_SR, sl.stop / RESEMBLYZER_INTERNAL_SR)
        for sl in wav_splits
    ]

    if len(partial_embeds) < num_speakers:
        # Not enough windows to cluster — assign all to A.
        segs = [
            DiarizationSegment(start=s, end=e, speaker="A") for s, e in intervals
        ]
        return _coalesce(segs)

    km = KMeans(n_clusters=num_speakers, n_init=10, random_state=0).fit(partial_embeds)
    raw: list[tuple[float, float, object]] = [
        (start, end, int(label))
        for (start, end), label in zip(intervals, km.labels_)
    ]
    return _coalesce(_label_by_first_appearance(raw))


def save_diarization(
    segments: Iterable[DiarizationSegment], out_path: Path | str
) -> Path:
    out = Path(out_path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def diarization_summary(segments: list[DiarizationSegment]) -> dict[str, float]:
    """Total speaking time per speaker (seconds)."""
    out: dict[str, float] = {}
    for s in segments:
        out[s.speaker] = out.get(s.speaker, 0.0) + max(0.0, s.end - s.start)
    return out
