from __future__ import annotations

import json
from pathlib import Path

import pytest

from encocoa.asr import TranscriptSegment
from encocoa.diarize import DiarizationSegment
from encocoa.merge import (
    DialogTurn,
    assign_speaker,
    coalesce_consecutive,
    merge,
    save_dialog,
)


def test_assign_speaker_max_overlap_wins() -> None:
    asr = TranscriptSegment(start=0.0, end=2.0, text="hello there")
    diar = [
        DiarizationSegment(start=0.0, end=0.6, speaker="A"),  # 0.6s overlap
        DiarizationSegment(start=0.6, end=2.0, speaker="B"),  # 1.4s overlap
    ]
    assert assign_speaker(asr, diar) == "B"


def test_assign_speaker_no_overlap_uses_closest() -> None:
    asr = TranscriptSegment(start=10.0, end=11.0, text="…")
    diar = [
        DiarizationSegment(start=0.0, end=1.0, speaker="A"),  # midpoint 0.5, dist 10
        DiarizationSegment(start=20.0, end=21.0, speaker="B"),  # midpoint 20.5, dist 10
    ]
    # Tie broken by `min` stability: the first one wins.
    assert assign_speaker(asr, diar) == "A"


def test_assign_speaker_empty_falls_back() -> None:
    asr = TranscriptSegment(start=0.0, end=1.0, text="hi")
    assert assign_speaker(asr, [], fallback="X") == "X"


def test_merge_attributes_each_asr_segment() -> None:
    asr_segs = [
        TranscriptSegment(start=0.0, end=1.0, text="hi"),
        TranscriptSegment(start=1.5, end=3.0, text="how are you"),
        TranscriptSegment(start=3.5, end=4.5, text="fine"),
    ]
    diar_segs = [
        DiarizationSegment(start=0.0, end=1.2, speaker="A"),
        DiarizationSegment(start=1.2, end=3.2, speaker="B"),
        DiarizationSegment(start=3.2, end=5.0, speaker="A"),
    ]
    turns = merge(asr_segs, diar_segs)
    assert [t.speaker for t in turns] == ["A", "B", "A"]
    assert [t.text for t in turns] == ["hi", "how are you", "fine"]


def test_coalesce_consecutive_joins_same_speaker() -> None:
    turns = [
        DialogTurn(start=0.0, end=1.0, speaker="A", text="hi"),
        DialogTurn(start=1.0, end=2.0, speaker="A", text="there"),
        DialogTurn(start=2.0, end=3.0, speaker="B", text="hello"),
        DialogTurn(start=3.0, end=4.0, speaker="A", text="how"),
        DialogTurn(start=4.0, end=5.0, speaker="A", text="are you"),
    ]
    out = coalesce_consecutive(turns)
    assert [(t.speaker, t.text) for t in out] == [
        ("A", "hi there"),
        ("B", "hello"),
        ("A", "how are you"),
    ]
    assert out[0].start == 0.0 and out[0].end == 2.0
    assert out[2].start == 3.0 and out[2].end == 5.0


def test_coalesce_consecutive_empty() -> None:
    assert coalesce_consecutive([]) == []


def test_save_dialog_writes_expected_schema(tmp_path: Path) -> None:
    turns = [
        DialogTurn(start=0.0, end=1.5, speaker="A", text="hi"),
        DialogTurn(start=1.5, end=3.0, speaker="B", text="hello"),
    ]
    out = tmp_path / "subdir" / "session.dialog.json"
    save_dialog(turns, out)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload == [
        {"start": 0.0, "end": 1.5, "speaker": "A", "text": "hi"},
        {"start": 1.5, "end": 3.0, "speaker": "B", "text": "hello"},
    ]
