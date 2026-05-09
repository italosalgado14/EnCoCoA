from __future__ import annotations

from pathlib import Path

from encocoa.merge import DialogTurn
from encocoa.report import format_timestamp, render_markdown, save_markdown


def test_format_timestamp_short() -> None:
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(5) == "00:05"
    assert format_timestamp(125.7) == "02:05"


def test_format_timestamp_with_hours() -> None:
    assert format_timestamp(3661) == "01:01:01"


def test_format_timestamp_negative_clamps() -> None:
    assert format_timestamp(-3.0) == "00:00"


def test_render_markdown_two_speakers() -> None:
    turns = [
        DialogTurn(start=0.0, end=4.0, speaker="A", text="Hi, how are you today?"),
        DialogTurn(start=5.0, end=9.0, speaker="B", text="I am good, and you?"),
    ]
    md = render_markdown(turns)
    assert "**Speaker A** (00:00–00:04): Hi, how are you today?" in md
    assert "**Speaker B** (00:05–00:09): I am good, and you?" in md


def test_render_markdown_with_title() -> None:
    turns = [DialogTurn(start=0, end=1, speaker="A", text="hi")]
    md = render_markdown(turns, title="session-001")
    assert md.startswith("# session-001\n")
    assert "**Speaker A**" in md


def test_save_markdown_creates_dirs(tmp_path: Path) -> None:
    turns = [DialogTurn(start=0, end=1, speaker="A", text="hi")]
    out = tmp_path / "deep" / "nest" / "session.transcript.md"
    save_markdown(turns, out, title="t")
    assert out.exists()
    content = out.read_text()
    assert "# t" in content
    assert "**Speaker A**" in content


def test_render_markdown_with_names() -> None:
    from encocoa.report import speaker_label

    turns = [
        DialogTurn(start=0, end=4, speaker="A", text="Hi."),
        DialogTurn(start=5, end=9, speaker="B", text="Hello."),
    ]
    md = render_markdown(turns, names={"A": "Italo", "B": "Maria"})
    assert "**Italo**" in md
    assert "**Maria**" in md
    assert "Speaker A" not in md
    assert "Speaker B" not in md
    # speaker_label helper
    assert speaker_label("A", {"A": "Italo"}) == "Italo"
    assert speaker_label("C", {"A": "Italo"}) == "Speaker C"
    assert speaker_label("A", None) == "Speaker A"


def test_render_markdown_partial_names() -> None:
    """When --names provides fewer entries than speakers, unmapped speakers fall back."""
    turns = [
        DialogTurn(start=0, end=1, speaker="A", text="hi"),
        DialogTurn(start=1, end=2, speaker="C", text="yo"),
    ]
    md = render_markdown(turns, names={"A": "Italo"})
    assert "**Italo**" in md
    assert "**Speaker C**" in md
