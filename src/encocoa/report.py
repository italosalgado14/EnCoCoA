from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .merge import DialogTurn


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def speaker_label(speaker: str, names: dict[str, str] | None = None) -> str:
    """Return a display label for a canonical speaker id (e.g. 'A')."""
    if names and speaker in names and names[speaker]:
        return names[speaker]
    return f"Speaker {speaker}"


def render_markdown(
    turns: Iterable[DialogTurn],
    *,
    title: str | None = None,
    names: dict[str, str] | None = None,
) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    for t in turns:
        ts = f"({format_timestamp(t.start)}–{format_timestamp(t.end)})"
        lines.append(f"**{speaker_label(t.speaker, names)}** {ts}: {t.text}")
    return "\n".join(lines) + "\n"


def save_markdown(
    turns: Iterable[DialogTurn],
    out_path: Path | str,
    *,
    title: str | None = None,
    names: dict[str, str] | None = None,
) -> Path:
    out = Path(out_path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(turns, title=title, names=names), encoding="utf-8")
    return out
