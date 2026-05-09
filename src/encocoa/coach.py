from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_LANGUAGE = "en-US"
DEFAULT_OLLAMA_MODEL = "phi3:mini"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
LLM_PROMPT = (
    "Rewrite this spoken English sentence to sound natural and grammatical. "
    "Keep the meaning. Return only the rewrite, with no quotes and no explanation.\n\n"
    "Sentence: {text}"
)


@dataclass(frozen=True)
class Correction:
    rule_id: str
    category: str
    message: str
    replacement: str | None
    offset: int
    length: int
    context: str

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "message": self.message,
            "replacement": self.replacement,
            "offset": self.offset,
            "length": self.length,
            "context": self.context,
        }


@dataclass(frozen=True)
class UtteranceReport:
    start: float
    end: float
    speaker: str
    text: str
    corrections: list[Correction] = field(default_factory=list)
    rewrite: str | None = None

    def to_dict(self) -> dict:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "speaker": self.speaker,
            "text": self.text,
            "corrections": [c.to_dict() for c in self.corrections],
            "rewrite": self.rewrite,
        }


@dataclass(frozen=True)
class SpeakerStats:
    speaker: str
    turn_count: int
    word_count: int
    correction_count: int
    top_categories: list[tuple[str, int]]
    top_rules: list[tuple[str, int]]

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "turn_count": self.turn_count,
            "word_count": self.word_count,
            "correction_count": self.correction_count,
            "top_categories": [{"name": n, "count": c} for n, c in self.top_categories],
            "top_rules": [{"id": n, "count": c} for n, c in self.top_rules],
        }


def load_dialog(path: Path | str) -> list[dict]:
    """Load a dialog JSON file produced by `encocoa process`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"dialog file must be a JSON list, got {type(payload).__name__}")
    return payload


def _match_to_correction(match: object, text: str) -> Correction:
    # language_tool_python v3+ exposes snake_case attributes (`rule_id`,
    # `error_length`); older versions used camelCase. Read both so the code
    # tolerates either.
    def _attr(*names: str, default: object = "") -> object:
        for n in names:
            v = getattr(match, n, None)
            if v is not None:
                return v
        return default

    rule_id = str(_attr("rule_id", "ruleId", default="") or "")
    category = str(_attr("category", default="") or "")
    message = str(_attr("message", default="") or "")
    replacements = list(_attr("replacements", default=[]) or [])
    replacement = str(replacements[0]) if replacements else None
    offset = int(_attr("offset", default=0) or 0)
    length = int(_attr("error_length", "errorLength", default=0) or 0)
    ctx_start = max(0, offset - 12)
    ctx_end = min(len(text), offset + length + 12)
    context = text[ctx_start:ctx_end]
    return Correction(
        rule_id=rule_id,
        category=category,
        message=message,
        replacement=replacement,
        offset=offset,
        length=length,
        context=context,
    )


def check_dialog(
    turns: list[dict],
    *,
    language: str = DEFAULT_LANGUAGE,
    tool: object | None = None,
) -> list[UtteranceReport]:
    """Run LanguageTool on each turn's text and return per-utterance reports.

    A single LanguageTool instance is opened for the whole batch (the JVM
    startup is the expensive part). Pass `tool` to reuse an already-opened
    instance, e.g. for testing.
    """
    owns_tool = False
    if tool is None:
        try:
            from language_tool_python import LanguageTool
        except ImportError as e:
            raise RuntimeError(
                "language_tool_python is not installed. "
                "Install with `uv add language-tool-python`."
            ) from e
        try:
            tool = LanguageTool(language)
        except Exception as e:
            raise RuntimeError(
                f"Could not start LanguageTool ({e}). "
                "LanguageTool needs a Java runtime; install one and retry. "
                "On Debian/Ubuntu: `sudo apt install default-jre`."
            ) from e
        owns_tool = True

    try:
        reports: list[UtteranceReport] = []
        for t in turns:
            text = str(t.get("text", ""))
            matches = tool.check(text) if text else []
            corrections = [_match_to_correction(m, text) for m in matches]
            reports.append(
                UtteranceReport(
                    start=float(t.get("start", 0.0)),
                    end=float(t.get("end", 0.0)),
                    speaker=str(t.get("speaker", "?")),
                    text=text,
                    corrections=corrections,
                )
            )
        return reports
    finally:
        if owns_tool:
            close = getattr(tool, "close", None)
            if callable(close):
                close()


def _ollama_generate(
    prompt: str,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = 60.0,
) -> str | None:
    """Call a local Ollama server's /api/generate. Returns None on any failure."""
    import urllib.error
    import urllib.request

    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None
    text = (payload.get("response") or "").strip()
    return text or None


def add_llm_rewrites(
    reports: list[UtteranceReport],
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    generate=None,
) -> list[UtteranceReport]:
    """Attach a phrasing rewrite from a local Ollama model to each report.

    `generate` is an injectable callable `(prompt, *, model, host) -> str|None` —
    defaults to the real Ollama HTTP client. Tests pass a fake.
    """
    gen = generate if generate is not None else _ollama_generate
    out: list[UtteranceReport] = []
    for r in reports:
        if not r.text.strip():
            out.append(r)
            continue
        rewrite = gen(LLM_PROMPT.format(text=r.text), model=model, host=host)
        out.append(
            UtteranceReport(
                start=r.start,
                end=r.end,
                speaker=r.speaker,
                text=r.text,
                corrections=r.corrections,
                rewrite=rewrite,
            )
        )
    return out


def per_speaker_stats(
    reports: Iterable[UtteranceReport], *, top_n: int = 5
) -> list[SpeakerStats]:
    by_speaker: dict[str, list[UtteranceReport]] = defaultdict(list)
    for r in reports:
        by_speaker[r.speaker].append(r)
    out: list[SpeakerStats] = []
    for speaker in sorted(by_speaker):
        items = by_speaker[speaker]
        words = sum(len(r.text.split()) for r in items)
        corrs = [c for r in items for c in r.corrections]
        cat_counter: Counter[str] = Counter(c.category for c in corrs if c.category)
        rule_counter: Counter[str] = Counter(c.rule_id for c in corrs if c.rule_id)
        out.append(
            SpeakerStats(
                speaker=speaker,
                turn_count=len(items),
                word_count=words,
                correction_count=len(corrs),
                top_categories=cat_counter.most_common(top_n),
                top_rules=rule_counter.most_common(top_n),
            )
        )
    return out


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def render_report(
    reports: list[UtteranceReport],
    stats: list[SpeakerStats],
    *,
    title: str | None = None,
    names: dict[str, str] | None = None,
) -> str:
    from .report import speaker_label

    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")

    lines.append("## Per-speaker summary")
    lines.append("")
    if not stats:
        lines.append("_No speakers found._")
        lines.append("")
    for s in stats:
        lines.append(f"### {speaker_label(s.speaker, names)}")
        lines.append(f"- Turns: {s.turn_count}")
        lines.append(f"- Words: {s.word_count}")
        lines.append(f"- Corrections: {s.correction_count}")
        if s.top_categories:
            lines.append(
                "- Top categories: "
                + ", ".join(f"{n} ({c})" for n, c in s.top_categories)
            )
        if s.top_rules:
            lines.append(
                "- Top rules: "
                + ", ".join(f"{n} ({c})" for n, c in s.top_rules)
            )
        lines.append("")

    lines.append("## Dialogue and corrections")
    lines.append("")
    if not reports:
        lines.append("_No utterances._")
        lines.append("")
    for r in reports:
        ts = f"({_format_timestamp(r.start)}–{_format_timestamp(r.end)})"
        lines.append(f"### {speaker_label(r.speaker, names)} {ts}")
        lines.append("")
        lines.append(f"> {r.text}" if r.text else "> _(empty utterance)_")
        lines.append("")
        if r.rewrite and r.rewrite.strip() and r.rewrite.strip() != r.text.strip():
            lines.append(f"**Rewrite:** {r.rewrite}")
            lines.append("")
        if r.corrections:
            for c in r.corrections:
                fix = c.replacement if c.replacement is not None else "—"
                rule = c.rule_id or "rule"
                cat = f" ({c.category})" if c.category else ""
                lines.append(f"- **{rule}**{cat}: {c.message}")
                if c.context:
                    lines.append(f"  - context: `…{c.context}…`")
                lines.append(f"  - suggested fix: `{fix}`")
            lines.append("")
        else:
            lines.append("_No mechanical issues found._")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_report(
    reports: list[UtteranceReport],
    stats: list[SpeakerStats],
    out_path: Path | str,
    *,
    title: str | None = None,
    names: dict[str, str] | None = None,
) -> Path:
    out = Path(out_path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_report(reports, stats, title=title, names=names), encoding="utf-8"
    )
    return out
