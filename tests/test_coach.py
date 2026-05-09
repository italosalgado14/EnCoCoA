from __future__ import annotations

import json
from pathlib import Path

import pytest

from encocoa.coach import (
    Correction,
    SpeakerStats,
    UtteranceReport,
    _match_to_correction,
    _ollama_generate,
    add_llm_rewrites,
    check_dialog,
    load_dialog,
    per_speaker_stats,
    render_report,
    save_report,
)


# ---------- fakes ----------


class _FakeMatch:
    def __init__(
        self,
        ruleId: str = "RULE_X",
        category: str = "GRAMMAR",
        message: str = "wrong",
        replacements: list[str] | None = None,
        offset: int = 0,
        errorLength: int = 0,
    ) -> None:
        # Mirrors language_tool_python v3 Match (snake_case).
        self.rule_id = ruleId
        self.category = category
        self.message = message
        self.replacements = replacements or []
        self.offset = offset
        self.error_length = errorLength


class _FakeLT:
    """Stand-in for `language_tool_python.LanguageTool`."""

    def __init__(self, mapping: dict[str, list[_FakeMatch]] | None = None) -> None:
        self.mapping = mapping or {}
        self.calls: list[str] = []
        self.closed = False

    def check(self, text: str) -> list[_FakeMatch]:
        self.calls.append(text)
        return list(self.mapping.get(text, []))

    def close(self) -> None:
        self.closed = True


# ---------- _match_to_correction ----------


def test_match_to_correction_extracts_fields() -> None:
    text = "I has been here yesterday."
    m = _FakeMatch(
        ruleId="HE_VERB_AGR",
        category="GRAMMAR",
        message="Use 'have', not 'has'.",
        replacements=["have", "had"],
        offset=2,
        errorLength=3,
    )
    c = _match_to_correction(m, text)
    assert c.rule_id == "HE_VERB_AGR"
    assert c.category == "GRAMMAR"
    assert c.replacement == "have"
    assert c.offset == 2 and c.length == 3
    assert "has" in c.context


def test_match_to_correction_no_replacements_yields_none() -> None:
    c = _match_to_correction(_FakeMatch(replacements=[], offset=0, errorLength=2), "hi")
    assert c.replacement is None


# ---------- load_dialog ----------


def test_load_dialog_round_trip(tmp_path: Path) -> None:
    payload = [{"start": 0, "end": 1, "speaker": "A", "text": "hi"}]
    p = tmp_path / "x.dialog.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert load_dialog(p) == payload


def test_load_dialog_rejects_non_list(tmp_path: Path) -> None:
    p = tmp_path / "bad.dialog.json"
    p.write_text('{"oops": true}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_dialog(p)


# ---------- check_dialog ----------


def test_check_dialog_uses_injected_tool() -> None:
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "A", "text": "I has a cat."},
        {"start": 1.0, "end": 2.0, "speaker": "B", "text": "Me too."},
        {"start": 2.0, "end": 3.0, "speaker": "A", "text": ""},  # empty -> no check
    ]
    fake = _FakeLT(
        mapping={
            "I has a cat.": [
                _FakeMatch(ruleId="HE_VERB_AGR", message="agr", replacements=["have"]),
            ],
            "Me too.": [],
        }
    )
    reports = check_dialog(turns, tool=fake)
    assert [r.speaker for r in reports] == ["A", "B", "A"]
    assert [len(r.corrections) for r in reports] == [1, 0, 0]
    # Empty text must not be sent to the tool.
    assert fake.calls == ["I has a cat.", "Me too."]
    # Injected tool is not closed by check_dialog.
    assert fake.closed is False


def test_check_dialog_keeps_dialog_metadata() -> None:
    turns = [{"start": 1.5, "end": 2.5, "speaker": "B", "text": "hello"}]
    fake = _FakeLT(mapping={"hello": []})
    [r] = check_dialog(turns, tool=fake)
    assert r.start == 1.5 and r.end == 2.5 and r.speaker == "B" and r.text == "hello"


# ---------- per_speaker_stats ----------


def _corr(rule: str = "R1", cat: str = "GRAMMAR") -> Correction:
    return Correction(
        rule_id=rule, category=cat, message="m", replacement=None, offset=0, length=0, context=""
    )


def test_per_speaker_stats_aggregates_by_speaker() -> None:
    reports = [
        UtteranceReport(0, 1, "A", "one two three", corrections=[_corr("R1"), _corr("R2", "STYLE")]),
        UtteranceReport(1, 2, "A", "four", corrections=[_corr("R1")]),
        UtteranceReport(2, 3, "B", "five six", corrections=[]),
    ]
    stats = per_speaker_stats(reports)
    by_sp = {s.speaker: s for s in stats}

    assert by_sp["A"].turn_count == 2
    assert by_sp["A"].word_count == 4
    assert by_sp["A"].correction_count == 3
    # R1 appears twice, R2 once.
    assert by_sp["A"].top_rules[0] == ("R1", 2)
    cats = dict(by_sp["A"].top_categories)
    assert cats["GRAMMAR"] == 2 and cats["STYLE"] == 1

    assert by_sp["B"].turn_count == 1
    assert by_sp["B"].word_count == 2
    assert by_sp["B"].correction_count == 0
    assert by_sp["B"].top_rules == [] and by_sp["B"].top_categories == []


def test_per_speaker_stats_empty() -> None:
    assert per_speaker_stats([]) == []


# ---------- render_report / save_report ----------


def test_render_report_contains_all_sections() -> None:
    reports = [
        UtteranceReport(
            0.0,
            4.0,
            "A",
            "I has a cat.",
            corrections=[
                Correction(
                    rule_id="HE_VERB_AGR",
                    category="GRAMMAR",
                    message="Use 'have'.",
                    replacement="have",
                    offset=2,
                    length=3,
                    context="I has a cat",
                )
            ],
        ),
        UtteranceReport(5.0, 9.0, "B", "I am fine.", corrections=[]),
    ]
    stats = per_speaker_stats(reports)
    md = render_report(reports, stats, title="session-001")
    assert md.startswith("# session-001\n")
    assert "## Per-speaker summary" in md
    assert "### Speaker A" in md and "### Speaker B" in md
    assert "## Dialogue and corrections" in md
    assert "**HE_VERB_AGR**" in md
    assert "suggested fix: `have`" in md
    assert "_No mechanical issues found._" in md
    assert "(00:00–00:04)" in md and "(00:05–00:09)" in md


def test_render_report_handles_empty() -> None:
    md = render_report([], [], title="empty")
    assert "_No speakers found._" in md
    assert "_No utterances._" in md


def test_render_report_shows_rewrite_when_different() -> None:
    reports = [
        UtteranceReport(0, 1, "A", "I has a cat.", rewrite="I have a cat."),
        UtteranceReport(1, 2, "A", "I am fine.", rewrite="I am fine."),
    ]
    md = render_report(reports, [], title=None)
    assert "**Rewrite:** I have a cat." in md
    # Identical rewrite should not be rendered twice as "Rewrite".
    assert md.count("**Rewrite:**") == 1


def test_save_report_creates_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nest" / "session.report.md"
    save_report([], [], out, title="t")
    assert out.exists()
    assert "# t" in out.read_text()


# ---------- add_llm_rewrites ----------


def test_add_llm_rewrites_uses_injected_generator() -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_gen(prompt: str, *, model: str, host: str) -> str:
        calls.append((prompt, model, host))
        return "rewritten."

    reports = [
        UtteranceReport(0, 1, "A", "I has a cat."),
        UtteranceReport(1, 2, "B", ""),  # empty: skipped
    ]
    out = add_llm_rewrites(
        reports, model="phi3:mini", host="http://h", generate=fake_gen
    )
    assert out[0].rewrite == "rewritten."
    assert out[1].rewrite is None
    assert len(calls) == 1
    assert calls[0][1] == "phi3:mini" and calls[0][2] == "http://h"
    assert "I has a cat." in calls[0][0]


def test_add_llm_rewrites_propagates_none_on_failure() -> None:
    def bad_gen(prompt: str, *, model: str, host: str) -> None:
        return None

    [r] = add_llm_rewrites(
        [UtteranceReport(0, 1, "A", "hi.")], generate=bad_gen
    )
    assert r.rewrite is None


# ---------- _ollama_generate ----------


def test_ollama_generate_returns_none_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error
    import urllib.request

    def boom(req, timeout=None):  # noqa: ARG001
        raise urllib.error.URLError("no server")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert _ollama_generate("hi", host="http://localhost:1") is None
