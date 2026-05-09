from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


# ---- helpers --------------------------------------------------------------


class _Slice:
    """Stand-in for sounddevice/numpy slice objects produced by Resemblyzer."""

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop


class _FakeVoiceEncoder:
    """Stub Resemblyzer encoder that returns hand-crafted partial embeddings."""

    def __init__(self, partials, splits, *_, **__):
        self._partials = partials
        self._splits = splits

    def embed_utterance(self, wav, return_partials: bool = False):  # noqa: ANN001
        if return_partials:
            return None, self._partials, self._splits
        return None


class _FakeKMeans:
    """Stub KMeans that assigns labels based on the sign of the first feature."""

    def __init__(self, n_clusters=2, n_init=10, random_state=0) -> None:
        self.n_clusters = n_clusters

    def fit(self, X):  # noqa: ANN001
        # Assign label 0 if first dim < 0, else 1
        import numpy as np

        self.labels_ = np.array([0 if row[0] < 0 else 1 for row in X], dtype=int)
        return self


def _stub_resemblyzer_module(partials, splits) -> types.ModuleType:
    mod = types.ModuleType("resemblyzer")

    def _make_encoder(*args, **kwargs):
        return _FakeVoiceEncoder(partials, splits)

    mod.VoiceEncoder = _make_encoder  # type: ignore[attr-defined]
    mod.preprocess_wav = lambda path: object()  # type: ignore[attr-defined]
    return mod


def _stub_sklearn_module() -> types.ModuleType:
    cluster_mod = types.ModuleType("sklearn.cluster")
    cluster_mod.KMeans = _FakeKMeans  # type: ignore[attr-defined]
    parent = types.ModuleType("sklearn")
    parent.cluster = cluster_mod  # type: ignore[attr-defined]
    return parent, cluster_mod


@pytest.fixture
def diarize_module():
    sys.modules.pop("encocoa.diarize", None)
    return importlib.import_module("encocoa.diarize")


# ---- pure logic -----------------------------------------------------------


def test_label_by_first_appearance_uses_time_order(diarize_module) -> None:
    raw = [
        (5.0, 6.0, "spkX"),  # appears second (5s)
        (0.0, 1.0, "spkY"),  # appears first (0s) -> A
        (2.0, 3.0, "spkY"),
        (7.0, 8.0, "spkX"),
    ]
    out = diarize_module._label_by_first_appearance(raw)
    speakers = [s.speaker for s in out]
    assert speakers == ["A", "A", "B", "B"]
    assert [s.start for s in out] == [0.0, 2.0, 5.0, 7.0]


def test_coalesce_merges_same_speaker(diarize_module) -> None:
    Seg = diarize_module.DiarizationSegment
    segs = [
        Seg(0.0, 1.0, "A"),
        Seg(1.05, 2.0, "A"),  # gap 0.05 -> merged
        Seg(2.5, 3.0, "B"),
        Seg(3.0, 3.5, "B"),  # touching -> merged
        Seg(5.0, 6.0, "B"),  # gap 1.5 -> NOT merged
    ]
    out = diarize_module._coalesce(segs, gap_tolerance=0.2)
    assert [(s.start, s.end, s.speaker) for s in out] == [
        (0.0, 2.0, "A"),
        (2.5, 3.5, "B"),
        (5.0, 6.0, "B"),
    ]


def test_coalesce_empty(diarize_module) -> None:
    assert diarize_module._coalesce([]) == []


# ---- simple backend -------------------------------------------------------


def test_diarize_simple_clusters_and_labels(
    diarize_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import numpy as np

    # Two distinct embeddings; alternation across time. KMeans stub assigns
    # label 0 if first dim < 0 else 1.
    embeds = np.array(
        [
            [-1.0, 0.0],  # window @ 0–1.5s   -> cluster 0 (first appearance -> "A")
            [+1.0, 0.0],  # window @ 1.5–3.0s -> cluster 1                   -> "B"
            [-1.0, 0.0],  # window @ 3.0–4.5s -> cluster 0                   -> "A"
        ],
        dtype=float,
    )
    splits = [_Slice(0, 24000), _Slice(24000, 48000), _Slice(48000, 72000)]

    monkeypatch.setitem(sys.modules, "resemblyzer", _stub_resemblyzer_module(embeds, splits))
    sk_parent, sk_cluster = _stub_sklearn_module()
    monkeypatch.setitem(sys.modules, "sklearn", sk_parent)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", sk_cluster)

    result = diarize_module.diarize_simple(tmp_path / "fake.wav", num_speakers=2)
    speakers = [s.speaker for s in result]
    assert speakers == ["A", "B", "A"]
    assert result[0].start == pytest.approx(0.0)
    assert result[1].start == pytest.approx(1.5)
    assert result[2].start == pytest.approx(3.0)


def test_diarize_simple_too_few_windows_assigns_all_to_a(
    diarize_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import numpy as np

    embeds = np.array([[0.5, 0.5]], dtype=float)
    splits = [_Slice(0, 16000)]
    monkeypatch.setitem(sys.modules, "resemblyzer", _stub_resemblyzer_module(embeds, splits))
    sk_parent, sk_cluster = _stub_sklearn_module()
    monkeypatch.setitem(sys.modules, "sklearn", sk_parent)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", sk_cluster)

    result = diarize_module.diarize_simple(tmp_path / "fake.wav", num_speakers=2)
    assert [s.speaker for s in result] == ["A"]


# ---- pyannote backend -----------------------------------------------------


def test_diarize_pyannote_requires_token(
    diarize_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pyannote = types.ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = object  # type: ignore[attr-defined]
    parent = types.ModuleType("pyannote")
    parent.audio = fake_pyannote  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", parent)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HuggingFace token"):
        diarize_module.diarize_pyannote(tmp_path / "x.wav", hf_token=None)


def test_diarize_pyannote_runs_pipeline(
    diarize_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Turn:
        def __init__(self, s, e):
            self.start, self.end = s, e

    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            assert yield_label
            yield _Turn(0.0, 1.0), None, "spk_red"
            yield _Turn(1.0, 2.0), None, "spk_blue"
            yield _Turn(2.0, 3.0), None, "spk_red"

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, use_auth_token=None):
            assert use_auth_token == "test-token"
            assert model_id == "pyannote/speaker-diarization-3.1"
            return cls()

        def __call__(self, audio_path, num_speakers=None):
            assert num_speakers == 2
            return _Annotation()

    fake_pyannote = types.ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = _FakePipeline  # type: ignore[attr-defined]
    parent = types.ModuleType("pyannote")
    parent.audio = fake_pyannote  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", parent)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)

    result = diarize_module.diarize_pyannote(
        tmp_path / "x.wav", num_speakers=2, hf_token="test-token"
    )
    speakers = [s.speaker for s in result]
    # spk_red appears first (t=0) -> A; spk_blue -> B
    assert speakers == ["A", "B", "A"]


# ---- io & summary ---------------------------------------------------------


def test_save_diarization_writes_schema(
    diarize_module, tmp_path: Path
) -> None:
    Seg = diarize_module.DiarizationSegment
    segs = [Seg(0.0, 1.0, "A"), Seg(1.0, 2.0, "B")]
    out = tmp_path / "session.diarization.json"
    diarize_module.save_diarization(segs, out)
    payload = json.loads(out.read_text())
    assert payload == [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "B"},
    ]


def test_diarization_summary(diarize_module) -> None:
    Seg = diarize_module.DiarizationSegment
    segs = [Seg(0, 2, "A"), Seg(2, 5, "B"), Seg(5, 7, "A")]
    out = diarize_module.diarization_summary(segs)
    assert out == {"A": 4.0, "B": 3.0}
