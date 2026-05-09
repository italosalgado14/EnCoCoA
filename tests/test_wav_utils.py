from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from encocoa.wav_utils import (
    SILENT_THRESHOLD_DBFS,
    is_effectively_silent,
    peak_dbfs,
)


def _write_wav(path: Path, samples: np.ndarray, samplerate: int = 16000) -> None:
    sf.write(str(path), samples, samplerate, subtype="PCM_16")


def test_peak_dbfs_full_scale(tmp_path: Path) -> None:
    p = tmp_path / "loud.wav"
    samples = np.full(1000, 32767, dtype=np.int16)
    _write_wav(p, samples)
    # int16 peak 32767 / 32768 ≈ -0.000265 dBFS, effectively 0 dBFS.
    assert abs(peak_dbfs(p)) < 0.01


def test_peak_dbfs_silence_returns_neg_inf(tmp_path: Path) -> None:
    p = tmp_path / "silent.wav"
    _write_wav(p, np.zeros(1000, dtype=np.int16))
    assert peak_dbfs(p) == -math.inf


def test_peak_dbfs_low_level(tmp_path: Path) -> None:
    p = tmp_path / "low.wav"
    # ~ -60 dBFS: amplitude 32 of full-scale 32768
    _write_wav(p, np.full(1000, 32, dtype=np.int16))
    val = peak_dbfs(p)
    assert -65.0 < val < -55.0


def test_is_effectively_silent(tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    _write_wav(silent, np.zeros(500, dtype=np.int16))
    assert is_effectively_silent(silent) is True

    loud = tmp_path / "loud.wav"
    _write_wav(loud, np.full(500, 30000, dtype=np.int16))
    assert is_effectively_silent(loud) is False


def test_is_effectively_silent_custom_threshold(tmp_path: Path) -> None:
    p = tmp_path / "midlevel.wav"
    # ~ -40 dBFS
    _write_wav(p, np.full(500, 327, dtype=np.int16))
    assert is_effectively_silent(p, threshold_dbfs=-30.0) is True
    assert is_effectively_silent(p, threshold_dbfs=-50.0) is False


def test_silent_threshold_constant() -> None:
    assert SILENT_THRESHOLD_DBFS == -50.0
