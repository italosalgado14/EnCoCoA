from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

SILENT_THRESHOLD_DBFS = -50.0


def peak_dbfs(wav_path: Path | str) -> float:
    """Return the peak amplitude of a WAV file in dBFS.

    Empty or all-zero audio returns -inf. Independent of the sounddevice/
    PortAudio backend so it can be called from `process` without requiring
    a working microphone stack.
    """
    data, _sr = sf.read(str(wav_path), dtype="int16", always_2d=False)
    arr = np.asarray(data)
    if arr.size == 0:
        return -math.inf
    peak = float(np.max(np.abs(arr))) / 32768.0
    if peak <= 0.0:
        return -math.inf
    return 20.0 * math.log10(peak)


def is_effectively_silent(
    wav_path: Path | str, *, threshold_dbfs: float = SILENT_THRESHOLD_DBFS
) -> bool:
    """True if the WAV's peak amplitude is below `threshold_dbfs`."""
    return peak_dbfs(wav_path) < threshold_dbfs
