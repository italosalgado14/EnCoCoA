from __future__ import annotations

import math
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float


def list_input_devices() -> list[InputDevice]:
    devices = sd.query_devices()
    out: list[InputDevice] = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            out.append(
                InputDevice(
                    index=i,
                    name=str(d.get("name", "")),
                    max_input_channels=int(d["max_input_channels"]),
                    default_samplerate=float(d.get("default_samplerate", 0.0) or 0.0),
                )
            )
    return out


def default_input_index() -> int | None:
    raw = sd.default.device
    # sounddevice exposes a `_InputOutputPair` here that supports indexing
    # like a 2-tuple; a plain int/str/None is also possible after `sd.default.device = ...`.
    try:
        idx = raw[0]
    except (TypeError, IndexError, KeyError):
        idx = raw
    if idx is None or idx == -1:
        return None
    try:
        return int(idx)
    except (TypeError, ValueError):
        return None


def resolve_input_device(spec: int | str | None) -> int | None:
    """Resolve a `--device` value to an integer index.

    Accepts:
      - None or empty string  → returns None (use system default)
      - int                   → returns it as-is (no validation)
      - all-digit string      → parsed as int
      - other string          → case-insensitive substring match against the
                                names of input-capable devices

    Raises ValueError if a name string does not match any input device, or if
    it matches more than one (the user must disambiguate).
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    s = str(spec).strip()
    if not s:
        return None
    if s.lstrip("-").isdigit():
        return int(s)

    needle = s.lower()
    devices = list_input_devices()
    matches = [d for d in devices if needle in d.name.lower()]
    if not matches:
        avail = ", ".join(f"[{d.index}] {d.name}" for d in devices) or "(none)"
        raise ValueError(
            f"No input device matches {spec!r}. Available: {avail}"
        )
    if len(matches) > 1:
        listing = "; ".join(f"[{m.index}] {m.name}" for m in matches)
        raise ValueError(
            f"Multiple input devices match {spec!r}: {listing}. "
            "Use a more specific name or pass the integer index."
        )
    return matches[0].index


def print_input_devices(stream=None) -> None:
    out = stream if stream is not None else sys.stdout
    default_in = default_input_index()
    devices = list_input_devices()
    if not devices:
        print("No input-capable devices found.", file=out)
        return
    print("Available input devices:", file=out)
    for d in devices:
        marker = "  [default]" if d.index == default_in else ""
        print(
            f"  [{d.index:2d}] {d.name}  ({d.max_input_channels} ch, "
            f"{d.default_samplerate:.0f} Hz){marker}",
            file=out,
        )


def _format_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _level_bar(rms: float, width: int = 24) -> str:
    if rms <= 0.0 or not math.isfinite(rms):
        db = -120.0
    else:
        db = 20.0 * math.log10(rms)
    norm = max(0.0, min(1.0, (db + 60.0) / 60.0))
    filled = int(round(norm * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {db:6.1f} dBFS"


def _rms_int16(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    x = chunk.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


def record_wav(
    out_path: Path | str,
    duration: float,
    samplerate: int = 16000,
    device: int | None = None,
    block_seconds: float = 0.1,
    progress: bool = True,
    stream=None,
) -> Path:
    """Record `duration` seconds of mono int16 audio at `samplerate` to `out_path`.

    Returns the path written. On Ctrl+C, the partial recording is still flushed to disk.
    """
    out = Path(out_path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)

    out_stream = stream if stream is not None else sys.stdout
    blocksize = max(1, int(samplerate * block_seconds))
    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, _time_info, status):  # noqa: ANN001
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        q.put(indata.copy())

    if progress:
        print(
            f"Recording {_format_hms(duration)} @ {samplerate} Hz mono → {out}",
            file=out_stream,
        )
        print("Press Ctrl+C to stop early.", file=stream)

    start = time.monotonic()
    deadline = start + float(duration)
    interrupted = False

    with sf.SoundFile(
        str(out),
        mode="w",
        samplerate=samplerate,
        channels=1,
        subtype="PCM_16",
    ) as wav:
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="int16",
            blocksize=blocksize,
            device=device,
            callback=callback,
        ):
            try:
                while time.monotonic() < deadline:
                    try:
                        chunk = q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    wav.write(chunk)
                    if progress:
                        elapsed = time.monotonic() - start
                        rms = _rms_int16(chunk)
                        print(
                            f"\r{_format_hms(elapsed)} / {_format_hms(duration)}  "
                            f"{_level_bar(rms)}",
                            end="",
                            flush=True,
                            file=out_stream,
                        )
            except KeyboardInterrupt:
                interrupted = True

        # Drain anything still in the queue from the callback after the stream stopped.
        while True:
            try:
                chunk = q.get_nowait()
            except queue.Empty:
                break
            wav.write(chunk)

    if progress:
        print(file=stream)
        if interrupted:
            print("[audio] interrupted; partial recording saved.", file=stream)

    return out
