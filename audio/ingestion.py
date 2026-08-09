"""Audio ingestion: uploaded files, live WebRTC/SIP frames, or fixtures.

Every source produces the same thing — mono float32 PCM at
`settings.audio.sample_rate` — so nothing downstream knows or cares where the
call came from.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import wave
from abc import ABC, abstractmethod
from array import array
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from config.settings import settings

logger = logging.getLogger(__name__)


def _brief(exc: Exception) -> str:
    """A one-line reason, since decoder stderr is often several lines of noise."""
    if isinstance(exc, subprocess.CalledProcessError):
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        return detail[-1][:120] if detail else f"exit {exc.returncode}"
    return f"{type(exc).__name__}: {exc}"[:120]


@dataclass
class AudioBuffer:
    """Mono PCM with a sample rate. `samples` is float32 in [-1, 1]."""

    samples: Sequence[float]
    sample_rate: int = settings.audio.sample_rate
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate or 1)

    def slice_seconds(self, start: float, end: float) -> "AudioBuffer":
        a = max(0, int(start * self.sample_rate))
        b = min(len(self.samples), int(end * self.sample_rate))
        return AudioBuffer(self.samples[a:b], self.sample_rate, self.source, dict(self.metadata))

    def chunks(self, seconds: float | None = None) -> Iterator[tuple[float, "AudioBuffer"]]:
        """Yield `(start_time, chunk)` for real-time processing (spec section 21)."""
        window = int((seconds or settings.audio.chunk_seconds) * self.sample_rate)
        if window <= 0:
            return
        for offset in range(0, len(self.samples), window):
            yield offset / self.sample_rate, AudioBuffer(
                self.samples[offset : offset + window],
                self.sample_rate,
                self.source,
                dict(self.metadata),
            )


class AudioSource(ABC):
    """Where audio comes from."""

    @abstractmethod
    def read(self) -> AudioBuffer:
        """Whole-call audio (batch analysis)."""

    def stream(self) -> Iterator[AudioBuffer]:
        """Successive chunks (live analysis). Defaults to chunking `read()`."""
        for _, chunk in self.read().chunks():
            yield chunk


class FileSource(AudioSource):
    """An uploaded recording. WAV natively; anything else via ffmpeg."""

    def __init__(self, path: str | Path, sample_rate: int | None = None) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate or settings.audio.sample_rate

    def read(self) -> AudioBuffer:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.suffix.lower() == ".wav":
            try:
                samples, rate = self._read_wav(self.path)
            except ValueError as exc:
                # 24-bit or float WAV — common from call recorders. `wave` cannot
                # read it, but the external decoders can.
                logger.info("%s needs decoding: %s", self.path.name, exc)
            else:
                if rate != self.sample_rate:
                    samples = _resample_linear(samples, rate, self.sample_rate)
                return AudioBuffer(samples, self.sample_rate, source=str(self.path))
        return AudioBuffer(self._decode(), self.sample_rate, source=str(self.path))

    @staticmethod
    def _read_wav(path: Path) -> tuple[list[float], int]:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
        if width != 2:
            raise ValueError(f"only 16-bit PCM WAV is supported directly (got {width * 8}-bit)")
        pcm = array("h")
        pcm.frombytes(raw)
        if channels > 1:  # downmix to mono
            mono = [
                sum(pcm[i : i + channels]) / channels for i in range(0, len(pcm) - channels + 1, channels)
            ]
        else:
            mono = list(pcm)
        return [value / 32768.0 for value in mono], rate

    def _decode(self) -> list[float]:
        """Decode a compressed recording, trying each available decoder in turn.

        Uploads arrive as mp3/m4a/opus far more often than as 16-bit WAV, so this
        path is the common one, not the exception. A decoder returns None when it
        is not installed and raises when it cannot handle the file; either way the
        next decoder gets a turn, because one decoder rejecting a file does not
        mean the others will (codec support differs). Only when every decoder is
        exhausted does this fail, and it fails with a single actionable message.
        """
        unavailable: list[str] = []
        failures: list[str] = []
        for name, decoder in (
            ("ffmpeg", self._decode_with_ffmpeg),
            ("afconvert", self._decode_with_afconvert),   # ships with macOS
            ("soundfile", self._decode_with_soundfile),   # pip, bundles libsndfile
        ):
            try:
                samples = decoder()
            except Exception as exc:  # noqa: BLE001 — any decoder failure is just "next"
                failures.append(f"{name} ({_brief(exc)})")
                logger.info("%s could not decode %s: %s", name, self.path.name, exc)
                continue
            if samples is None:
                unavailable.append(name)
                continue
            logger.info("decoded %s with %s (%d samples)", self.path.name, name, len(samples))
            return samples

        if failures:
            raise RuntimeError(
                f"{self.path.name} could not be decoded — it may be corrupt, empty, or in an "
                f"unsupported codec. Decoders tried: {', '.join(failures)}."
            )
        raise RuntimeError(
            f"cannot decode {self.path.suffix} — no decoder is available (looked for "
            f"{', '.join(unavailable)}). Install ffmpeg (`brew install ffmpeg`) or run "
            "`pip install soundfile`, or upload 16-bit PCM WAV instead."
        )

    def _decode_with_ffmpeg(self) -> list[float] | None:
        if shutil.which("ffmpeg") is None:
            return None
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(self.path),
            "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1",
            "-ar", str(self.sample_rate), "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        pcm = array("h")
        pcm.frombytes(raw[: len(raw) - (len(raw) % 2)])
        return [value / 32768.0 for value in pcm]

    def _decode_with_afconvert(self) -> list[float] | None:
        """CoreAudio's converter — present on every macOS box, so nothing to install."""
        if shutil.which("afconvert") is None:
            return None
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "decoded.wav"
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", f"LEI16@{self.sample_rate}", "-c", "1",
                 str(self.path), str(target)],
                capture_output=True, check=True,
            )
            samples, rate = self._read_wav(target)
        return _resample_linear(samples, rate, self.sample_rate) if rate != self.sample_rate else samples

    def _decode_with_soundfile(self) -> list[float] | None:
        try:
            import soundfile
        except ImportError:
            return None
        data, rate = soundfile.read(str(self.path), dtype="float32", always_2d=True)
        mono = [float(sum(frame) / len(frame)) for frame in data]
        return _resample_linear(mono, rate, self.sample_rate) if rate != self.sample_rate else mono


class LiveFrameSource(AudioSource):
    """Live telephony: WebRTC / SIP media frames pushed in from the transport.

    The transport (aiortc data callback, a SIP RTP reader, a browser mic) calls
    `push()`; the pipeline calls `stream()`. Frames are buffered until a full
    analysis window is available, which is what makes incremental analysis
    possible without waiting for the call to end.
    """

    def __init__(self, sample_rate: int | None = None, window_seconds: float | None = None) -> None:
        self.sample_rate = sample_rate or settings.audio.sample_rate
        self.window = int((window_seconds or settings.audio.chunk_seconds) * self.sample_rate)
        self._pending: deque[float] = deque()
        self._closed = False
        self.total_samples = 0

    def push(self, frame: Sequence[float] | bytes) -> None:
        if isinstance(frame, (bytes, bytearray)):
            pcm = array("h")
            pcm.frombytes(bytes(frame)[: len(frame) - (len(frame) % 2)])
            frame = [value / 32768.0 for value in pcm]
        self._pending.extend(frame)
        self.total_samples += len(frame)

    def close(self) -> None:
        self._closed = True

    def read(self) -> AudioBuffer:
        return AudioBuffer(list(self._pending), self.sample_rate, source="live")

    def stream(self) -> Iterator[AudioBuffer]:
        while True:
            if len(self._pending) >= self.window:
                window = [self._pending.popleft() for _ in range(self.window)]
                yield AudioBuffer(window, self.sample_rate, source="live")
            elif self._closed:
                if self._pending:
                    tail = list(self._pending)
                    self._pending.clear()
                    yield AudioBuffer(tail, self.sample_rate, source="live")
                return
            else:
                return  # caller polls again once more frames have arrived


class FixtureSource(AudioSource):
    """A pre-transcribed call — the corpus, a labelled sample, or a demo.

    Lets the whole pipeline (normalizer → LLM → risk engine → UI) run with no
    audio dependencies, which is how the synthetic corpus is evaluated.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> AudioBuffer:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        duration = max((seg.get("end", 0.0) for seg in payload.get("segments", [])), default=0.0)
        return AudioBuffer(
            samples=[],
            sample_rate=settings.audio.sample_rate,
            source=str(self.path),
            metadata={"fixture": payload, "duration": duration},
        )


def _resample_linear(samples: Sequence[float], src_rate: int, dst_rate: int) -> list[float]:
    """Linear resampling — adequate for 8 kHz telephony → 16 kHz ASR input."""
    if src_rate == dst_rate or not samples:
        return list(samples)
    ratio = dst_rate / src_rate
    out_len = int(len(samples) * ratio)
    out: list[float] = []
    for i in range(out_len):
        pos = i / ratio
        left = int(pos)
        right = min(left + 1, len(samples) - 1)
        frac = pos - left
        out.append(samples[left] * (1 - frac) + samples[right] * frac)
    return out


def open_source(target: str | Path) -> AudioSource:
    """Pick a source from a path: `.json` fixtures vs real audio."""
    path = Path(target)
    if path.suffix.lower() == ".json":
        return FixtureSource(path)
    return FileSource(path)
