"""Voice activity detection.

Produces speech regions so that diarization and ASR only see speech, and so
timestamps stay aligned to the original recording — timestamps are evidence
(spec section 4), so every implementation reports times in the source timeline.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from audio.ingestion import AudioBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechRegion:
    start: float
    end: float
    energy: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class VoiceActivityDetector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, audio: AudioBuffer) -> list[SpeechRegion]:
        ...


class EnergyVAD(VoiceActivityDetector):
    """Adaptive short-time-energy VAD. No dependencies, works on telephony audio.

    Frame energy is compared against a noise floor estimated from the quietest
    frames, so it adapts to a noisy line instead of using a fixed threshold.
    """

    name = "energy"

    def __init__(
        self,
        frame_ms: int = 30,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        margin_db: float = 8.0,
    ) -> None:
        self.frame_ms = frame_ms
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.margin_db = margin_db

    def detect(self, audio: AudioBuffer) -> list[SpeechRegion]:
        samples = audio.samples
        if not samples:
            return []
        frame_len = max(1, int(audio.sample_rate * self.frame_ms / 1000))
        frames: list[float] = []
        for offset in range(0, len(samples), frame_len):
            window = samples[offset : offset + frame_len]
            if not window:
                continue
            rms = math.sqrt(sum(value * value for value in window) / len(window))
            frames.append(20 * math.log10(rms + 1e-9))

        if not frames:
            return []
        ordered = sorted(frames)
        noise_floor = ordered[max(0, int(len(ordered) * 0.15))]
        peak = ordered[max(0, int(len(ordered) * 0.95))]
        if peak - noise_floor < 6.0:
            # Effectively flat: either silence or constant tone. Treat as one region
            # rather than fabricating turn boundaries.
            return [SpeechRegion(0.0, audio.duration, peak)]
        threshold = noise_floor + self.margin_db

        voiced = [level > threshold for level in frames]
        regions: list[SpeechRegion] = []
        frame_seconds = frame_len / audio.sample_rate
        min_speech = self.min_speech_ms / 1000
        max_gap = self.min_silence_ms / 1000

        start: float | None = None
        silence = 0.0
        for index, is_voiced in enumerate(voiced):
            time = index * frame_seconds
            if is_voiced:
                if start is None:
                    start = time
                silence = 0.0
            elif start is not None:
                silence += frame_seconds
                if silence >= max_gap:
                    end = time - silence + frame_seconds
                    if end - start >= min_speech:
                        regions.append(SpeechRegion(start, end, threshold))
                    start, silence = None, 0.0
        if start is not None:
            end = len(voiced) * frame_seconds
            if end - start >= min_speech:
                regions.append(SpeechRegion(start, end, threshold))
        return regions


class SileroVAD(VoiceActivityDetector):
    """Silero VAD via torch.hub — better on noisy lines and cross-talk."""

    name = "silero"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = None
        self._utils = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # local import: torch is optional for the rest of the app

        self._model, self._utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )

    def detect(self, audio: AudioBuffer) -> list[SpeechRegion]:
        try:
            self._load()
            import torch

            get_timestamps = self._utils[0]
            tensor = torch.tensor(list(audio.samples), dtype=torch.float32)
            stamps = get_timestamps(
                tensor,
                self._model,
                sampling_rate=audio.sample_rate,
                threshold=self.threshold,
            )
            return [
                SpeechRegion(item["start"] / audio.sample_rate, item["end"] / audio.sample_rate)
                for item in stamps
            ]
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Silero VAD unavailable (%s); falling back to energy VAD", exc)
            return EnergyVAD().detect(audio)


def build_vad(backend: str | None = None) -> VoiceActivityDetector:
    from config.settings import settings

    choice = (backend or settings.audio.vad_backend).lower()
    return SileroVAD() if choice == "silero" else EnergyVAD()
