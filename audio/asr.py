"""Speech-to-text for Kazakh, Russian and code-switched speech.

Two facts drive the design:

* No mainstream ASR handles Kazakh-Russian code-switching well. Whisper returns a
  *single* language tag per segment, which is wrong for "Сіздің картаңыз
  заблокирована." QORGAU therefore ignores the model's language label and
  re-derives it per utterance with `lexicon.detect_language`, which can return
  `mixed`.
* Confidence must be preserved. A finding resting on a 0.4-confidence segment is
  weaker evidence, and the risk engine dampens the score accordingly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from audio.ingestion import AudioBuffer
from config.settings import settings
from models.fraud_llm.lexicon import detect_language

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    text: str
    confidence: float = 1.0
    language: str = "unknown"
    start: float = 0.0
    end: float = 0.0


class SpeechRecognizer(ABC):
    name = "base"

    @abstractmethod
    def transcribe(self, audio: AudioBuffer, *, start_offset: float = 0.0) -> list[ASRResult]:
        ...

    @staticmethod
    def relabel_language(text: str, model_language: str | None = None) -> str:
        """Our detector wins: it can say `mixed`, which Whisper cannot."""
        detected = detect_language(text)
        if detected != "unknown":
            return detected
        return model_language or "unknown"


class FasterWhisperASR(SpeechRecognizer):
    """CTranslate2 Whisper — the practical default for Kazakh + Russian."""

    name = "faster_whisper"

    def __init__(
        self,
        model_size: str | None = None,
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size or settings.audio.asr_model
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )

    def transcribe(self, audio: AudioBuffer, *, start_offset: float = 0.0) -> list[ASRResult]:
        self._load()
        import numpy as np

        segments, _info = self._model.transcribe(
            np.asarray(list(audio.samples), dtype="float32"),
            beam_size=5,
            vad_filter=False,           # QORGAU's own VAD already ran
            word_timestamps=False,
            condition_on_previous_text=True,
            #: Do not force a language — the caller may switch mid-sentence.
            task="transcribe",
        )
        results: list[ASRResult] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            # avg_logprob → a rough 0-1 confidence for the UI and risk dampening.
            confidence = max(0.05, min(1.0, float(2 ** getattr(segment, "avg_logprob", -0.5))))
            results.append(
                ASRResult(
                    text=text,
                    confidence=round(confidence, 3),
                    language=self.relabel_language(text, getattr(segment, "language", None)),
                    start=start_offset + float(segment.start or 0.0),
                    end=start_offset + float(segment.end or 0.0),
                )
            )
        return results


class WhisperASR(SpeechRecognizer):
    """Reference OpenAI Whisper implementation."""

    name = "whisper"

    def __init__(self, model_size: str | None = None) -> None:
        self.model_size = model_size or settings.audio.asr_model
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.model_size)

    def transcribe(self, audio: AudioBuffer, *, start_offset: float = 0.0) -> list[ASRResult]:
        self._load()
        import numpy as np

        payload = self._model.transcribe(
            np.asarray(list(audio.samples), dtype="float32"), verbose=False
        )
        results: list[ASRResult] = []
        for segment in payload.get("segments", []):
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            confidence = max(0.05, min(1.0, 1.0 - float(segment.get("no_speech_prob", 0.3))))
            results.append(
                ASRResult(
                    text=text,
                    confidence=round(confidence, 3),
                    language=self.relabel_language(text, payload.get("language")),
                    start=start_offset + float(segment.get("start", 0.0)),
                    end=start_offset + float(segment.get("end", 0.0)),
                )
            )
        return results


class FixtureASR(SpeechRecognizer):
    """Replays a pre-transcribed call (corpus, labelled sample, or demo)."""

    name = "fixture"

    def transcribe(self, audio: AudioBuffer, *, start_offset: float = 0.0) -> list[ASRResult]:
        fixture = (audio.metadata or {}).get("fixture") or {}
        results: list[ASRResult] = []
        for segment in fixture.get("segments", []):
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            results.append(
                ASRResult(
                    text=text,
                    confidence=float(segment.get("confidence", 1.0)),
                    language=segment.get("language") or self.relabel_language(text),
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                )
            )
        return results


def build_asr(backend: str | None = None) -> SpeechRecognizer:
    """Pick an ASR backend, preferring real ASR when its dependency is installed."""
    choice = (backend or settings.audio.asr_backend).lower()
    if choice == "fixture":
        return FixtureASR()
    if choice in {"auto", "faster_whisper"}:
        try:
            import faster_whisper  # noqa: F401

            return FasterWhisperASR()
        except ImportError:
            if choice == "faster_whisper":
                raise
    if choice in {"auto", "whisper"}:
        try:
            import whisper  # noqa: F401

            return WhisperASR()
        except ImportError:
            if choice == "whisper":
                raise
    logger.info("no ASR engine installed — using fixture ASR (pre-transcribed calls only)")
    return FixtureASR()
