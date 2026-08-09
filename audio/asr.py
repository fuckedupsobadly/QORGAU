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

    #: Whisper is multilingual, but this system only claims Kazakh and Russian.
    #: A short telephone clip is easily mis-detected as a third language, and the
    #: result is transliterated nonsense rather than a transcript.
    SUPPORTED_LANGUAGES = ("ru", "kk")

    def __init__(
        self,
        model_size: str | None = None,
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self.model_size = model_size or settings.audio.asr_model
        self.device = device
        self.compute_type = compute_type
        #: Pin a language when the line is known to be monolingual; leaving it
        #: unset keeps per-utterance detection, which code-switching needs.
        self.language = language or settings.audio.asr_language
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        device, compute = self.device, self.compute_type
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if device == "cpu" and compute in {"float16", "int8_float16"}:
            compute = "int8"  # float16 is not supported on CPU
        logger.info("loading faster-whisper %s on %s (%s)", self.model_size, device, compute)
        self._model = WhisperModel(self.model_size, device=device, compute_type=compute)

    def transcribe(self, audio: AudioBuffer, *, start_offset: float = 0.0) -> list[ASRResult]:
        self._load()
        import numpy as np

        samples = np.asarray(list(audio.samples), dtype="float32")
        options = dict(
            beam_size=5,
            vad_filter=False,           # QORGAU's own VAD already ran
            word_timestamps=False,
            condition_on_previous_text=True,
            task="transcribe",
        )
        #: Do not force a language by default — the caller may switch mid-sentence.
        segments, info = self._model.transcribe(samples, language=self.language, **options)

        if self.language is None and getattr(info, "language", None) not in self.SUPPORTED_LANGUAGES:
            # Whisper detected something this system does not support, which on a
            # short clip means it is about to transliterate rather than transcribe.
            # Re-run constrained to the likelier of the two languages we serve.
            # `info` is returned before the generator is consumed, so nothing is
            # decoded twice here.
            forced = self._best_supported(info)
            logger.info(
                "faster-whisper detected %r; re-transcribing as %r",
                getattr(info, "language", None), forced,
            )
            segments, info = self._model.transcribe(samples, language=forced, **options)

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

    def _best_supported(self, info) -> str:
        """Pick the most probable of the supported languages from Whisper's own scores."""
        probs = dict(getattr(info, "all_language_probs", None) or [])
        ranked = sorted(
            self.SUPPORTED_LANGUAGES, key=lambda code: probs.get(code, 0.0), reverse=True
        )
        return ranked[0]


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
