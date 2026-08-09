"""The `AudioProcessor` abstraction from spec section 4.

    audio → VAD → diarization → ASR → normalization → Transcript

Each stage is an injected component, so a stage can be replaced (energy VAD →
Silero, heuristic diarizer → pyannote, Whisper → a Kazakh-specific ASR) without
touching anything else.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from audio.asr import ASRResult, SpeechRecognizer, build_asr
from audio.diarization import Diarizer, DiarizedRegion, build_diarizer
from audio.ingestion import AudioBuffer, AudioSource, FixtureSource, open_source
from audio.vad import VoiceActivityDetector, build_vad
from config.ontology import Language, Speaker
from models.fraud_llm.lexicon import detect_language
from transcription.normalizer import TranscriptNormalizer
from transcription.schemas import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class AudioProcessor:
    """Turns audio into a structured, normalized, speaker-attributed transcript."""

    vad: VoiceActivityDetector | None = None
    diarizer: Diarizer | None = None
    asr: SpeechRecognizer | None = None
    normalizer: TranscriptNormalizer | None = None

    def __post_init__(self) -> None:
        self.vad = self.vad or build_vad()
        self.diarizer = self.diarizer or build_diarizer()
        self.asr = self.asr or build_asr()
        self.normalizer = self.normalizer or TranscriptNormalizer()

    # ------------------------------------------------------------------
    def process(
        self,
        audio: AudioBuffer | AudioSource | str | Path,
        *,
        call_id: str | None = None,
        call_direction: str = "unknown",
    ) -> Transcript:
        """Full-call analysis. Accepts a buffer, a source, or a path."""
        source: AudioSource | None = None
        if isinstance(audio, (str, Path)):
            source = open_source(audio)
            buffer = source.read()
        elif isinstance(audio, AudioSource):
            source = audio
            buffer = audio.read()
        else:
            buffer = audio

        call_id = call_id or f"call_{uuid.uuid4().hex[:10]}"

        # Pre-transcribed fixtures skip VAD/diarization: the turn structure and
        # speaker labels are already known.
        if isinstance(source, FixtureSource) or (buffer.metadata or {}).get("fixture"):
            return self._from_fixture(buffer, call_id=call_id, call_direction=call_direction)

        regions = self.vad.detect(buffer)
        if not regions:
            logger.info("no speech detected in %s", buffer.source)
            return Transcript(call_id=call_id, call_direction=call_direction, audio_path=buffer.source)

        diarized = self.diarizer.diarize(buffer, regions, call_direction=call_direction)
        segments: list[TranscriptSegment] = []
        for turn in diarized:
            clip = buffer.slice_seconds(turn.start, turn.end)
            for result in self.asr.transcribe(clip, start_offset=turn.start):
                segments.append(self._to_segment(len(segments), result, turn))

        if regions and not segments and self.asr.name == "fixture":
            # Speech was found but no engine could transcribe it. Say so — an
            # empty transcript here otherwise reads as "this call was silent",
            # which would be a dangerous thing to report about a scam call.
            raise RuntimeError(
                f"{len(regions)} speech region(s) were detected in {buffer.source!r} but no "
                "speech-recognition engine is installed, so the call cannot be transcribed. "
                "Install faster-whisper (see requirements-ml.txt) or set QORGAU_ASR explicitly."
            )

        transcript = Transcript(
            call_id=call_id,
            segments=segments,
            call_direction=call_direction,
            audio_path=buffer.source,
            notes=[
                f"vad={self.vad.name}",
                f"diarization={self.diarizer.name}",
                f"asr={self.asr.name}",
                f"speech_regions={len(regions)}",
            ],
        )
        return transcript

    # ------------------------------------------------------------------
    def process_incremental(
        self,
        source: AudioSource,
        *,
        call_id: str | None = None,
        call_direction: str = "unknown",
    ) -> Iterator[Transcript]:
        """Yield a growing transcript, one analysis window at a time.

        This is what makes real-time detection possible: the caller re-runs the
        LLM and risk engine on each yielded transcript instead of waiting for the
        call to end (spec section 21).
        """
        call_id = call_id or f"call_{uuid.uuid4().hex[:10]}"
        transcript = Transcript(call_id=call_id, call_direction=call_direction)
        elapsed = 0.0
        for chunk in source.stream():
            regions = self.vad.detect(chunk) or []
            if regions:
                diarized = self.diarizer.diarize(chunk, regions, call_direction=call_direction)
                for turn in diarized:
                    clip = chunk.slice_seconds(turn.start, turn.end)
                    for result in self.asr.transcribe(clip, start_offset=elapsed + turn.start):
                        shifted = DiarizedRegion(
                            elapsed + turn.start, elapsed + turn.end, turn.speaker,
                            turn.cluster, turn.confidence,
                        )
                        transcript.segments.append(
                            self._to_segment(len(transcript.segments), result, shifted)
                        )
            elapsed += chunk.duration
            yield transcript.model_copy(deep=True)

    # ------------------------------------------------------------------
    def _to_segment(
        self, index: int, result: ASRResult, turn: DiarizedRegion
    ) -> TranscriptSegment:
        normalized = self.normalizer.normalize(result.text)
        notes = list(normalized.notes)
        if normalized.reverted:
            notes.extend(f"reverted: {item}" for item in normalized.reverted)
        if turn.confidence < 0.5:
            notes.append(f"low diarization confidence ({turn.confidence:.2f})")
        return TranscriptSegment(
            index=index,
            speaker=turn.speaker,
            start=result.start or turn.start,
            end=result.end or turn.end,
            language=_language(result.language, normalized.text),
            text=normalized.text,
            text_original=result.text,
            confidence=result.confidence,
            normalization_notes=notes,
        )

    def _from_fixture(
        self, buffer: AudioBuffer, *, call_id: str, call_direction: str
    ) -> Transcript:
        fixture = (buffer.metadata or {}).get("fixture") or {}
        segments: list[TranscriptSegment] = []
        for raw in fixture.get("segments", []):
            text = (raw.get("text") or "").strip()
            if not text:
                continue
            normalized = self.normalizer.normalize(text)
            segments.append(
                TranscriptSegment(
                    index=len(segments),
                    speaker=Speaker(raw.get("speaker", "UNKNOWN")),
                    start=float(raw.get("start", 0.0)),
                    end=float(raw.get("end", 0.0)),
                    language=_language(raw.get("language"), normalized.text),
                    text=normalized.text,
                    text_original=text,
                    confidence=float(raw.get("confidence", 1.0)),
                    normalization_notes=normalized.notes,
                )
            )
        return Transcript(
            call_id=fixture.get("call_id") or call_id,
            segments=segments,
            call_direction=fixture.get("call_direction") or call_direction,
            audio_path=buffer.source,
            notes=["source=fixture"],
        )


def _language(reported: str | None, text: str) -> Language:
    candidate = (reported or "").strip().lower()
    if candidate in {member.value for member in Language}:
        # Re-check for code-switching: a single-language tag is often wrong here.
        detected = detect_language(text)
        if detected == Language.MIXED.value:
            return Language.MIXED
        return Language(candidate)
    detected = detect_language(text)
    return Language(detected) if detected in {m.value for m in Language} else Language.UNKNOWN


# ---------------------------------------------------------------------------
# Convenience constructors used by the corpus tools, tests and the UI
# ---------------------------------------------------------------------------


def transcript_from_turns(
    turns: Sequence[dict],
    *,
    call_id: str | None = None,
    call_direction: str = "unknown",
    seconds_per_turn: float = 6.0,
    normalize: bool = True,
) -> Transcript:
    """Build a Transcript from `[{speaker, text, language?, confidence?, start?}]`.

    Timestamps are synthesised at a constant cadence when absent, which is how
    the synthetic corpus produces evidence-anchored timelines.
    """
    normalizer = TranscriptNormalizer()
    segments: list[TranscriptSegment] = []
    cursor = 1.5

    def value(turn: dict, key: str, fallback: float) -> float:
        """Tolerate a key that is present but null.

        Pydantic's `model_dump()` emits `{"start": None}` for unset optional
        fields, so `dict.get(key, default)` returns None rather than the default.
        """
        raw = turn.get(key)
        return fallback if raw is None else float(raw)

    for turn in turns:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        start = value(turn, "start", cursor)
        end = value(turn, "end", start + max(2.0, min(9.0, len(text) / 14)))
        result = normalizer.normalize(text) if normalize else None
        body = result.text if result else text
        segments.append(
            TranscriptSegment(
                index=len(segments),
                speaker=Speaker(turn.get("speaker", "UNKNOWN")),
                start=start,
                end=end,
                language=_language(turn.get("language"), body),
                text=body,
                text_original=text,
                confidence=max(0.0, min(1.0, value(turn, "confidence", 1.0))),
                normalization_notes=(result.notes if result else []),
            )
        )
        cursor = end + seconds_per_turn - max(2.0, min(9.0, len(text) / 14))
        cursor = max(cursor, end + 0.6)
    return Transcript(
        call_id=call_id or f"call_{uuid.uuid4().hex[:10]}",
        segments=segments,
        call_direction=call_direction,
    )
