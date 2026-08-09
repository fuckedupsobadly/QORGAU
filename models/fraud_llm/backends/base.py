"""The seam between QORGAU and whatever produces `LLMAnalysis`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from transcription.schemas import LLMAnalysis, Transcript


class FraudLLMBackend(ABC):
    """A fraud-analysis engine.

    Every backend takes a `Transcript` and returns an `LLMAnalysis` conforming to
    the schema in spec section 17. Backends never compute a risk score and never
    decide what to do about a call — that is the risk engine's job.
    """

    #: Short identifier recorded on every analysis for auditability.
    name: str = "base"

    @abstractmethod
    def analyze(self, transcript: Transcript, *, realtime: bool = False) -> LLMAnalysis:
        ...

    def warmup(self) -> None:  # pragma: no cover - optional hook
        """Load weights / open connections ahead of the first request."""

    # -- shared post-processing -----------------------------------------
    def ground(self, analysis: LLMAnalysis, transcript: Transcript) -> LLMAnalysis:
        """Attach every finding to a real transcript segment; drop the rest.

        This is the mechanical enforcement of "never invent evidence": a finding
        whose quote cannot be located in the transcript is removed from the
        analysis and recorded in `dropped_findings`.
        """
        kept = []
        dropped: list[str] = []
        for factor in analysis.risk_factors:
            if not factor.evidence.strip():
                dropped.append(f"{factor.category}: empty evidence")
                continue
            index, score = transcript.locate(factor.evidence)
            if index is None:
                dropped.append(
                    f"{factor.category} @ {factor.timestamp}: evidence not found in transcript "
                    f"(best match {score:.2f}) — {factor.evidence[:60]!r}"
                )
                continue
            segment = transcript.segments[index]
            factor.segment_index = index
            factor.grounding_score = round(score, 3)
            # Repair hallucinated timestamps/speakers from the matched segment.
            factor.timestamp = segment.timestamp
            factor.speaker = segment.speaker
            kept.append(factor)
        analysis.risk_factors = kept
        analysis.dropped_findings = dropped
        analysis.model_backend = self.name
        return analysis
