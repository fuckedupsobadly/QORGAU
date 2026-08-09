"""Incremental analysis of a live call (spec section 21).

The loop is:

    new transcript chunk → LLM re-analyses the conversation SO FAR
      → risk engine re-scores → state updated → alert if a threshold crossed

Two properties matter and are enforced here:

* **Never classify a sentence in isolation.** Each analysis pass sees a context
  window of previous turns, not just the newest one.
* **Risk never silently drops.** A caller who asks for an OTP at 00:42 and then
  chats pleasantly is still dangerous at 01:30, so the session keeps a high-water
  mark and reports both the current and the peak score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Iterator

from config.ontology import RiskLevel, STAGE_DEPTH
from config.settings import settings
from models.inference import analyze_transcript
from realtime.alerts import AlertPolicy
from risk.engine import assess
from transcription.schemas import (
    Alert,
    CallAnalysis,
    ConversationState,
    LLMAnalysis,
    RiskAssessment,
    Transcript,
)


@dataclass
class SessionUpdate:
    """One incremental result — what the WebSocket streams and the UI renders."""

    call_id: str
    transcript: Transcript
    analysis: LLMAnalysis
    risk: RiskAssessment
    state: ConversationState
    new_events: list = field(default_factory=list)
    alert: Alert | None = None
    segments_added: int = 0

    def to_ws_payload(self) -> dict:
        """The `risk_update` frame from spec section 28."""
        payload: dict = {
            "type": "risk_update",
            "call_id": self.call_id,
            "risk_score": self.risk.risk_score,
            "classification": self.risk.risk_level.value,
            "model_classification": self.analysis.classification.value,
            "current_stage": self.state.current_stage,
            "detected_tactics": self.state.detected_tactics,
            "peak_risk_score": self.state.risk_score,
            "segments_seen": self.state.segments_seen,
        }
        if self.new_events:
            newest = self.new_events[-1]
            payload["event"] = {
                "category": newest.category,
                "severity": newest.severity.value,
                "timestamp": newest.timestamp,
                "evidence": newest.evidence,
            }
        if self.alert and self.alert.level != "NONE":
            payload["alert"] = self.alert.model_dump()
        return payload


class LiveCallSession:
    """Stateful wrapper around the stateless analysis stack."""

    def __init__(
        self,
        call_id: str,
        *,
        call_direction: str = "unknown",
        backend: str | None = None,
        context_window: int | None = None,
        alert_policy: AlertPolicy | None = None,
    ) -> None:
        self.call_id = call_id
        self.call_direction = call_direction
        self.backend = backend
        self.context_window = context_window or settings.model.context_window_turns
        self.policy = alert_policy or AlertPolicy()
        self.state = ConversationState(call_id=call_id)
        self.transcript = Transcript(call_id=call_id, call_direction=call_direction)
        self.history: list[SessionUpdate] = []
        self._last_analysis: LLMAnalysis | None = None
        self._last_risk: RiskAssessment | None = None

    # ------------------------------------------------------------------
    def ingest(self, transcript: Transcript) -> SessionUpdate:
        """Analyse the conversation as heard so far."""
        added = max(0, len(transcript.segments) - len(self.transcript.segments))
        self.transcript = transcript

        window = (
            transcript
            if len(transcript.segments) <= self.context_window
            else transcript.window(self.context_window)
        )
        analysis = analyze_transcript(window, realtime=True, backend=self.backend)
        risk = assess(analysis, window)

        new_events = self.state.merge_events(analysis.risk_factors)
        self.state.merge_stage(analysis.conversation_stage)
        # Peak-hold: an attack that already happened does not un-happen.
        self.state.risk_score = max(self.state.risk_score, risk.risk_score)
        self.state.classification = (
            risk.risk_level.value
            if risk.risk_score >= self.state.risk_score
            else RiskLevel(_band(self.state.risk_score)).value
        )
        for tactic in analysis.tactics:
            if tactic not in self.state.detected_tactics:
                self.state.detected_tactics.append(tactic)
        for scam_type in analysis.scam_types:
            if scam_type not in self.state.detected_scam_types:
                self.state.detected_scam_types.append(scam_type)
        self.state.segments_seen = len(transcript.segments)
        self.state.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        alert = self.policy.evaluate(risk, new_events, self.state)
        if alert and alert.level != "NONE":
            self.state.alerts.append(alert)

        update = SessionUpdate(
            call_id=self.call_id,
            transcript=transcript,
            analysis=analysis,
            risk=risk,
            state=self.state.model_copy(deep=True),
            new_events=new_events,
            alert=alert,
            segments_added=added,
        )
        self._last_analysis, self._last_risk = analysis, risk
        self.history.append(update)
        return update

    # ------------------------------------------------------------------
    def replay(
        self,
        transcripts: Iterable[Transcript],
        on_update: Callable[[SessionUpdate], None] | None = None,
    ) -> Iterator[SessionUpdate]:
        """Drive the session from a sequence of growing transcripts."""
        for transcript in transcripts:
            update = self.ingest(transcript)
            if on_update:
                on_update(update)
            yield update

    def finalize(self) -> CallAnalysis:
        """Full-call (non-realtime) analysis for the investigation report."""
        analysis = analyze_transcript(self.transcript, realtime=False, backend=self.backend)
        risk = assess(analysis, self.transcript)
        return CallAnalysis(
            call_id=self.call_id,
            transcript=self.transcript,
            analysis=analysis,
            risk=risk,
            stage_timeline=self.stage_timeline(),
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def stage_timeline(self) -> list[dict[str, str]]:
        """`[{timestamp, stage}]` — when the call entered each stage."""
        timeline: list[dict[str, str]] = []
        for update in self.history:
            stage = update.analysis.conversation_stage
            timestamp = (
                update.transcript.segments[-1].timestamp if update.transcript.segments else "00:00"
            )
            if not timeline or timeline[-1]["stage"] != stage:
                timeline.append({"timestamp": timestamp, "stage": stage})
        # Keep only forward progress: stage regressions are analysis noise.
        pruned: list[dict[str, str]] = []
        deepest = -1
        for entry in timeline:
            depth = STAGE_DEPTH.get(entry["stage"], 0)
            if depth >= deepest:
                pruned.append(entry)
                deepest = depth
        return pruned


def _band(score: int) -> str:
    from config.settings import risk_level

    return risk_level(score)


# ---------------------------------------------------------------------------
# Convenience: turn one finished transcript into a realistic replay
# ---------------------------------------------------------------------------


def growing_transcripts(transcript: Transcript, step: int = 1) -> Iterator[Transcript]:
    """Yield prefixes of a transcript, simulating a call arriving in real time."""
    for end in range(step, len(transcript.segments) + step, step):
        yield transcript.model_copy(update={"segments": transcript.segments[:end]}, deep=True)
