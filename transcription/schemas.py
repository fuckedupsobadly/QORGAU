"""Pydantic contracts that every QORGAU component speaks.

These schemas are the seams of the architecture: the audio layer produces
`Transcript`, the fine-tuned LLM produces `LLMAnalysis`, and the risk engine
turns `LLMAnalysis` into `RiskAssessment`. Any component can be swapped as long
as it keeps producing the same object.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config.ontology import (
    Classification,
    EVENT_TO_STAGE,
    EVENT_TO_TACTIC,
    EventCategory,
    Language,
    RiskLevel,
    SEVERITY_ORDER,
    STAGE_DEPTH,
    ScamType,
    Severity,
    Speaker,
    Stage,
    Tactic,
    enum_values,
)

# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def fmt_ts(seconds: float) -> str:
    """Seconds -> `MM:SS` (the only timestamp format used in LLM output)."""
    seconds = max(0.0, float(seconds))
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def parse_ts(value: str) -> float:
    """`MM:SS` or `HH:MM:SS` -> seconds. Returns -1.0 when unparseable."""
    if not value:
        return -1.0
    parts = value.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return -1.0
    total = 0.0
    for num in nums:
        total = total * 60 + num
    return total


# ---------------------------------------------------------------------------
# Text comparison helpers (evidence grounding)
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\sЀ-ӿ]+", re.UNICODE)
_WS = re.compile(r"\s+")


def canon(text: str) -> str:
    """Case/punctuation/whitespace-insensitive form used for evidence matching."""
    lowered = (text or "").lower().replace("ё", "е")
    return _WS.sub(" ", _PUNCT.sub(" ", lowered)).strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, canon(a), canon(b)).ratio()


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


class TranscriptSegment(BaseModel):
    """One diarized, transcribed utterance.

    `text_original` is immutable evidence straight from the ASR. `text` is the
    normalized form the LLM reads. Both are kept so any finding can be traced
    back to what was actually said (spec section 6).
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, description="Position in the conversation, 0-based.")
    speaker: Speaker = Speaker.UNKNOWN
    start: float = Field(ge=0.0, description="Seconds from call start.")
    end: float = Field(ge=0.0)
    language: Language = Language.UNKNOWN
    text: str = Field(description="Normalized text (LLM input).")
    text_original: str = Field(default="", description="Raw ASR text — never modified.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="ASR confidence.")
    normalization_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _defaults(self) -> "TranscriptSegment":
        if not self.text_original:
            object.__setattr__(self, "text_original", self.text)
        if self.end < self.start:
            object.__setattr__(self, "end", self.start)
        return self

    @property
    def timestamp(self) -> str:
        return fmt_ts(self.start)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Transcript(BaseModel):
    """Structured transcript for a whole call (or the part heard so far)."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    #: "inbound" = the victim dialled (they chose who to call, much safer),
    #: "outbound" = an unknown party dialled the victim. Metadata, not a verdict.
    call_direction: Literal["inbound", "outbound", "unknown"] = "unknown"
    audio_path: str | None = None
    notes: list[str] = Field(default_factory=list)

    # -- derived ----------------------------------------------------------
    @property
    def duration(self) -> float:
        return max((s.end for s in self.segments), default=0.0)

    @property
    def language_profile(self) -> dict[str, float]:
        if not self.segments:
            return {}
        counts: dict[str, int] = {}
        for seg in self.segments:
            counts[seg.language.value] = counts.get(seg.language.value, 0) + 1
        total = len(self.segments)
        return {k: round(v / total, 3) for k, v in sorted(counts.items())}

    @property
    def dominant_language(self) -> str:
        profile = self.language_profile
        if not profile:
            return Language.UNKNOWN.value
        mixed_share = profile.get(Language.MIXED.value, 0.0)
        kk = profile.get(Language.KK.value, 0.0)
        ru = profile.get(Language.RU.value, 0.0)
        if mixed_share >= 0.25 or (kk > 0.2 and ru > 0.2):
            return Language.MIXED.value
        return max(profile, key=lambda k: profile[k])

    @property
    def mean_confidence(self) -> float:
        if not self.segments:
            return 1.0
        return sum(s.confidence for s in self.segments) / len(self.segments)

    def by_speaker(self, speaker: Speaker | str) -> list[TranscriptSegment]:
        target = Speaker(speaker)
        return [s for s in self.segments if s.speaker == target]

    def window(self, last_n: int) -> "Transcript":
        """The most recent `last_n` segments, as a Transcript (real-time context)."""
        return self.model_copy(update={"segments": self.segments[-last_n:]})

    def render(self, *, normalized: bool = True, with_confidence: bool = False) -> str:
        """The exact text block handed to the LLM."""
        lines: list[str] = []
        for seg in self.segments:
            body = seg.text if normalized else seg.text_original
            suffix = f" (asr={seg.confidence:.2f})" if with_confidence else ""
            lines.append(f"[{seg.timestamp}] {seg.speaker.value} ({seg.language.value}){suffix}: {body}")
        return "\n".join(lines)

    def locate(self, evidence: str, threshold: float = 0.72) -> tuple[int | None, float]:
        """Find the segment an evidence quote came from.

        Returns `(segment_index, score)`. Substring containment scores 1.0; else
        the best fuzzy ratio against any segment (original or normalized text).
        """
        needle = canon(evidence)
        if not needle:
            return None, 0.0
        best: tuple[int | None, float] = (None, 0.0)
        for seg in self.segments:
            for candidate in (seg.text, seg.text_original):
                hay = canon(candidate)
                if not hay:
                    continue
                if needle in hay or hay in needle:
                    return seg.index, 1.0
                score = difflib.SequenceMatcher(None, needle, hay).ratio()
                # Also try the best matching sub-window, so a short quote pulled
                # from a long utterance still matches.
                if score < threshold and len(hay) > len(needle):
                    match = difflib.SequenceMatcher(None, needle, hay).find_longest_match(
                        0, len(needle), 0, len(hay)
                    )
                    if match.size:
                        score = max(score, match.size / max(1, len(needle)))
                if score > best[1]:
                    best = (seg.index, score)
        return (best[0], best[1]) if best[1] >= threshold else (None, best[1])


# ---------------------------------------------------------------------------
# LLM output (spec section 17)
# ---------------------------------------------------------------------------


class RiskFactor(BaseModel):
    """A single suspicious event with mandatory transcript evidence."""

    model_config = ConfigDict(extra="ignore")

    timestamp: str = ""
    speaker: Speaker = Speaker.UNKNOWN
    category: str = ""
    severity: Severity = Severity.LOW
    evidence: str = ""
    reason: str = ""
    #: Filled in by the grounding pass — which transcript segment backs this up.
    segment_index: int | None = None
    grounding_score: float = 0.0

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if v and v not in set(enum_values(EventCategory)):
            raise ValueError(f"unknown event category: {v}")
        return v

    @property
    def is_grounded(self) -> bool:
        return self.segment_index is not None and bool(self.evidence.strip())

    @property
    def tactic(self) -> str | None:
        return EVENT_TO_TACTIC.get(self.category)


class LLMAnalysis(BaseModel):
    """Exactly the JSON the fine-tuned model must emit, plus provenance fields."""

    model_config = ConfigDict(extra="ignore")

    classification: Classification = Classification.SAFE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    scam_types: list[str] = Field(default_factory=list)
    tactics: list[str] = Field(default_factory=list)
    conversation_stage: str = Stage.UNKNOWN.value
    requested_actions: list[str] = Field(default_factory=list)
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    explanation: str = ""
    recommended_action: str = ""

    # provenance (not part of the trained output schema)
    model_backend: str = ""
    dropped_findings: list[str] = Field(default_factory=list)

    @field_validator("scam_types", mode="before")
    @classmethod
    def _clean_scam_types(cls, v: Any) -> list[str]:
        return _filter_labels(v, set(enum_values(ScamType)))

    @field_validator("tactics", mode="before")
    @classmethod
    def _clean_tactics(cls, v: Any) -> list[str]:
        return _filter_labels(v, set(enum_values(Tactic)))

    @field_validator("conversation_stage", mode="before")
    @classmethod
    def _clean_stage(cls, v: Any) -> str:
        candidate = str(v or "").strip().upper()
        return candidate if candidate in set(enum_values(Stage)) else Stage.UNKNOWN.value

    @property
    def event_categories(self) -> set[str]:
        return {rf.category for rf in self.risk_factors if rf.category}

    def grounded_factors(self) -> list[RiskFactor]:
        return [rf for rf in self.risk_factors if rf.is_grounded]

    def strongest(self, category: str) -> RiskFactor | None:
        matches = [rf for rf in self.risk_factors if rf.category == category]
        if not matches:
            return None
        return max(matches, key=lambda rf: SEVERITY_ORDER.get(rf.severity.value, 0))

    def public_json(self) -> dict[str, Any]:
        """The schema-exact payload (spec section 17), no provenance fields."""
        return {
            "classification": self.classification.value,
            "confidence": round(self.confidence, 2),
            "scam_types": self.scam_types,
            "tactics": self.tactics,
            "conversation_stage": self.conversation_stage,
            "requested_actions": self.requested_actions,
            "risk_factors": [
                {
                    "timestamp": rf.timestamp,
                    "speaker": rf.speaker.value,
                    "category": rf.category,
                    "severity": rf.severity.value,
                    "evidence": rf.evidence,
                    "reason": rf.reason,
                }
                for rf in self.risk_factors
            ],
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
        }


def _filter_labels(value: Any, allowed: set[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for item in value:
        label = str(item).strip().upper().replace(" ", "_")
        if label in allowed and label not in out:
            out.append(label)
    return out


# ---------------------------------------------------------------------------
# Risk engine output
# ---------------------------------------------------------------------------


class RiskContribution(BaseModel):
    """One auditable line item in the risk score."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["event", "group_cap", "interaction", "mitigation", "floor", "dampening"]
    label: str
    points: float
    detail: str = ""
    evidence: str = ""
    timestamp: str = ""


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["NONE", "MONITOR", "WARNING", "CRITICAL"]
    headline: str
    detail: str = ""
    triggered_by: str = ""
    timestamp: str = ""
    risk_score: int = 0


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    contributions: list[RiskContribution] = Field(default_factory=list)
    alert: Alert
    #: Set when the LLM's verdict and the deterministic score disagree — surfaced,
    #: never silently resolved.
    disagreement: str | None = None
    explanation: str = ""

    def top_contributions(self, n: int = 6) -> list[RiskContribution]:
        return sorted(self.contributions, key=lambda c: -abs(c.points))[:n]


# ---------------------------------------------------------------------------
# Combined result / real-time state
# ---------------------------------------------------------------------------


class ConversationState(BaseModel):
    """Rolling state maintained during a live call (spec section 21)."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    current_stage: str = Stage.UNKNOWN.value
    deepest_stage: str = Stage.UNKNOWN.value
    risk_score: int = 0
    classification: str = RiskLevel.SAFE.value
    detected_tactics: list[str] = Field(default_factory=list)
    detected_scam_types: list[str] = Field(default_factory=list)
    events: list[RiskFactor] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    segments_seen: int = 0
    updated_at: str = ""

    def merge_stage(self, stage: str) -> None:
        self.current_stage = stage
        if STAGE_DEPTH.get(stage, 0) >= STAGE_DEPTH.get(self.deepest_stage, 0):
            self.deepest_stage = stage

    def merge_events(self, events: Iterable[RiskFactor]) -> list[RiskFactor]:
        """Add events not already present; returns the newly added ones."""
        known = {(e.category, e.timestamp, canon(e.evidence)) for e in self.events}
        fresh: list[RiskFactor] = []
        for event in events:
            key = (event.category, event.timestamp, canon(event.evidence))
            if key in known:
                continue
            known.add(key)
            self.events.append(event)
            fresh.append(event)
        return fresh


class CallAnalysis(BaseModel):
    """Everything the API and UI need for one call."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    transcript: Transcript
    analysis: LLMAnalysis
    risk: RiskAssessment
    stage_timeline: list[dict[str, str]] = Field(default_factory=list)
    generated_at: str = ""

    def report_markdown(self) -> str:
        from risk.engine import render_report  # local import avoids a cycle

        return render_report(self)


# ---------------------------------------------------------------------------
# Robust JSON extraction from model output
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(raw: str) -> dict[str, Any]:
    """Pull the JSON object out of a model completion.

    The fine-tuned model is trained to emit bare JSON, but inference-time
    robustness matters more than punishing it: we strip fences and take the
    outermost balanced object. Raises `ValueError` when nothing parses, so the
    caller can count JSON-validity failures in evaluation.
    """
    if not raw or not raw.strip():
        raise ValueError("empty completion")
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in completion")
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise ValueError("completion JSON is not an object")
                return parsed
    raise ValueError("unbalanced JSON object in completion")
