"""The deterministic risk engine.

Contract: `LLMAnalysis` (+ the transcript, for attribution checks) in, an
integer 0-100 score with a full audit trail out. No randomness, no model calls,
no hidden state — the same findings always produce the same score, and every
point is traceable to a named rule in `risk/rules.py`.

Combination rule
----------------
Group contributions are capped, then combined with a **soft sum**:

    score = 100 * (1 - Π (1 - cᵢ/100))

Each component still moves the score in a fixed, explainable way (its marginal
contribution is reported), but ten pressure signals can never add up to the same
weight as a completed money transfer. Policy **floors** then enforce spec
section 22: a CRITICAL request for credentials, money or device access is
treated as high risk regardless of how the additive terms land.
"""

from __future__ import annotations

from datetime import datetime, timezone

from config.ontology import (
    Classification,
    EventCategory,
    HARMFUL_ACTION_EVENTS,
    RiskLevel,
    SEVERITY_ORDER,
    Severity,
    Speaker,
)
from config.settings import (
    ALERT_CRITICAL_THRESHOLD,
    ALERT_MONITOR_THRESHOLD,
    ALERT_WARNING_THRESHOLD,
    risk_level,
)
from risk import rules
from transcription.schemas import (
    Alert,
    CallAnalysis,
    LLMAnalysis,
    RiskAssessment,
    RiskContribution,
    RiskFactor,
    Transcript,
)

#: Categories whose weight only counts when the CALLER is the one doing it.
#: "Не говорите никому код" from the victim must never add risk.
CALLER_ONLY_CATEGORIES = HARMFUL_ACTION_EVENTS | {
    EventCategory.IMPERSONATION.value,
    EventCategory.THREAT.value,
    EventCategory.SECRECY.value,
    EventCategory.ISOLATION.value,
    EventCategory.FALSE_PROBLEM.value,
    EventCategory.FALSE_SOLUTION.value,
    EventCategory.PERSONAL_DATA_REQUEST.value,
}


def _soft_sum(components: list[float]) -> float:
    """Diminishing-returns combination; never exceeds 100."""
    remaining = 1.0
    for value in components:
        remaining *= max(0.0, 1.0 - max(0.0, value) / 100.0)
    return 100.0 * (1.0 - remaining)


class RiskEngine:
    """Turns model findings into an auditable production risk score."""

    def assess(self, analysis: LLMAnalysis, transcript: Transcript) -> RiskAssessment:
        contributions: list[RiskContribution] = []
        scored, rejected = self._partition(analysis.risk_factors)
        contributions.extend(rejected)

        components: list[float] = []
        group_totals: dict[str, float] = {}

        # -- 1. per-event weights, aggregated per group and capped -------
        for group_name, members in rules.GROUPS.items():
            group_events = [e for e in scored if e.category in members]
            if not group_events:
                continue
            raw = 0.0
            for event in sorted(
                group_events,
                key=lambda e: -rules.event_points(e.category, e.severity.value),
            ):
                points = rules.event_points(event.category, event.severity.value)
                if points <= 0:
                    continue
                raw += points
                contributions.append(
                    RiskContribution(
                        kind="event",
                        label=f"{event.category} ({event.severity.value})",
                        points=round(points, 1),
                        detail=f"{rules.GROUP_LABELS[group_name]} — {event.reason}",
                        evidence=event.evidence,
                        timestamp=event.timestamp,
                    )
                )
            cap = rules.GROUP_CAPS[group_name]
            capped = min(raw, cap)
            if raw > cap:
                contributions.append(
                    RiskContribution(
                        kind="group_cap",
                        label=f"{group_name} capped",
                        points=round(capped - raw, 1),
                        detail=(
                            f"{rules.GROUP_LABELS[group_name]} raw total {raw:.1f} capped at {cap:.0f} "
                            "— repeating the same kind of signal does not multiply the danger."
                        ),
                    )
                )
            group_totals[group_name] = capped
            components.append(capped)

        active_groups = {g for g, v in group_totals.items() if v > 0}
        present_categories = {e.category for e in scored}

        # -- 2. interaction effects --------------------------------------
        for group_a, group_b, points, why in rules.GROUP_INTERACTIONS:
            if group_a in active_groups and group_b in active_groups:
                components.append(points)
                contributions.append(
                    RiskContribution(
                        kind="interaction",
                        label=f"{group_a} + {group_b}",
                        points=float(points),
                        detail=why,
                    )
                )
        for cat_a, cat_b, points, why in rules.CATEGORY_INTERACTIONS:
            if cat_a in present_categories and cat_b in present_categories:
                components.append(points)
                contributions.append(
                    RiskContribution(
                        kind="interaction",
                        label=f"{cat_a} + {cat_b}",
                        points=float(points),
                        detail=why,
                    )
                )

        # -- 3. the model's own verdict (bounded) ------------------------
        # Only counted when the model produced at least one grounded finding. A
        # verdict with no traceable evidence behind it must not move the score —
        # otherwise a hallucinated SCAM label would raise risk on its own.
        verdict_points = rules.VERDICT_POINTS.get(analysis.classification.value, 0.0)
        if scored and verdict_points and analysis.confidence >= rules.VERDICT_MIN_CONFIDENCE:
            components.append(verdict_points)
            contributions.append(
                RiskContribution(
                    kind="event",
                    label=f"model verdict {analysis.classification.value}",
                    points=verdict_points,
                    detail=(
                        f"Contextual model reports {analysis.classification.value} at confidence "
                        f"{analysis.confidence:.2f}. Bounded contribution — the score is driven by "
                        "the evidence, not the verdict."
                    ),
                )
            )

        subtotal = _soft_sum(components)

        # -- 4. dampening for unreliable input --------------------------
        if transcript.mean_confidence < rules.LOW_CONFIDENCE_ASR_THRESHOLD and subtotal > 0:
            damped = subtotal * rules.LOW_CONFIDENCE_DAMPENING
            contributions.append(
                RiskContribution(
                    kind="dampening",
                    label="low ASR confidence",
                    points=round(damped - subtotal, 1),
                    detail=(
                        f"Mean ASR confidence {transcript.mean_confidence:.2f} is below "
                        f"{rules.LOW_CONFIDENCE_ASR_THRESHOLD}; the evidence itself is uncertain."
                    ),
                )
            )
            subtotal = damped

        # -- 5. mitigations ---------------------------------------------
        harmful = {e.category for e in scored} & HARMFUL_ACTION_EVENTS
        if not harmful:
            protective = [
                e for e in analysis.risk_factors
                if e.category == EventCategory.PROTECTIVE_ADVICE.value
            ]
            if protective:
                subtotal -= rules.PROTECTIVE_ADVICE_MITIGATION
                contributions.append(
                    RiskContribution(
                        kind="mitigation",
                        label="protective advice present",
                        points=-float(rules.PROTECTIVE_ADVICE_MITIGATION),
                        detail=(
                            "A speaker explicitly warned against disclosing credentials or pointed to "
                            "an official channel, and no harmful action was requested."
                        ),
                        evidence=protective[0].evidence,
                        timestamp=protective[0].timestamp,
                    )
                )
            if transcript.call_direction == "inbound":
                subtotal -= rules.INBOUND_CALL_MITIGATION
                contributions.append(
                    RiskContribution(
                        kind="mitigation",
                        label="customer-initiated call",
                        points=-float(rules.INBOUND_CALL_MITIGATION),
                        detail=(
                            "The victim placed this call, so the counterparty was chosen by them "
                            "rather than by an unknown caller."
                        ),
                    )
                )

        score = max(0.0, min(100.0, subtotal))

        # -- 6. policy floors ------------------------------------------
        floor, floor_reason, floor_event = self._floor(scored, analysis)
        if floor > score:
            contributions.append(
                RiskContribution(
                    kind="floor",
                    label=f"policy floor {floor}",
                    points=round(floor - score, 1),
                    detail=floor_reason,
                    evidence=floor_event.evidence if floor_event else "",
                    timestamp=floor_event.timestamp if floor_event else "",
                )
            )
            score = float(floor)

        final = int(round(score))
        level = RiskLevel(risk_level(final))
        alert = self.alert_for(final, level, scored)
        return RiskAssessment(
            risk_score=final,
            risk_level=level,
            contributions=contributions,
            alert=alert,
            disagreement=self._disagreement(analysis, final),
            explanation=self._explain(final, level, group_totals, scored),
        )

    # ------------------------------------------------------------------
    def _partition(self, factors: list[RiskFactor]) -> tuple[list[RiskFactor], list[RiskContribution]]:
        """Keep scoreable findings; record why the rest were excluded."""
        scored: list[RiskFactor] = []
        rejected: list[RiskContribution] = []
        for factor in factors:
            if factor.category == EventCategory.PROTECTIVE_ADVICE.value:
                continue  # handled as a mitigation, never as positive risk
            if not factor.is_grounded:
                rejected.append(
                    RiskContribution(
                        kind="mitigation",
                        label=f"{factor.category} ignored",
                        points=0.0,
                        detail="Finding was not traceable to the transcript, so it is not scored.",
                        evidence=factor.evidence,
                        timestamp=factor.timestamp,
                    )
                )
                continue
            if factor.category in CALLER_ONLY_CATEGORIES and factor.speaker != Speaker.CALLER:
                rejected.append(
                    RiskContribution(
                        kind="mitigation",
                        label=f"{factor.category} not scored",
                        points=0.0,
                        detail=(
                            f"Attributed to {factor.speaker.value}, not the caller. The same words "
                            "from the victim (or a warning) are not an attack."
                        ),
                        evidence=factor.evidence,
                        timestamp=factor.timestamp,
                    )
                )
                continue
            scored.append(factor)
        return scored, rejected

    def _floor(
        self, scored: list[RiskFactor], analysis: LLMAnalysis
    ) -> tuple[int, str, RiskFactor | None]:
        harmful = [e for e in scored if e.category in HARMFUL_ACTION_EVENTS]
        if not harmful:
            return 0, "", None
        categories = {e.category for e in scored}
        framed = bool(
            categories & {EventCategory.IMPERSONATION.value, EventCategory.FALSE_PROBLEM.value}
        )
        critical = [e for e in harmful if e.severity == Severity.CRITICAL]
        if critical and framed:
            return (
                rules.FLOOR_CRITICAL_WITH_CHAIN,
                "A critical request for credentials, money movement or device access was made inside "
                "a false-authority or invented-problem frame. Policy treats this as critical "
                "regardless of the additive score.",
                critical[0],
            )
        if critical:
            return (
                rules.FLOOR_CRITICAL_ALONE,
                "A critical request for credentials, money movement or device access was made. Even "
                "without a full attack frame this warrants an immediate warning.",
                critical[0],
            )
        high = [e for e in harmful if e.severity == Severity.HIGH]
        if high and framed:
            return (
                rules.FLOOR_HIGH_HARMFUL_WITH_CHAIN,
                "The caller asked the victim to hand over credentials, money or device control inside "
                "a false-authority or invented-problem frame.",
                high[0],
            )
        if high:
            return (
                rules.FLOOR_HIGH_HARMFUL,
                "The caller asked the victim to take an action that transfers money, credentials or "
                "device control.",
                high[0],
            )
        return 0, "", None

    def alert_for(
        self, score: int, level: RiskLevel, scored: list[RiskFactor]
    ) -> Alert:
        """Spec section 22, including immediate escalation on a critical event."""
        critical_events = [
            e
            for e in scored
            if e.category in HARMFUL_ACTION_EVENTS and e.severity == Severity.CRITICAL
        ]
        if critical_events:
            worst = critical_events[0]
            return Alert(
                level="CRITICAL",
                headline="Stop — do not follow the caller's instructions",
                detail=(
                    f"{worst.category.replace('_', ' ').title()} detected at {worst.timestamp}. "
                    "End the call and contact the bank yourself using the number on your card."
                ),
                triggered_by=worst.category,
                timestamp=worst.timestamp,
                risk_score=score,
            )
        if score >= ALERT_CRITICAL_THRESHOLD:
            return Alert(
                level="CRITICAL",
                headline="Critical fraud warning",
                detail="Multiple independent fraud indicators are present. End the call.",
                triggered_by="risk_score",
                risk_score=score,
            )
        if score >= ALERT_WARNING_THRESHOLD:
            return Alert(
                level="WARNING",
                headline="Suspicious call",
                detail=(
                    "This call matches a social-engineering pattern. Do not share codes or move "
                    "money; verify independently."
                ),
                triggered_by="risk_score",
                risk_score=score,
            )
        if score >= ALERT_MONITOR_THRESHOLD:
            return Alert(
                level="MONITOR",
                headline="Monitoring",
                detail="Some pressure or problem-framing signals are present. Continuing to monitor.",
                triggered_by="risk_score",
                risk_score=score,
            )
        return Alert(
            level="NONE",
            headline="No fraud indicators",
            detail="",
            triggered_by="risk_score",
            risk_score=score,
        )

    def _disagreement(self, analysis: LLMAnalysis, score: int) -> str | None:
        """Surface model/engine conflicts rather than silently resolving them."""
        level = risk_level(score)
        if analysis.classification == Classification.SCAM and score < ALERT_WARNING_THRESHOLD:
            return (
                f"The contextual model reports SCAM (confidence {analysis.confidence:.2f}) but the "
                f"deterministic score is only {score} ({level}): the grounded evidence is thinner "
                "than the verdict. Treat as suspicious and review manually."
            )
        if analysis.classification == Classification.SAFE and score >= ALERT_WARNING_THRESHOLD:
            return (
                f"The contextual model reports SAFE but the deterministic score is {score} ({level}) "
                "from grounded events. The rule engine wins for alerting; this call should be "
                "reviewed and, if the model is wrong, added to the training set."
            )
        if analysis.dropped_findings:
            return (
                f"{len(analysis.dropped_findings)} model finding(s) were discarded as ungrounded and "
                "did not affect the score."
            )
        return None

    def _explain(
        self,
        score: int,
        level: RiskLevel,
        group_totals: dict[str, float],
        scored: list[RiskFactor],
    ) -> str:
        if not scored:
            return (
                f"Risk {score}/100 ({level.value}). No scoreable fraud events were grounded in the "
                "transcript."
            )
        ordered = sorted(group_totals.items(), key=lambda kv: -kv[1])
        drivers = ", ".join(
            f"{rules.GROUP_LABELS[name].lower()} ({value:.0f} pts capped)"
            for name, value in ordered
            if value > 0
        )
        worst = max(scored, key=lambda e: SEVERITY_ORDER.get(e.severity.value, 0))
        return (
            f"Risk {score}/100 ({level.value}). Driven by {drivers}. The most severe single event is "
            f"{worst.category} at {worst.timestamp} ({worst.severity.value}). "
            "The score is a system risk estimate from the rules in risk/rules.py, not a calibrated "
            "probability."
        )


#: Module-level singleton — the engine is stateless.
engine = RiskEngine()


def assess(analysis: LLMAnalysis, transcript: Transcript) -> RiskAssessment:
    return engine.assess(analysis, transcript)


# ---------------------------------------------------------------------------
# Final investigation report
# ---------------------------------------------------------------------------


def render_report(result: CallAnalysis) -> str:
    """Markdown investigation report for a completed call."""
    a, r, t = result.analysis, result.risk, result.transcript
    bar_filled = int(round(r.risk_score / 5))
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    lines = [
        f"# QORGAU investigation report — call `{result.call_id}`",
        "",
        f"*Generated {result.generated_at or datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"· analysis backend `{a.model_backend or 'unknown'}`*",
        "",
        "## Verdict",
        "",
        f"| | |",
        f"|---|---|",
        f"| **Risk** | `{bar}` **{r.risk_score}/100** — {r.risk_level.value} |",
        f"| **Model classification** | {a.classification.value} (confidence {a.confidence:.2f}) |",
        f"| **Scam types** | {', '.join(a.scam_types) or '—'} |",
        f"| **Tactics** | {', '.join(a.tactics) or '—'} |",
        f"| **Conversation stage** | {a.conversation_stage} |",
        f"| **Duration** | {int(t.duration)}s, {len(t.segments)} segments |",
        f"| **Languages** | {t.dominant_language} {t.language_profile} |",
        f"| **Call direction** | {t.call_direction} |",
        f"| **Mean ASR confidence** | {t.mean_confidence:.2f} |",
        "",
        "## Recommended action",
        "",
        f"> {a.recommended_action}",
        "",
        "## Why",
        "",
        a.explanation,
        "",
        r.explanation,
    ]
    if r.disagreement:
        lines += ["", "> ⚠️ **Model / rule-engine disagreement:** " + r.disagreement]

    if a.requested_actions:
        lines += ["", "## Actions the caller requested", ""]
        lines += [f"- {action}" for action in a.requested_actions]

    lines += ["", "## Evidence timeline", "", "| Time | Speaker | Event | Severity | Evidence |", "|---|---|---|---|---|"]
    for event in sorted(a.risk_factors, key=lambda e: e.timestamp):
        evidence = event.evidence.replace("|", "\\|")
        lines.append(
            f"| `{event.timestamp}` | {event.speaker.value} | {event.category} | "
            f"{event.severity.value} | {evidence} |"
        )

    lines += ["", "## Risk score breakdown", "", "| Points | Rule | Detail |", "|---|---|---|"]
    for contribution in sorted(r.contributions, key=lambda c: -abs(c.points)):
        detail = contribution.detail.replace("|", "\\|")
        lines.append(f"| {contribution.points:+.1f} | {contribution.label} | {detail} |")

    if a.dropped_findings:
        lines += ["", "## Discarded findings (evidence not found in transcript)", ""]
        lines += [f"- {item}" for item in a.dropped_findings]

    lines += ["", "## Transcript", "", "```text", t.render(), "```"]
    return "\n".join(lines)
