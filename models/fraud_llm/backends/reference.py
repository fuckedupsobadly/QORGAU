"""Reference behavioural analyser — the offline stand-in for the fine-tuned LLM.

Why this exists
---------------
QORGAU's contextual intelligence layer is a fine-tuned LLM (`local_adapter`).
Training it needs a GPU and the corpus in `datasets/`. This backend implements
the *same contract* (`Transcript -> LLMAnalysis`) with an explicit, inspectable
model of the social-engineering sequence, so that:

* the risk engine, API, UI, database and evaluator can be developed and tested
  end-to-end without weights,
* the fine-tuned model has a non-trivial baseline to beat on the held-out sets,
* the synthetic corpus can be sanity-checked (a labelled conversation the
  reference analyser reads completely differently is usually a labelling bug).

It is **not** a keyword classifier. A concept hit alone produces nothing. What
produces a finding is a *speech act by a specific speaker over a concept*, and
what produces a SCAM verdict is a *chain*:

    identity/authority claim -> invented problem -> pressure -> proposed
    solution -> request whose payoff is credentials, money or device access

The same lexical hit means opposite things depending on speaker and act:
"не сообщайте код никому" is PROTECTIVE_ADVICE, "продиктуйте код" is
OTP_REQUEST. Its weakness compared to a fine-tuned model is exactly what you
would expect — paraphrases outside the lexicon are missed, so recall on unseen
scam scripts is the metric that should improve after fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.ontology import (
    CREDENTIAL_EXTRACTION_EVENTS,
    Classification,
    DEVICE_ACCESS_EVENTS,
    EVENT_TO_STAGE,
    EVENT_TO_TACTIC,
    EventCategory,
    HARMFUL_ACTION_EVENTS,
    MONEY_MOVEMENT_EVENTS,
    PRESSURE_EVENTS,
    SEVERITY_ORDER,
    STAGE_DEPTH,
    ScamType,
    Severity,
    Speaker,
    Stage,
    Tactic,
)
from models.fraud_llm.lexicon import Clause, Concept, SpeechAct, analyse_clauses
from models.fraud_llm.backends.base import FraudLLMBackend
from transcription.schemas import LLMAnalysis, RiskFactor, Transcript, TranscriptSegment

# ---------------------------------------------------------------------------
# Static tables
# ---------------------------------------------------------------------------

ORG_CONCEPTS = (
    Concept.ORG_BANK,
    Concept.ORG_POLICE,
    Concept.ORG_INVESTIGATOR,
    Concept.ORG_GOVERNMENT,
    Concept.ORG_TELCO,
    Concept.ORG_DELIVERY,
    Concept.ORG_MARKETPLACE,
    Concept.ORG_CRYPTO,
    Concept.ORG_INVESTMENT,
    Concept.ORG_EMPLOYER,
    Concept.SECURITY_DEPT,
)

PROBLEM_CONCEPTS = (
    Concept.PROBLEM_LOAN,
    Concept.PROBLEM_TRANSACTION,
    Concept.PROBLEM_BLOCKED,
    Concept.PROBLEM_COMPROMISE,
    Concept.PROBLEM_LEGAL,
    Concept.PROBLEM_REFUND,
)

REQUESTED_ACTION_TEXT = {
    EventCategory.OTP_REQUEST.value: "Dictate the one-time SMS/push confirmation code to the caller",
    EventCategory.CARD_REQUEST.value: "Disclose full card number, expiry date or CVV/CVC",
    EventCategory.PASSWORD_REQUEST.value: "Disclose account password, login or PIN",
    EventCategory.PERSONAL_DATA_REQUEST.value: "Disclose personal identification data (IIN, ID document, date of birth)",
    EventCategory.MONEY_TRANSFER.value: "Transfer or deposit funds as instructed by the caller",
    EventCategory.SAFE_ACCOUNT.value: "Move funds to a so-called 'safe account' nominated by the caller",
    EventCategory.CRYPTO_TRANSFER.value: "Send funds to a crypto wallet or trading account",
    EventCategory.REMOTE_ACCESS.value: "Grant remote access to the victim's phone or computer",
    EventCategory.SCREEN_SHARING.value: "Share the victim's screen during a banking session",
    EventCategory.APP_INSTALLATION.value: "Install an application supplied by the caller",
    EventCategory.ISOLATION.value: "Stay on the line and not contact the bank independently",
    EventCategory.SECRECY.value: "Keep the conversation secret from family and bank staff",
}

RECOMMENDED_ACTION = {
    Classification.SCAM.value: (
        "Terminate the call. Do not share codes, card data or passwords and do not move money. "
        "Contact the bank yourself using the number on the back of the card or the official app, "
        "and report the call to the bank's anti-fraud line."
    ),
    Classification.SUSPICIOUS.value: (
        "Do not act on the caller's instructions. End the call and verify independently through the "
        "official bank app or hotline before sharing anything or moving money."
    ),
    Classification.SAFE.value: (
        "No fraud indicators requiring intervention. Standard advice applies: never share one-time "
        "codes, card details or passwords with anyone who calls you."
    ),
}


@dataclass
class _Observation:
    segment: TranscriptSegment
    clause: Clause


@dataclass
class _Context:
    """Everything the sequence reasoner accumulates while reading the call."""

    identity_claims: list[_Observation] = field(default_factory=list)
    problem_claims: list[_Observation] = field(default_factory=list)
    solution_frames: list[_Observation] = field(default_factory=list)
    protective: list[_Observation] = field(default_factory=list)
    compliance: list[_Observation] = field(default_factory=list)
    doubt: list[_Observation] = field(default_factory=list)
    reward_frames: list[_Observation] = field(default_factory=list)
    caller_orgs: set[Concept] = field(default_factory=set)
    caller_problems: set[Concept] = field(default_factory=set)
    #: Problems the VICTIM raised first. A caller who then discusses the same
    #: problem is answering a question, not inventing an emergency.
    victim_problems: set[Concept] = field(default_factory=set)
    otp_announced: bool = False
    customer_initiated: bool = False
    first_speaker: Speaker | None = None
    #: Set once the caller has actually demanded a secret. After that, a
    #: "don't tell anyone" from the same caller is secrecy, not advice.
    credential_requested: bool = False


class ReferenceAnalyzer(FraudLLMBackend):
    name = "reference"

    def analyze(self, transcript: Transcript, *, realtime: bool = False) -> LLMAnalysis:
        ctx = _Context()
        events: list[RiskFactor] = []

        for segment in transcript.segments:
            if ctx.first_speaker is None:
                ctx.first_speaker = segment.speaker
            # Concepts established by earlier clauses of the same utterance, so a
            # pronoun ("не сообщайте ЕГО никому") resolves to what it refers to.
            prior: set[Concept] = set()
            for clause in analyse_clauses(segment.text):
                if segment.speaker == Speaker.CALLER:
                    events.extend(self._caller_clause(segment, clause, ctx, prior))
                elif segment.speaker == Speaker.VICTIM:
                    events.extend(self._victim_clause(segment, clause, ctx, prior))
                else:
                    # UNKNOWN speaker: only record protective/legitimacy signals,
                    # never attribute a harmful request to an unidentified voice.
                    if clause.has(
                        Concept.NEVER_ASKS, Concept.OFFICIAL_CHANNEL, Concept.WARNING_FRAME
                    ):
                        ctx.protective.append(_Observation(segment, clause))
                prior |= clause.concepts

        events = self._dedupe(events)
        payoff = {e.category for e in events} & HARMFUL_ACTION_EVENTS
        pressure = {e.category for e in events} & PRESSURE_EVENTS

        # Framing events are only emitted once the call shows what the framing is
        # FOR. This is spec section 9: an organisation's name, or a problem
        # report, is not evidence of fraud by itself.
        events.extend(self._framing_events(ctx, payoff, pressure))
        events = self._dedupe(events)

        transcript_has_protection = bool(ctx.protective)
        classification, links = self._classify(
            events=events,
            ctx=ctx,
            transcript=transcript,
            realtime=realtime,
        )

        if classification == Classification.SAFE and transcript_has_protection:
            # Keep the exculpatory evidence visible in the report.
            events.extend(
                RiskFactor(
                    timestamp=obs.segment.timestamp,
                    speaker=obs.segment.speaker,
                    category=EventCategory.PROTECTIVE_ADVICE.value,
                    severity=Severity.LOW,
                    evidence=obs.clause.text,
                    reason="Speaker warns against disclosing credentials or points to an official channel — consistent with legitimate contact.",
                )
                for obs in ctx.protective[:3]
            )
            events = self._dedupe(events)

        events.sort(key=lambda e: (e.timestamp, e.category))
        scam_types = self._scam_types(events, ctx, classification)
        tactics = self._tactics(events, ctx)
        stage = self._stage(events, ctx, transcript)

        analysis = LLMAnalysis(
            classification=classification,
            confidence=self._confidence(classification, links, events, transcript),
            scam_types=scam_types,
            tactics=tactics,
            conversation_stage=stage,
            requested_actions=self._requested_actions(events),
            risk_factors=events,
            explanation=self._explain(events, ctx, classification, links),
            recommended_action=RECOMMENDED_ACTION[classification.value],
        )
        return self.ground(analysis, transcript)

    # ------------------------------------------------------------------
    # Per-clause rules
    # ------------------------------------------------------------------
    #: Concepts that count as "a secret belonging to the victim".
    CREDENTIAL_TARGETS = (
        Concept.TARGET_OTP,
        Concept.TARGET_CODE_GENERIC,
        Concept.TARGET_CARD_FULL,
        Concept.TARGET_PASSWORD,
    )

    def _caller_clause(
        self,
        segment: TranscriptSegment,
        clause: Clause,
        ctx: _Context,
        prior: set[Concept] | None = None,
    ) -> list[RiskFactor]:
        out: list[RiskFactor] = []
        prior = prior or set()
        act = clause.act
        asks = act == SpeechAct.REQUEST
        forbids = act == SpeechAct.PROHIBITION

        def event(category: EventCategory, severity: Severity, reason: str) -> None:
            if category.value in CREDENTIAL_EXTRACTION_EVENTS:
                ctx.credential_requested = True
            out.append(
                RiskFactor(
                    timestamp=segment.timestamp,
                    speaker=segment.speaker,
                    category=category.value,
                    severity=severity,
                    evidence=clause.text,
                    reason=reason,
                )
            )

        # --- legitimacy signals first: they can flip the meaning of a clause ---
        # "Никому не сообщайте код" protects the victim; "Никому не говорите об
        # этом разговоре" isolates them. Same prohibition form, opposite intent —
        # the difference is whether the thing to be kept quiet is the victim's
        # own secret or the call itself. A caller who has already demanded a
        # secret gets no benefit of the doubt.
        credential_in_scope = clause.has(*self.CREDENTIAL_TARGETS) or bool(
            prior & set(self.CREDENTIAL_TARGETS)
        ) or Concept.NEVER_ASKS in prior
        if clause.has(Concept.NEVER_ASKS, Concept.OFFICIAL_CHANNEL, Concept.WARNING_FRAME) or (
            forbids and credential_in_scope and not ctx.credential_requested
        ):
            ctx.protective.append(_Observation(segment, clause))
            return out

        # --- credential extraction ---
        code_word = clause.has(Concept.TARGET_CODE_WORD)
        # "Вам пришёл код из СМС? Продиктуйте его мне." — the second clause is the
        # request, but the object is a pronoun. Resolve it against the same
        # utterance rather than missing the request entirely.
        otp_in_scope = clause.has(Concept.TARGET_OTP) or (
            Concept.TARGET_OTP in prior and asks
        )
        if otp_in_scope and not code_word:
            if asks:
                event(
                    EventCategory.OTP_REQUEST,
                    Severity.CRITICAL,
                    "Caller asks the other party to disclose a one-time authentication code. "
                    "A one-time code authorises a transaction; no legitimate employee needs it.",
                )
            else:
                ctx.otp_announced = True
        elif clause.has(Concept.TARGET_CODE_GENERIC) and asks and not code_word:
            if ctx.otp_announced or ctx.caller_problems:
                event(
                    EventCategory.OTP_REQUEST,
                    Severity.CRITICAL,
                    "Caller demands 'the code' after announcing an incoming SMS or an account problem — "
                    "a one-time code request in context.",
                )
            else:
                event(
                    EventCategory.OTP_REQUEST,
                    Severity.HIGH,
                    "Caller asks for a code without establishing any legitimate reason to need one.",
                )
        elif code_word and asks:
            event(
                EventCategory.PERSONAL_DATA_REQUEST,
                Severity.LOW,
                "Caller asks for the account code word — a routine bank verification step, "
                "not a one-time transaction code.",
            )

        if clause.has(Concept.TARGET_CARD_FULL) and asks:
            event(
                EventCategory.CARD_REQUEST,
                Severity.CRITICAL,
                "Caller asks for full card credentials (number / expiry / CVV), which are sufficient "
                "to make payments.",
            )
        elif clause.has(Concept.TARGET_CARD_PARTIAL) and asks:
            event(
                EventCategory.PERSONAL_DATA_REQUEST,
                Severity.LOW,
                "Caller asks for the last digits of the card — commonly used for legitimate "
                "identification and not sufficient to transact.",
            )

        if clause.has(Concept.TARGET_PASSWORD):
            if asks:
                event(
                    EventCategory.PASSWORD_REQUEST,
                    Severity.CRITICAL,
                    "Caller asks for a password, login or PIN, which grants standing access to the account.",
                )
            elif ctx.caller_problems and act == SpeechAct.STATEMENT:
                # Implied request: no imperative verb, but the caller has invented
                # a problem and is now naming the victim's password as the next
                # step ("Теперь пароль от личного кабинета…").
                event(
                    EventCategory.PASSWORD_REQUEST,
                    Severity.HIGH,
                    "Caller names the victim's password as the next step in resolving the problem it "
                    "described. No legitimate process needs the account holder's password.",
                )
        if clause.has(Concept.TARGET_CARD_FULL) and not asks and ctx.caller_problems and (
            act == SpeechAct.STATEMENT
        ):
            event(
                EventCategory.CARD_REQUEST,
                Severity.HIGH,
                "Caller names full card credentials as the next step in resolving the problem it "
                "described, without an explicit imperative.",
            )

        if clause.has(Concept.TARGET_PERSONAL_DATA) and asks:
            event(
                EventCategory.PERSONAL_DATA_REQUEST,
                Severity.MEDIUM,
                "Caller collects identification data (IIN / ID document / date of birth).",
            )

        # --- money movement ---
        if clause.has(Concept.TARGET_SAFE_ACCOUNT):
            event(
                EventCategory.SAFE_ACCOUNT,
                Severity.CRITICAL,
                "Caller directs the victim's money to a 'safe / reserve / special' account. "
                "No such account exists at Kazakhstan banks — this is the payoff step of the classic "
                "safe-account scam.",
            )
        elif clause.has(Concept.TARGET_MONEY) and asks:
            severity = Severity.CRITICAL if clause.has(Concept.TARGET_CASH_POINT) else Severity.HIGH
            event(
                EventCategory.MONEY_TRANSFER,
                severity,
                "Caller instructs the victim to move or deposit funds.",
            )
        if clause.has(Concept.TARGET_CRYPTO) and (asks or clause.has(Concept.TARGET_MONEY)):
            event(
                EventCategory.CRYPTO_TRANSFER,
                Severity.HIGH,
                "Caller directs funds into a crypto wallet or trading account — an irreversible transfer.",
            )

        # --- device access: even a statement of intent counts, because naming a
        # remote-desktop tool on a bank call has no legitimate use case.
        if clause.has(Concept.TARGET_REMOTE_APP, Concept.TARGET_REMOTE_ACCESS):
            event(
                EventCategory.REMOTE_ACCESS,
                Severity.CRITICAL,
                "Caller seeks remote control of the victim's device, which allows transactions to be "
                "made directly from the victim's banking session.",
            )
        if clause.has(Concept.TARGET_SCREEN) and (asks or act == SpeechAct.STATEMENT):
            event(
                EventCategory.SCREEN_SHARING,
                Severity.HIGH,
                "Caller asks the victim to share their screen, exposing balances and incoming codes.",
            )
        if clause.has(Concept.TARGET_APP_INSTALL) and asks:
            event(
                EventCategory.APP_INSTALLATION,
                Severity.HIGH,
                "Caller asks the victim to install software supplied by the caller"
                + (" from a link they sent." if clause.has(Concept.TARGET_LINK) else "."),
            )

        # --- pressure ---
        if clause.has(Concept.THREAT):
            event(
                EventCategory.THREAT,
                Severity.HIGH,
                "Caller threatens loss of funds, blocking or legal consequences to force compliance.",
            )
        elif clause.has(Concept.FEAR):
            event(
                EventCategory.FEAR,
                Severity.MEDIUM,
                "Caller frames the victim's money or identity as being under active threat.",
            )
        if clause.has(Concept.URGENCY):
            event(
                EventCategory.URGENCY,
                Severity.MEDIUM if ctx.caller_problems else Severity.LOW,
                "Caller imposes a deadline, removing the victim's time to verify independently.",
            )
        if clause.has(Concept.SECRECY):
            event(
                EventCategory.SECRECY,
                Severity.HIGH,
                "Caller instructs the victim to keep the conversation secret — legitimate institutions "
                "never require secrecy from the account holder's own family or bank staff.",
            )
        if clause.has(Concept.ISOLATION):
            event(
                EventCategory.ISOLATION,
                Severity.HIGH,
                "Caller tries to prevent independent verification (stay on the line / do not call the bank).",
            )

        # --- framing signals recorded for the sequence reasoner ---
        orgs = {c for c in ORG_CONCEPTS if c in clause.concepts}
        if orgs:
            ctx.caller_orgs |= orgs
            if clause.has(Concept.SELF_INTRO, Concept.ROLE_CLAIM, Concept.SECURITY_DEPT):
                ctx.identity_claims.append(_Observation(segment, clause))
        elif clause.has(Concept.ROLE_CLAIM, Concept.SECURITY_DEPT) and clause.has(Concept.SELF_INTRO):
            ctx.identity_claims.append(_Observation(segment, clause))

        problems = {c for c in PROBLEM_CONCEPTS if c in clause.concepts} - ctx.victim_problems
        if problems:
            ctx.caller_problems |= problems
            ctx.problem_claims.append(_Observation(segment, clause))

        if clause.has(Concept.SOLUTION_FRAME):
            ctx.solution_frames.append(_Observation(segment, clause))
        if clause.has(Concept.REWARD_FRAME):
            ctx.reward_frames.append(_Observation(segment, clause))

        return out

    def _victim_clause(
        self,
        segment: TranscriptSegment,
        clause: Clause,
        ctx: _Context,
        prior: set[Concept] | None = None,
    ) -> list[RiskFactor]:
        """Victim turns never create attack events — they contextualise them."""
        out: list[RiskFactor] = []
        prior = prior or set()
        if clause.act == SpeechAct.PROHIBITION and (
            clause.has(*self.CREDENTIAL_TARGETS) or bool(prior & set(self.CREDENTIAL_TARGETS))
        ):
            ctx.protective.append(_Observation(segment, clause))
        if clause.has(
            Concept.NEVER_ASKS, Concept.OFFICIAL_CHANNEL, Concept.VICTIM_REFUSAL,
            Concept.WARNING_FRAME,
        ):
            ctx.protective.append(_Observation(segment, clause))
        if clause.has(Concept.VICTIM_COMPLIANCE):
            ctx.compliance.append(_Observation(segment, clause))
        if clause.has(Concept.VICTIM_DOUBT):
            ctx.doubt.append(_Observation(segment, clause))
        if clause.has(Concept.CUSTOMER_INQUIRY):
            ctx.customer_initiated = True
        # Whatever problem the victim raises is theirs, not the caller's invention.
        ctx.victim_problems |= {c for c in PROBLEM_CONCEPTS if c in clause.concepts}
        return out

    # ------------------------------------------------------------------
    # Sequence reasoning
    # ------------------------------------------------------------------
    def _framing_events(
        self, ctx: _Context, payoff: set[str], pressure: set[str]
    ) -> list[RiskFactor]:
        """Turn identity/problem claims into findings only when they serve an attack."""
        out: list[RiskFactor] = []
        attack_context = bool(payoff) or len(pressure) >= 2

        if ctx.identity_claims and attack_context:
            obs = ctx.identity_claims[0]
            out.append(
                RiskFactor(
                    timestamp=obs.segment.timestamp,
                    speaker=obs.segment.speaker,
                    category=EventCategory.IMPERSONATION.value,
                    severity=Severity.HIGH if payoff else Severity.MEDIUM,
                    evidence=obs.clause.text,
                    reason=(
                        "Caller claims institutional authority and then uses it to justify actions that "
                        "harm the account holder — the authority claim is the lever, not a legitimate "
                        "introduction."
                    ),
                )
            )

        if ctx.problem_claims and attack_context:
            obs = ctx.problem_claims[0]
            contradicted = bool(ctx.doubt)
            out.append(
                RiskFactor(
                    timestamp=obs.segment.timestamp,
                    speaker=obs.segment.speaker,
                    category=EventCategory.FALSE_PROBLEM.value,
                    severity=Severity.HIGH if payoff else Severity.MEDIUM,
                    evidence=obs.clause.text,
                    reason=(
                        "Caller asserts an account problem the victim did not initiate"
                        + (" and explicitly denies" if contradicted else "")
                        + ", creating the emergency that the caller's instructions then 'solve'."
                    ),
                )
            )

        if ctx.solution_frames and ctx.problem_claims and attack_context:
            obs = ctx.solution_frames[0]
            out.append(
                RiskFactor(
                    timestamp=obs.segment.timestamp,
                    speaker=obs.segment.speaker,
                    category=EventCategory.FALSE_SOLUTION.value,
                    severity=Severity.MEDIUM,
                    evidence=obs.clause.text,
                    reason=(
                        "Caller offers themselves as the only remedy for the problem they invented, "
                        "channelling the victim into the requested action."
                    ),
                )
            )
        return out

    def _classify(
        self,
        *,
        events: list[RiskFactor],
        ctx: _Context,
        transcript: Transcript,
        realtime: bool,
    ) -> tuple[Classification, dict[str, bool]]:
        categories = {e.category for e in events}
        payoff = categories & HARMFUL_ACTION_EVENTS
        pressure = categories & PRESSURE_EVENTS
        payoff_families = sum(
            1
            for family in (CREDENTIAL_EXTRACTION_EVENTS, MONEY_MOVEMENT_EVENTS, DEVICE_ACCESS_EVENTS)
            if payoff & family
        )

        links = {
            "identity_claim": bool(ctx.identity_claims),
            "false_problem": EventCategory.FALSE_PROBLEM.value in categories,
            "pressure": bool(pressure - {EventCategory.FALSE_PROBLEM.value, EventCategory.FALSE_SOLUTION.value}),
            "false_solution": EventCategory.FALSE_SOLUTION.value in categories,
            "harmful_request": bool(payoff),
        }
        framed = links["identity_claim"] or links["false_problem"] or links["false_solution"] or len(pressure) >= 2

        # Inbound call the customer placed, with no harmful request, is a normal
        # service conversation however much banking vocabulary it contains.
        inbound_service = (
            not payoff
            and (transcript.call_direction == "inbound" or ctx.customer_initiated)
        )

        if payoff and framed:
            if realtime and not links["identity_claim"] and not links["false_problem"] and len(pressure) < 2:
                return Classification.SUSPICIOUS, links
            return Classification.SCAM, links
        if payoff:
            if payoff_families >= 2 or (payoff & MONEY_MOVEMENT_EVENTS and EventCategory.SAFE_ACCOUNT.value in payoff):
                return Classification.SCAM, links
            if payoff & CREDENTIAL_EXTRACTION_EVENTS or payoff & DEVICE_ACCESS_EVENTS:
                # An unexplained demand for a secret or for device control is not
                # provably a scam without framing, but it is never routine.
                return Classification.SUSPICIOUS, links
            return Classification.SUSPICIOUS, links
        if inbound_service:
            return Classification.SAFE, links
        if links["false_problem"] and links["pressure"]:
            return Classification.SUSPICIOUS, links
        if len(pressure) >= 3:
            return Classification.SUSPICIOUS, links
        return Classification.SAFE, links

    def _confidence(
        self,
        classification: Classification,
        links: dict[str, bool],
        events: list[RiskFactor],
        transcript: Transcript,
    ) -> float:
        n_links = sum(1 for v in links.values() if v)
        critical = sum(1 for e in events if e.severity == Severity.CRITICAL)
        if classification == Classification.SCAM:
            base = 0.62 + 0.06 * n_links + 0.03 * min(critical, 3)
            cap = 0.95
        elif classification == Classification.SUSPICIOUS:
            base = 0.42 + 0.05 * n_links
            cap = 0.75
        else:
            positive = [e for e in events if e.category != EventCategory.PROTECTIVE_ADVICE.value]
            base = 0.88 - 0.06 * len(positive)
            cap = 0.92
        # A noisy transcript is weaker ground for any verdict.
        if transcript.mean_confidence < 0.75:
            base -= 0.08
        if len(transcript.segments) < 4:
            base -= 0.10
        return round(max(0.30, min(cap, base)), 2)

    def _scam_types(
        self, events: list[RiskFactor], ctx: _Context, classification: Classification
    ) -> list[str]:
        if classification == Classification.SAFE:
            return []
        categories = {e.category for e in events}
        types: list[str] = []

        def add(value: str) -> None:
            if value not in types:
                types.append(value)

        impersonating = EventCategory.IMPERSONATION.value in categories
        if impersonating:
            if Concept.SECURITY_DEPT in ctx.caller_orgs or Concept.ORG_BANK in ctx.caller_orgs:
                add(ScamType.BANK_IMPERSONATION.value)
            if Concept.ORG_POLICE in ctx.caller_orgs:
                add(ScamType.POLICE_IMPERSONATION.value)
            if Concept.ORG_INVESTIGATOR in ctx.caller_orgs:
                add(ScamType.INVESTIGATOR_IMPERSONATION.value)
            if Concept.ORG_GOVERNMENT in ctx.caller_orgs:
                add(ScamType.GOVERNMENT_IMPERSONATION.value)

        if Concept.PROBLEM_LOAN in ctx.caller_problems:
            add(ScamType.FAKE_LOAN.value)
        if Concept.PROBLEM_TRANSACTION in ctx.caller_problems:
            add(ScamType.FAKE_TRANSACTION.value)
        if ctx.caller_problems & {Concept.PROBLEM_COMPROMISE, Concept.PROBLEM_BLOCKED}:
            add(ScamType.ACCOUNT_COMPROMISE.value)
        if Concept.PROBLEM_REFUND in ctx.caller_problems:
            add(ScamType.FAKE_REFUND.value)

        if EventCategory.OTP_REQUEST.value in categories:
            add(ScamType.OTP_THEFT.value)
        if EventCategory.CARD_REQUEST.value in categories:
            add(ScamType.CARD_DATA_THEFT.value)
        if EventCategory.PASSWORD_REQUEST.value in categories:
            add(ScamType.CREDENTIAL_THEFT.value)
        if EventCategory.SAFE_ACCOUNT.value in categories:
            add(ScamType.SAFE_ACCOUNT_SCAM.value)
        if EventCategory.MONEY_TRANSFER.value in categories:
            add(ScamType.MONEY_TRANSFER_SCAM.value)
        if categories & DEVICE_ACCESS_EVENTS:
            add(ScamType.REMOTE_ACCESS_SCAM.value)
        if EventCategory.CRYPTO_TRANSFER.value in categories or Concept.ORG_CRYPTO in ctx.caller_orgs:
            add(ScamType.CRYPTO_SCAM.value)
        if Concept.ORG_INVESTMENT in ctx.caller_orgs and ctx.reward_frames:
            add(ScamType.INVESTMENT_SCAM.value)
        if Concept.ORG_DELIVERY in ctx.caller_orgs:
            add(ScamType.DELIVERY_SCAM.value)
        if Concept.ORG_MARKETPLACE in ctx.caller_orgs:
            add(ScamType.MARKETPLACE_SCAM.value)
        if Concept.ORG_EMPLOYER in ctx.caller_orgs and ctx.reward_frames:
            add(ScamType.JOB_SCAM.value)

        if not types:
            add(ScamType.OTHER_SOCIAL_ENGINEERING.value)
        return types

    def _tactics(self, events: list[RiskFactor], ctx: _Context) -> list[str]:
        tactics: list[str] = []
        for event in events:
            tactic = EVENT_TO_TACTIC.get(event.category)
            if tactic and tactic not in tactics:
                tactics.append(tactic)
        authority_orgs = {Concept.ORG_POLICE, Concept.ORG_INVESTIGATOR, Concept.ORG_GOVERNMENT}
        has_impersonation = EventCategory.IMPERSONATION.value in {e.category for e in events}
        if has_impersonation and ctx.caller_orgs & authority_orgs and Tactic.AUTHORITY.value not in tactics:
            tactics.append(Tactic.AUTHORITY.value)
        return tactics

    def _stage(self, events: list[RiskFactor], ctx: _Context, transcript: Transcript) -> str:
        if not events:
            if not transcript.segments:
                return Stage.UNKNOWN.value
            return Stage.INTRODUCTION.value if len(transcript.segments) <= 4 else Stage.UNKNOWN.value
        # "Current" stage = deepest stage signalled in the most recent third of
        # the call, falling back to the deepest stage overall.
        recent_cutoff = transcript.segments[-1].index - max(1, len(transcript.segments) // 3)
        recent = [
            e for e in events
            if e.segment_index is not None and e.segment_index >= recent_cutoff
        ] or events
        stages = [EVENT_TO_STAGE.get(e.category) for e in recent]
        stages = [s for s in stages if s]
        if not stages:
            stages = [s for s in (EVENT_TO_STAGE.get(e.category) for e in events) if s]
        if not stages:
            return Stage.UNKNOWN.value
        deepest = max(stages, key=lambda s: STAGE_DEPTH.get(s, 0))
        if deepest == Stage.MONEY_TRANSFER.value and ctx.compliance:
            return Stage.PAYMENT.value
        return deepest

    def _requested_actions(self, events: list[RiskFactor]) -> list[str]:
        seen: list[str] = []
        for event in sorted(events, key=lambda e: e.timestamp):
            text = REQUESTED_ACTION_TEXT.get(event.category)
            if text and text not in seen:
                seen.append(text)
        return seen

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------
    def _explain(
        self,
        events: list[RiskFactor],
        ctx: _Context,
        classification: Classification,
        links: dict[str, bool],
    ) -> str:
        def first(category: EventCategory) -> RiskFactor | None:
            matches = [e for e in events if e.category == category.value]
            return matches[0] if matches else None

        if classification == Classification.SAFE:
            parts = ["No social-engineering sequence was found."]
            if ctx.customer_initiated:
                parts.append("The customer initiated the enquiry and set the topic.")
            if ctx.protective:
                obs = ctx.protective[0]
                parts.append(
                    f"At {obs.segment.timestamp} a speaker explicitly warns against disclosing "
                    f'credentials or refers to an official channel ("{obs.clause.text}"), which is the '
                    "opposite of a fraudulent request."
                )
            parts.append(
                "Banking vocabulary alone (bank names, cards, transfers, codes, loans) is not treated "
                "as evidence: no request for one-time codes, card credentials, passwords, money movement "
                "or device access was made by the caller."
            )
            return " ".join(parts)

        steps: list[str] = []
        imp = first(EventCategory.IMPERSONATION)
        if imp:
            steps.append(f'claims institutional authority at {imp.timestamp} ("{imp.evidence}")')
        prob = first(EventCategory.FALSE_PROBLEM)
        if prob:
            steps.append(f'asserts an unverifiable account problem at {prob.timestamp} ("{prob.evidence}")')
        for category in (EventCategory.THREAT, EventCategory.FEAR, EventCategory.URGENCY):
            hit = first(category)
            if hit:
                steps.append(f'applies {category.value.lower()} at {hit.timestamp} ("{hit.evidence}")')
                break
        for category in (EventCategory.SECRECY, EventCategory.ISOLATION):
            hit = first(category)
            if hit:
                label = "demands secrecy" if category == EventCategory.SECRECY else "blocks independent verification"
                steps.append(f'{label} at {hit.timestamp} ("{hit.evidence}")')
        sol = first(EventCategory.FALSE_SOLUTION)
        if sol:
            steps.append(f'offers itself as the only remedy at {sol.timestamp} ("{sol.evidence}")')

        payoff_events = [e for e in events if e.category in HARMFUL_ACTION_EVENTS]
        for event in payoff_events[:3]:
            steps.append(
                f'requests {event.category.replace("_", " ").lower()} at {event.timestamp} ("{event.evidence}")'
            )

        head = (
            "The caller follows a recognisable social-engineering sequence: "
            if classification == Classification.SCAM
            else "The call shows part of a social-engineering sequence: "
        )
        body = "; then ".join(steps) if steps else "insufficient detail to reconstruct the sequence"
        tail = ""
        if payoff_events:
            tail = (
                " The requested action transfers control of the victim's money or credentials to the caller, "
                "which is the payoff step of the attack."
            )
        elif links["false_problem"]:
            tail = (
                " No harmful request has been made yet, so the verdict stays below SCAM; the pattern is "
                "consistent with the opening phase of an attack."
            )
        if ctx.compliance:
            tail += (
                f" The victim shows signs of complying ({ctx.compliance[0].segment.timestamp}: "
                f'"{ctx.compliance[0].clause.text}"), so intervention is time-critical.'
            )
        if ctx.doubt and not ctx.compliance:
            tail += " The victim contradicts the caller's premise, which further undermines the claim."
        return head + body + "." + tail

    # ------------------------------------------------------------------
    @staticmethod
    def _dedupe(events: list[RiskFactor]) -> list[RiskFactor]:
        """One event per (category, timestamp), keeping the highest severity."""
        best: dict[tuple[str, str], RiskFactor] = {}
        for event in events:
            key = (event.category, event.timestamp)
            current = best.get(key)
            if current is None or SEVERITY_ORDER.get(event.severity.value, 0) > SEVERITY_ORDER.get(
                current.severity.value, 0
            ):
                best[key] = event
        return list(best.values())
