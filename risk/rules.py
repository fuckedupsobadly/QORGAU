"""Risk weights, groups, caps and interaction effects (spec section 19).

Every number here is a policy decision, not a learned parameter. They live in one
file so an analyst can audit and tune them without touching the engine, and so
`risk/calibration.py` can sweep them against the labelled corpus.
"""

from __future__ import annotations

from config.ontology import (
    CREDENTIAL_EXTRACTION_EVENTS,
    DEVICE_ACCESS_EVENTS,
    EventCategory,
    MONEY_MOVEMENT_EVENTS,
    PRESSURE_EVENTS,
    Severity,
)

# ---------------------------------------------------------------------------
# Base weights — points at HIGH severity
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    EventCategory.IMPERSONATION.value: 20,
    EventCategory.THREAT.value: 12,
    EventCategory.URGENCY.value: 10,
    EventCategory.FEAR.value: 10,
    EventCategory.SECRECY.value: 15,
    EventCategory.ISOLATION.value: 15,
    EventCategory.FALSE_PROBLEM.value: 15,
    EventCategory.FALSE_SOLUTION.value: 12,
    EventCategory.OTP_REQUEST.value: 35,
    EventCategory.PASSWORD_REQUEST.value: 35,
    EventCategory.CARD_REQUEST.value: 30,
    EventCategory.MONEY_TRANSFER.value: 35,
    EventCategory.SAFE_ACCOUNT.value: 40,
    EventCategory.REMOTE_ACCESS.value: 35,
    EventCategory.SCREEN_SHARING.value: 35,
    EventCategory.APP_INSTALLATION.value: 25,
    EventCategory.CRYPTO_TRANSFER.value: 30,
    EventCategory.PERSONAL_DATA_REQUEST.value: 6,
    #: Exculpatory — contributes no positive risk, ever.
    EventCategory.PROTECTIVE_ADVICE.value: 0,
}

SEVERITY_MULTIPLIER: dict[str, float] = {
    Severity.LOW.value: 0.40,
    Severity.MEDIUM.value: 0.70,
    Severity.HIGH.value: 1.00,
    Severity.CRITICAL.value: 1.15,
}


# ---------------------------------------------------------------------------
# Groups and caps — "do not blindly sum everything"
# ---------------------------------------------------------------------------

IDENTITY_GROUP = frozenset({EventCategory.IMPERSONATION.value})
EXTRACTION_GROUP = CREDENTIAL_EXTRACTION_EVENTS | {EventCategory.PERSONAL_DATA_REQUEST.value}
MOVEMENT_GROUP = MONEY_MOVEMENT_EVENTS
ACCESS_GROUP = DEVICE_ACCESS_EVENTS
PRESSURE_GROUP = PRESSURE_EVENTS

GROUPS: dict[str, frozenset[str]] = {
    "IDENTITY": IDENTITY_GROUP,
    "PRESSURE": PRESSURE_GROUP,
    "EXTRACTION": EXTRACTION_GROUP,
    "MOVEMENT": MOVEMENT_GROUP,
    "ACCESS": ACCESS_GROUP,
}

#: Ten urgency mentions are not ten times one urgency mention.
GROUP_CAPS: dict[str, float] = {
    "IDENTITY": 18,
    "PRESSURE": 24,
    "EXTRACTION": 40,
    "MOVEMENT": 42,
    "ACCESS": 38,
}

GROUP_LABELS: dict[str, str] = {
    "IDENTITY": "Impersonation / false authority",
    "PRESSURE": "Psychological pressure",
    "EXTRACTION": "Credential extraction",
    "MOVEMENT": "Money movement",
    "ACCESS": "Device access",
}


# ---------------------------------------------------------------------------
# Interaction effects — the combination is worse than the parts
# ---------------------------------------------------------------------------

#: (group_a, group_b, points, explanation)
GROUP_INTERACTIONS: tuple[tuple[str, str, float, str], ...] = (
    (
        "IDENTITY",
        "EXTRACTION",
        10,
        "A false authority claim is what makes the credential request credible — together they "
        "are the core of an OTP/card theft attack.",
    ),
    (
        "IDENTITY",
        "MOVEMENT",
        8,
        "Impersonating an institution is how a caller obtains the authority to redirect the "
        "victim's money.",
    ),
    (
        "IDENTITY",
        "ACCESS",
        8,
        "A claimed support role is the standard pretext for obtaining remote access.",
    ),
    (
        "PRESSURE",
        "EXTRACTION",
        6,
        "Urgency and fear remove the victim's opportunity to verify before disclosing a secret.",
    ),
    (
        "PRESSURE",
        "MOVEMENT",
        6,
        "Pressure applied immediately before a transfer request is designed to prevent reflection.",
    ),
    (
        "PRESSURE",
        "ACCESS",
        6,
        "Pressure combined with a device-access request is a hallmark of fake technical support.",
    ),
    (
        "ACCESS",
        "EXTRACTION",
        6,
        "Device access plus a credential request lets the caller both read incoming codes and use them.",
    ),
)

#: (category_a, category_b, points, explanation)
CATEGORY_INTERACTIONS: tuple[tuple[str, str, float, str], ...] = (
    (
        EventCategory.FALSE_PROBLEM.value,
        EventCategory.FALSE_SOLUTION.value,
        5,
        "Inventing a problem and then supplying the only remedy is the problem/solution pair that "
        "defines social engineering.",
    ),
    (
        EventCategory.SAFE_ACCOUNT.value,
        EventCategory.MONEY_TRANSFER.value,
        8,
        "A transfer instruction combined with a 'safe account' destination is the completed "
        "safe-account scam.",
    ),
    (
        EventCategory.SECRECY.value,
        EventCategory.MONEY_TRANSFER.value,
        5,
        "Demanding secrecy around a transfer removes the family/bank checks that normally stop "
        "this attack.",
    ),
    (
        EventCategory.ISOLATION.value,
        EventCategory.OTP_REQUEST.value,
        5,
        "Keeping the victim on the line while a code arrives prevents them from calling the bank.",
    ),
)


# ---------------------------------------------------------------------------
# Model-verdict contribution
# ---------------------------------------------------------------------------

#: The contextual model's own verdict is one deterministic input among several —
#: it never overrides the evidence-derived score, it adds a bounded amount.
VERDICT_POINTS: dict[str, float] = {"SCAM": 12, "SUSPICIOUS": 5, "SAFE": 0}
VERDICT_MIN_CONFIDENCE = 0.60


# ---------------------------------------------------------------------------
# Mitigations
# ---------------------------------------------------------------------------

#: Someone in the call warns against disclosure and nothing harmful was requested.
PROTECTIVE_ADVICE_MITIGATION = 25
#: The victim placed the call themselves and nothing harmful was requested.
INBOUND_CALL_MITIGATION = 12

#: A transcript this noisy is weak ground for any verdict.
LOW_CONFIDENCE_ASR_THRESHOLD = 0.55
LOW_CONFIDENCE_DAMPENING = 0.85


# ---------------------------------------------------------------------------
# Policy floors — spec section 22
# ---------------------------------------------------------------------------

#: A CRITICAL request for credentials / money / device access, inside an
#: impersonation or false-problem frame, is CRITICAL by policy regardless of how
#: the additive terms land.
FLOOR_CRITICAL_WITH_CHAIN = 80
#: The same request with no framing established yet still warrants a warning.
FLOOR_CRITICAL_ALONE = 60
#: A HIGH-severity harmful request inside an impersonation / false-problem frame.
#: Same logic as the critical floor one band down: the framing is what turns a
#: request into an attack, so the pair is scored as high risk even when the
#: individual signals are only HIGH.
FLOOR_HIGH_HARMFUL_WITH_CHAIN = 60
#: A HIGH-severity harmful request with no framing established warrants monitoring.
FLOOR_HIGH_HARMFUL = 45


def event_points(category: str, severity: str) -> float:
    """Base weight scaled by severity."""
    return WEIGHTS.get(category, 0.0) * SEVERITY_MULTIPLIER.get(severity, 1.0)


def group_of(category: str) -> str | None:
    for name, members in GROUPS.items():
        if category in members:
            return name
    return None
