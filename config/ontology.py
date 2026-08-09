"""QORGAU fraud ontology — the single source of truth for every label in the system.

The fine-tuned LLM, the risk engine, the dataset synthesiser, the evaluator and the
UI all import their vocabulary from here. If a label is not in this file it does not
exist anywhere in QORGAU.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """`str`-valued enum (kept explicit for 3.10 compatibility)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# ---------------------------------------------------------------------------
# Classification / risk vocabulary
# ---------------------------------------------------------------------------


class Classification(StrEnum):
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    SCAM = "SCAM"


class RiskLevel(StrEnum):
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


SEVERITY_ORDER: dict[str, int] = {
    Severity.LOW.value: 0,
    Severity.MEDIUM.value: 1,
    Severity.HIGH.value: 2,
    Severity.CRITICAL.value: 3,
}


class Speaker(StrEnum):
    CALLER = "CALLER"
    VICTIM = "VICTIM"
    UNKNOWN = "UNKNOWN"


class Language(StrEnum):
    KK = "kk"
    RU = "ru"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Scam types (spec section 7)
# ---------------------------------------------------------------------------


class ScamType(StrEnum):
    BANK_IMPERSONATION = "BANK_IMPERSONATION"
    POLICE_IMPERSONATION = "POLICE_IMPERSONATION"
    GOVERNMENT_IMPERSONATION = "GOVERNMENT_IMPERSONATION"
    INVESTIGATOR_IMPERSONATION = "INVESTIGATOR_IMPERSONATION"

    ACCOUNT_COMPROMISE = "ACCOUNT_COMPROMISE"
    FAKE_LOAN = "FAKE_LOAN"
    FAKE_TRANSACTION = "FAKE_TRANSACTION"
    FAKE_REFUND = "FAKE_REFUND"

    OTP_THEFT = "OTP_THEFT"
    CARD_DATA_THEFT = "CARD_DATA_THEFT"
    CREDENTIAL_THEFT = "CREDENTIAL_THEFT"

    SAFE_ACCOUNT_SCAM = "SAFE_ACCOUNT_SCAM"
    MONEY_TRANSFER_SCAM = "MONEY_TRANSFER_SCAM"

    REMOTE_ACCESS_SCAM = "REMOTE_ACCESS_SCAM"

    INVESTMENT_SCAM = "INVESTMENT_SCAM"
    CRYPTO_SCAM = "CRYPTO_SCAM"

    MARKETPLACE_SCAM = "MARKETPLACE_SCAM"
    DELIVERY_SCAM = "DELIVERY_SCAM"

    ROMANCE_SCAM = "ROMANCE_SCAM"
    JOB_SCAM = "JOB_SCAM"

    OTHER_SOCIAL_ENGINEERING = "OTHER_SOCIAL_ENGINEERING"


IMPERSONATION_SCAM_TYPES: frozenset[str] = frozenset(
    {
        ScamType.BANK_IMPERSONATION.value,
        ScamType.POLICE_IMPERSONATION.value,
        ScamType.GOVERNMENT_IMPERSONATION.value,
        ScamType.INVESTIGATOR_IMPERSONATION.value,
    }
)


# ---------------------------------------------------------------------------
# Manipulation tactics (spec section 8)
# ---------------------------------------------------------------------------


class Tactic(StrEnum):
    IMPERSONATION = "IMPERSONATION"
    URGENCY = "URGENCY"
    FEAR = "FEAR"
    AUTHORITY = "AUTHORITY"
    SECRECY = "SECRECY"
    ISOLATION = "ISOLATION"
    FALSE_PROBLEM = "FALSE_PROBLEM"
    FALSE_SOLUTION = "FALSE_SOLUTION"

    OTP_REQUEST = "OTP_REQUEST"
    CARD_REQUEST = "CARD_REQUEST"
    PASSWORD_REQUEST = "PASSWORD_REQUEST"

    MONEY_TRANSFER_REQUEST = "MONEY_TRANSFER_REQUEST"
    REMOTE_ACCESS_REQUEST = "REMOTE_ACCESS_REQUEST"
    INSTALLATION_REQUEST = "INSTALLATION_REQUEST"
    SCREEN_SHARING = "SCREEN_SHARING"

    CRYPTO_TRANSFER = "CRYPTO_TRANSFER"


# ---------------------------------------------------------------------------
# Risk event categories (spec section 12)
# ---------------------------------------------------------------------------


class EventCategory(StrEnum):
    IMPERSONATION = "IMPERSONATION"
    THREAT = "THREAT"
    URGENCY = "URGENCY"
    FEAR = "FEAR"
    SECRECY = "SECRECY"
    ISOLATION = "ISOLATION"
    FALSE_PROBLEM = "FALSE_PROBLEM"
    FALSE_SOLUTION = "FALSE_SOLUTION"
    OTP_REQUEST = "OTP_REQUEST"
    CARD_REQUEST = "CARD_REQUEST"
    PASSWORD_REQUEST = "PASSWORD_REQUEST"
    MONEY_TRANSFER = "MONEY_TRANSFER"
    SAFE_ACCOUNT = "SAFE_ACCOUNT"
    REMOTE_ACCESS = "REMOTE_ACCESS"
    SCREEN_SHARING = "SCREEN_SHARING"
    APP_INSTALLATION = "APP_INSTALLATION"
    CRYPTO_TRANSFER = "CRYPTO_TRANSFER"
    PERSONAL_DATA_REQUEST = "PERSONAL_DATA_REQUEST"
    #: Exculpatory signal: a speaker tells the other party NOT to share a secret,
    #: or refuses to accept sensitive data. Never contributes positive risk.
    PROTECTIVE_ADVICE = "PROTECTIVE_ADVICE"


#: Categories that describe an attempt to obtain a secret.
CREDENTIAL_EXTRACTION_EVENTS: frozenset[str] = frozenset(
    {
        EventCategory.OTP_REQUEST.value,
        EventCategory.CARD_REQUEST.value,
        EventCategory.PASSWORD_REQUEST.value,
    }
)

#: Categories that describe an attempt to move the victim's money.
MONEY_MOVEMENT_EVENTS: frozenset[str] = frozenset(
    {
        EventCategory.MONEY_TRANSFER.value,
        EventCategory.SAFE_ACCOUNT.value,
        EventCategory.CRYPTO_TRANSFER.value,
    }
)

#: Categories that describe an attempt to take control of the victim's device.
DEVICE_ACCESS_EVENTS: frozenset[str] = frozenset(
    {
        EventCategory.REMOTE_ACCESS.value,
        EventCategory.SCREEN_SHARING.value,
        EventCategory.APP_INSTALLATION.value,
    }
)

#: Psychological pressure — never a payoff on its own.
PRESSURE_EVENTS: frozenset[str] = frozenset(
    {
        EventCategory.URGENCY.value,
        EventCategory.FEAR.value,
        EventCategory.THREAT.value,
        EventCategory.SECRECY.value,
        EventCategory.ISOLATION.value,
        EventCategory.FALSE_PROBLEM.value,
        EventCategory.FALSE_SOLUTION.value,
    }
)

#: The "payoff" families — a completed scam needs at least one of these.
HARMFUL_ACTION_EVENTS: frozenset[str] = (
    CREDENTIAL_EXTRACTION_EVENTS | MONEY_MOVEMENT_EVENTS | DEVICE_ACCESS_EVENTS
)


# ---------------------------------------------------------------------------
# Conversation stages (spec section 11)
# ---------------------------------------------------------------------------


class Stage(StrEnum):
    INTRODUCTION = "INTRODUCTION"
    IDENTITY_CLAIM = "IDENTITY_CLAIM"
    PROBLEM_CREATION = "PROBLEM_CREATION"
    FEAR_ESCALATION = "FEAR_ESCALATION"
    TRUST_BUILDING = "TRUST_BUILDING"
    INFORMATION_EXTRACTION = "INFORMATION_EXTRACTION"
    CREDENTIAL_EXTRACTION = "CREDENTIAL_EXTRACTION"
    MONEY_TRANSFER = "MONEY_TRANSFER"
    REMOTE_ACCESS = "REMOTE_ACCESS"
    PAYMENT = "PAYMENT"
    EXIT = "EXIT"
    UNKNOWN = "UNKNOWN"


#: Rough progression order — used to report the deepest stage reached so far in
#: incremental (real-time) analysis. Higher means deeper into the attack.
STAGE_DEPTH: dict[str, int] = {
    Stage.UNKNOWN.value: 0,
    Stage.INTRODUCTION.value: 1,
    Stage.IDENTITY_CLAIM.value: 2,
    Stage.TRUST_BUILDING.value: 3,
    Stage.PROBLEM_CREATION.value: 4,
    Stage.FEAR_ESCALATION.value: 5,
    Stage.INFORMATION_EXTRACTION.value: 6,
    Stage.REMOTE_ACCESS.value: 7,
    Stage.CREDENTIAL_EXTRACTION.value: 8,
    Stage.MONEY_TRANSFER.value: 9,
    Stage.PAYMENT.value: 10,
    Stage.EXIT.value: 11,
}


# ---------------------------------------------------------------------------
# Shared mappings (analyser, risk engine, synthesiser)
# ---------------------------------------------------------------------------

#: An event category implies a tactic, so the model never has to restate itself.
EVENT_TO_TACTIC: dict[str, str] = {
    EventCategory.IMPERSONATION.value: Tactic.IMPERSONATION.value,
    EventCategory.THREAT.value: Tactic.FEAR.value,
    EventCategory.URGENCY.value: Tactic.URGENCY.value,
    EventCategory.FEAR.value: Tactic.FEAR.value,
    EventCategory.SECRECY.value: Tactic.SECRECY.value,
    EventCategory.ISOLATION.value: Tactic.ISOLATION.value,
    EventCategory.FALSE_PROBLEM.value: Tactic.FALSE_PROBLEM.value,
    EventCategory.FALSE_SOLUTION.value: Tactic.FALSE_SOLUTION.value,
    EventCategory.OTP_REQUEST.value: Tactic.OTP_REQUEST.value,
    EventCategory.CARD_REQUEST.value: Tactic.CARD_REQUEST.value,
    EventCategory.PASSWORD_REQUEST.value: Tactic.PASSWORD_REQUEST.value,
    EventCategory.MONEY_TRANSFER.value: Tactic.MONEY_TRANSFER_REQUEST.value,
    EventCategory.SAFE_ACCOUNT.value: Tactic.MONEY_TRANSFER_REQUEST.value,
    EventCategory.REMOTE_ACCESS.value: Tactic.REMOTE_ACCESS_REQUEST.value,
    EventCategory.SCREEN_SHARING.value: Tactic.SCREEN_SHARING.value,
    EventCategory.APP_INSTALLATION.value: Tactic.INSTALLATION_REQUEST.value,
    EventCategory.CRYPTO_TRANSFER.value: Tactic.CRYPTO_TRANSFER.value,
}

#: Event category -> the conversation stage that category signals.
EVENT_TO_STAGE: dict[str, str] = {
    EventCategory.IMPERSONATION.value: Stage.IDENTITY_CLAIM.value,
    EventCategory.FALSE_PROBLEM.value: Stage.PROBLEM_CREATION.value,
    EventCategory.THREAT.value: Stage.FEAR_ESCALATION.value,
    EventCategory.FEAR.value: Stage.FEAR_ESCALATION.value,
    EventCategory.URGENCY.value: Stage.FEAR_ESCALATION.value,
    EventCategory.FALSE_SOLUTION.value: Stage.TRUST_BUILDING.value,
    EventCategory.SECRECY.value: Stage.TRUST_BUILDING.value,
    EventCategory.ISOLATION.value: Stage.TRUST_BUILDING.value,
    EventCategory.PERSONAL_DATA_REQUEST.value: Stage.INFORMATION_EXTRACTION.value,
    EventCategory.OTP_REQUEST.value: Stage.CREDENTIAL_EXTRACTION.value,
    EventCategory.CARD_REQUEST.value: Stage.CREDENTIAL_EXTRACTION.value,
    EventCategory.PASSWORD_REQUEST.value: Stage.CREDENTIAL_EXTRACTION.value,
    EventCategory.MONEY_TRANSFER.value: Stage.MONEY_TRANSFER.value,
    EventCategory.SAFE_ACCOUNT.value: Stage.MONEY_TRANSFER.value,
    EventCategory.CRYPTO_TRANSFER.value: Stage.PAYMENT.value,
    EventCategory.REMOTE_ACCESS.value: Stage.REMOTE_ACCESS.value,
    EventCategory.SCREEN_SHARING.value: Stage.REMOTE_ACCESS.value,
    EventCategory.APP_INSTALLATION.value: Stage.REMOTE_ACCESS.value,
}


def enum_values(enum_cls: type[Enum]) -> list[str]:
    """All values of an enum in declaration order (used to build prompts)."""
    return [m.value for m in enum_cls]
