"""Real-time alert policy (spec section 22).

Two rules, in priority order:

1. **A critical event alerts immediately**, even if the total score has not
   reached 80. Waiting for the score to accumulate while the victim is reading a
   one-time code aloud defeats the purpose of a live system.
2. Otherwise the band thresholds apply: <30 silent, 30-59 monitor, 60-79 warn,
   80+ critical.

Alerts also de-duplicate: the same category does not re-alert, and the level only
escalates. A warning that fires every 8 seconds is noise, and noise gets ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from config.ontology import HARMFUL_ACTION_EVENTS, Severity
from config.settings import (
    ALERT_CRITICAL_THRESHOLD,
    ALERT_MONITOR_THRESHOLD,
    ALERT_WARNING_THRESHOLD,
)
from transcription.schemas import Alert, ConversationState, RiskAssessment, RiskFactor

_LEVEL_ORDER = {"NONE": 0, "MONITOR": 1, "WARNING": 2, "CRITICAL": 3}

#: Victim-facing guidance per critical category, in the two languages that matter.
IMMEDIATE_GUIDANCE: dict[str, str] = {
    "OTP_REQUEST": (
        "Do not read the SMS/push code aloud. A bank never needs it. / "
        "Кодты айтпаңыз — банк оны сұрамайды. / Не диктуйте код — банк его не спрашивает."
    ),
    "CARD_REQUEST": (
        "Do not give the full card number, expiry or CVV. / "
        "Карта нөмірін және CVV-ді айтпаңыз. / Не сообщайте номер карты и CVV."
    ),
    "PASSWORD_REQUEST": (
        "Do not give your password, login or PIN. / "
        "Құпия сөзді айтпаңыз. / Не сообщайте пароль или ПИН."
    ),
    "MONEY_TRANSFER": (
        "Do not transfer or deposit money on this call. / "
        "Ақша аудармаңыз. / Не переводите деньги."
    ),
    "SAFE_ACCOUNT": (
        "There is no such thing as a 'safe account' — this is a scam. / "
        "«Қауіпсіз шот» деген жоқ, бұл алаяқтық. / «Безопасного счёта» не существует, это мошенники."
    ),
    "REMOTE_ACCESS": (
        "Do not install remote-access software or grant access to your phone. / "
        "Қашықтан кіру қолданбасын орнатпаңыз. / Не устанавливайте программы удалённого доступа."
    ),
    "SCREEN_SHARING": (
        "Do not share your screen while your banking app is open. / "
        "Экранды көрсетпеңіз. / Не включайте демонстрацию экрана."
    ),
    "APP_INSTALLATION": (
        "Do not install apps from links sent during a call. / "
        "Сілтемедегі қолданбаны орнатпаңыз. / Не устанавливайте приложения по ссылке из звонка."
    ),
    "CRYPTO_TRANSFER": (
        "Crypto transfers cannot be reversed. Do not send funds. / "
        "Крипто аударымын жасамаңыз. / Не отправляйте средства в крипто-кошелёк."
    ),
}


@dataclass
class AlertPolicy:
    """Decides whether an update should surface an alert."""

    monitor_threshold: int = ALERT_MONITOR_THRESHOLD
    warning_threshold: int = ALERT_WARNING_THRESHOLD
    critical_threshold: int = ALERT_CRITICAL_THRESHOLD
    #: Categories already alerted on, so we do not repeat ourselves.
    seen_categories: set[str] = field(default_factory=set)
    highest_level: str = "NONE"

    def evaluate(
        self,
        risk: RiskAssessment,
        new_events: Sequence[RiskFactor],
        state: ConversationState | None = None,
    ) -> Alert | None:
        critical_new = [
            event
            for event in new_events
            if event.category in HARMFUL_ACTION_EVENTS
            and event.severity == Severity.CRITICAL
            and event.category not in self.seen_categories
        ]
        if critical_new:
            event = critical_new[0]
            self.seen_categories.add(event.category)
            return self._emit(
                Alert(
                    level="CRITICAL",
                    headline=f"STOP — {event.category.replace('_', ' ').lower()} requested",
                    detail=IMMEDIATE_GUIDANCE.get(
                        event.category,
                        "End the call and contact your bank using the number on your card.",
                    ),
                    triggered_by=event.category,
                    timestamp=event.timestamp,
                    risk_score=risk.risk_score,
                )
            )

        score = risk.risk_score
        if score >= self.critical_threshold:
            level, headline = "CRITICAL", "Critical fraud warning"
            detail = "Multiple independent fraud indicators. End the call now."
        elif score >= self.warning_threshold:
            level, headline = "WARNING", "Suspicious call"
            detail = (
                "This call follows a social-engineering pattern. Do not share codes or move money; "
                "hang up and call the bank yourself."
            )
        elif score >= self.monitor_threshold:
            level, headline = "MONITOR", "Monitoring this call"
            detail = "Pressure or problem-framing detected. No harmful request yet."
        else:
            return None

        # Only escalate; do not re-fire the same level every chunk.
        if _LEVEL_ORDER[level] <= _LEVEL_ORDER[self.highest_level]:
            return None
        return self._emit(
            Alert(
                level=level,
                headline=headline,
                detail=detail,
                triggered_by="risk_score",
                timestamp=(state.events[-1].timestamp if state and state.events else ""),
                risk_score=score,
            )
        )

    def _emit(self, alert: Alert) -> Alert:
        if _LEVEL_ORDER[alert.level] > _LEVEL_ORDER[self.highest_level]:
            self.highest_level = alert.level
        return alert

    def reset(self) -> None:
        self.seen_categories.clear()
        self.highest_level = "NONE"
