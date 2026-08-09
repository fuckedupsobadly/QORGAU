"""Transcript normalization (spec section 6).

Two invariants, both enforced in code rather than by convention:

1. **The original is never modified.** `TranscriptSegment.text_original` is the
   evidence of record; normalization only ever writes `text`.
2. **Normalization can never remove evidence.** Every rule is applied
   speculatively and reverted if it would drop a protected token — a digit, a
   money amount, an OTP reference, an organisation, a name, or a requested
   action. Code-switching is likewise preserved: a Kazakh word is never
   translated into Russian or vice versa.

The result is that any LLM finding can be traced back to the words actually
spoken, and no rule can quietly delete the thing that made a call suspicious.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from models.fraud_llm.lexicon import LEXICON, Concept, canon

# ---------------------------------------------------------------------------
# Protected material
# ---------------------------------------------------------------------------

#: Concepts whose surface forms are evidence and must survive normalization.
PROTECTED_CONCEPTS: tuple[Concept, ...] = (
    Concept.ORG_BANK, Concept.ORG_POLICE, Concept.ORG_INVESTIGATOR, Concept.TARGET_CODE_WORD,
    Concept.ORG_GOVERNMENT, Concept.ORG_TELCO, Concept.ORG_DELIVERY,
    Concept.ORG_MARKETPLACE, Concept.ORG_CRYPTO, Concept.ORG_INVESTMENT,
    Concept.SECURITY_DEPT, Concept.TARGET_OTP, Concept.TARGET_CODE_GENERIC,
    Concept.TARGET_CARD_FULL, Concept.TARGET_CARD_PARTIAL, Concept.TARGET_PASSWORD,
    Concept.TARGET_PERSONAL_DATA, Concept.TARGET_MONEY, Concept.TARGET_SAFE_ACCOUNT,
    Concept.TARGET_CASH_POINT, Concept.TARGET_REMOTE_APP, Concept.TARGET_REMOTE_ACCESS,
    Concept.TARGET_SCREEN, Concept.TARGET_APP_INSTALL, Concept.TARGET_LINK,
    Concept.TARGET_CRYPTO, Concept.PROBLEM_LOAN, Concept.PROBLEM_TRANSACTION,
    Concept.PROBLEM_BLOCKED, Concept.PROBLEM_COMPROMISE, Concept.PROBLEM_LEGAL,
    Concept.PROBLEM_REFUND, Concept.NEVER_ASKS, Concept.OFFICIAL_CHANNEL,
)

_PROTECTED_STEMS: tuple[str, ...] = tuple(
    sorted(
        {canon(stem) for concept in PROTECTED_CONCEPTS for stem in LEXICON[concept] if canon(stem)},
        key=len,
        reverse=True,
    )
)

_DIGITS = re.compile(r"\d+")

#: Non-lexical disfluencies. Deliberately excludes real words like "ну", "вот",
#: "значит" — dropping those would edit what the speaker actually said.
FILLERS: frozenset[str] = frozenset(
    {"ээ", "эээ", "ээээ", "эм", "эмм", "мм", "ммм", "мммм", "аа", "ааа", "ыы", "ыыы",
     "әә", "әәә", "ммда", "эх", "ааэ", "uh", "um", "ehm"}
)

#: ASR artefacts: annotation markers, unknown-token markers, stray symbols.
_ARTIFACTS = re.compile(
    r"(\[[^\]]{0,30}\]|<[^>]{0,20}>|\((?:музыка|шум|смех|неразборчиво|inaudible|noise)\)|«\s*»)",
    re.IGNORECASE,
)

#: Curated fixes for recurring Kazakh/Russian telephony ASR errors. These only
#: rejoin or re-spell material — they never delete it.
KNOWN_FIXES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bэни\s?дес?к\b", re.I), "AnyDesk", "remote-access app name rejoined"),
    (re.compile(r"\bтим\s?вью?вер\b", re.I), "TeamViewer", "remote-access app name rejoined"),
    (re.compile(r"\bраст\s?дес?к\b", re.I), "RustDesk", "remote-access app name rejoined"),
    (re.compile(r"\bкас\s?пи\b", re.I), "Kaspi", "bank name rejoined"),
    (re.compile(r"\bсмс\s?код\b", re.I), "СМС-код", "OTP reference normalised"),
    (re.compile(r"\bэс\s?эм\s?эс\b", re.I), "СМС", "spelled-out SMS normalised"),
    (re.compile(r"\bси\s?ви\s?ви\b", re.I), "CVV", "spelled-out CVV normalised"),
    (re.compile(r"\bпин\s?код\b", re.I), "ПИН-код", "PIN reference normalised"),
    (re.compile(r"\bбезопасн(ый|ом|ый)?\s+щот\b", re.I), r"безопасный счет", "misheard 'счет' corrected"),
    (re.compile(r"\bқауіпсіз\s+шод\b", re.I), "қауіпсіз шот", "misheard Kazakh 'шот' corrected"),
    (re.compile(r"\bи\s?и\s?н\b", re.I), "ИИН", "spelled-out IIN normalised"),
)

_MULTI_SPACE = re.compile(r"\s{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_REPEATED_PUNCT = re.compile(r"([,.!?])\1{1,}")
_ELLIPSIS = re.compile(r"\.{2,}")
_HYPHEN_STUTTER = re.compile(r"\b(\w{1,3})(?:-\1){1,}(\w*)\b", re.UNICODE)


@dataclass
class NormalizationResult:
    text: str
    notes: list[str] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.notes)


def protected_counts(text: str) -> dict[str, int]:
    """Multiset of protected material present in `text`."""
    hay = canon(text)
    counts: dict[str, int] = {}
    for digits in _DIGITS.findall(text):
        counts[f"num:{digits}"] = counts.get(f"num:{digits}", 0) + 1
    for stem in _PROTECTED_STEMS:
        occurrences = hay.count(stem)
        if occurrences:
            counts[f"stem:{stem}"] = occurrences
    return counts


def _preserves(before: dict[str, int], after_text: str) -> bool:
    after = protected_counts(after_text)
    return all(after.get(key, 0) >= value for key, value in before.items())


class TranscriptNormalizer:
    """Applies safe, reversible cleanups to one ASR utterance."""

    def normalize(self, text: str) -> NormalizationResult:
        original = text or ""
        if not original.strip():
            return NormalizationResult(text=original)

        baseline = protected_counts(original)
        current = original
        notes: list[str] = []
        reverted: list[str] = []

        for rule_name, rule in (
            ("asr artefacts removed", self._strip_artifacts),
            ("stutter collapsed", self._collapse_stutter),
            ("fillers removed", self._drop_fillers),
            ("known ASR errors corrected", self._apply_known_fixes),
            ("punctuation restored", self._restore_punctuation),
        ):
            candidate, detail = rule(current)
            if candidate == current:
                continue
            if not _preserves(baseline, candidate):
                # A rule that would delete evidence is dropped, not tuned.
                reverted.append(f"{rule_name} (would have removed protected material)")
                continue
            current = candidate
            notes.append(detail or rule_name)

        return NormalizationResult(text=current, notes=notes, reverted=reverted)

    # ------------------------------------------------------------------
    @staticmethod
    def _strip_artifacts(text: str) -> tuple[str, str]:
        cleaned = _ARTIFACTS.sub(" ", text)
        cleaned = _ELLIPSIS.sub(" ", cleaned)
        cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
        return cleaned, "ASR artefacts and ellipses removed"

    @staticmethod
    def _collapse_stutter(text: str) -> tuple[str, str]:
        cleaned = _HYPHEN_STUTTER.sub(lambda m: m.group(1) + m.group(2), text)
        tokens = cleaned.split()
        out: list[str] = []
        for token in tokens:
            if out and canon(out[-1]) == canon(token) and canon(token):
                continue  # "ваш ваш счет" -> "ваш счет"
            out.append(token)
        return " ".join(out), "repeated words collapsed"

    @staticmethod
    def _drop_fillers(text: str) -> tuple[str, str]:
        tokens = text.split()
        kept = [token for token in tokens if canon(token) not in FILLERS]
        if len(kept) == len(tokens):
            return text, ""
        if not kept:
            return text, ""  # an utterance that is nothing but a filler stays as-is
        return " ".join(kept), f"{len(tokens) - len(kept)} filler token(s) removed"

    @staticmethod
    def _apply_known_fixes(text: str) -> tuple[str, str]:
        applied: list[str] = []
        cleaned = text
        for pattern, replacement, note in KNOWN_FIXES:
            replaced = pattern.sub(replacement, cleaned)
            if replaced != cleaned:
                cleaned = replaced
                applied.append(note)
        return cleaned, "; ".join(applied)

    @staticmethod
    def _restore_punctuation(text: str) -> tuple[str, str]:
        cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        cleaned = _REPEATED_PUNCT.sub(r"\1", cleaned)
        cleaned = _MULTI_SPACE.sub(" ", cleaned).strip(" ,;:")
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned, "punctuation and sentence boundary restored"


normalizer = TranscriptNormalizer()


def normalize_text(text: str) -> NormalizationResult:
    return normalizer.normalize(text)
