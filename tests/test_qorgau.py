"""Behavioural tests for QORGAU.

These target the properties that make the system trustworthy rather than the
implementation details:

* legitimate banking calls do not raise alarms (the false-positive failure mode)
* full scam sequences do (the missed-detection failure mode)
* a warning about a code is never scored as a request for it
* normalization can never delete evidence
* findings that cannot be traced to the transcript are discarded
* the risk score is deterministic, bounded, and never driven by victim speech
* train/test share no script family

Run with `pytest tests/` or, without pytest installed, `python tests/test_qorgau.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.ontology import EventCategory, Severity, Speaker  # noqa: E402
from config.settings import risk_level  # noqa: E402
from models.fraud_llm.backends.reference import ReferenceAnalyzer  # noqa: E402
from models.fraud_llm.lexicon import detect_language, find_concepts, mask_sensitive  # noqa: E402
from risk.engine import assess  # noqa: E402
from transcription.normalizer import TranscriptNormalizer, protected_counts  # noqa: E402
from transcription.processor import transcript_from_turns  # noqa: E402
from transcription.schemas import LLMAnalysis, RiskFactor  # noqa: E402

analyzer = ReferenceAnalyzer()


def analyse(turns, *, direction="outbound", call_id="t"):
    transcript = transcript_from_turns(turns, call_id=call_id, call_direction=direction)
    analysis = analyzer.analyze(transcript)
    return transcript, analysis, assess(analysis, transcript)


# ---------------------------------------------------------------------------
# The two failure modes that matter
# ---------------------------------------------------------------------------


def test_legitimate_calls_do_not_alarm():
    """Every legitimate family in the corpus must stay below the warning band."""
    from training.corpus import LEGIT_FAMILIES, render_family

    offenders = []
    for family in LEGIT_FAMILIES:
        for language in ("ru", "kk", "mixed"):
            rendered = render_family(family, language_mode=language, noisy=False, seed=1)
            _, analysis, risk = analyse(
                [{"speaker": t["speaker"], "text": t["text"], "start": t["start"], "end": t["end"]}
                 for t in rendered.turns],
                direction=family.call_direction,
                call_id=rendered.call_id,
            )
            if risk.risk_score >= 60:
                offenders.append((rendered.call_id, risk.risk_score, sorted(analysis.event_categories)))
    assert not offenders, f"legitimate calls flagged as high risk: {offenders}"


def test_scam_families_alarm():
    """Every scam family in the corpus must reach at least the warning band."""
    from training.corpus import SCAM_FAMILIES, render_family

    misses = []
    for family in SCAM_FAMILIES:
        for language in ("ru", "kk", "mixed"):
            rendered = render_family(family, language_mode=language, noisy=False, seed=1)
            _, _, risk = analyse(
                [{"speaker": t["speaker"], "text": t["text"], "start": t["start"], "end": t["end"]}
                 for t in rendered.turns],
                direction=family.call_direction,
                call_id=rendered.call_id,
            )
            if risk.risk_score < 60:
                misses.append((rendered.call_id, risk.risk_score))
    assert not misses, f"scam calls not flagged: {misses}"


def test_canonical_scam_is_critical():
    _, analysis, risk = analyse([
        {"speaker": "CALLER", "text": "Здравствуйте, это служба безопасности банка Halyk."},
        {"speaker": "VICTIM", "text": "Иә, тыңдап тұрмын."},
        {"speaker": "CALLER", "text": "На ваше имя оформляется кредит на 3 миллиона тенге."},
        {"speaker": "VICTIM", "text": "Но я ничего не оформлял."},
        {"speaker": "CALLER", "text": "Нужно срочно отменить операцию, иначе деньги спишут."},
        {"speaker": "CALLER", "text": "Қазір вам СМС келеді. Кодты айтып жіберіңіз."},
    ])
    assert analysis.classification.value == "SCAM"
    assert EventCategory.OTP_REQUEST.value in analysis.event_categories
    assert risk.risk_level.value == "CRITICAL"
    assert risk.alert.level == "CRITICAL"


# ---------------------------------------------------------------------------
# Warning vs request — the discrimination the whole design rests on
# ---------------------------------------------------------------------------


def test_bank_warning_about_otp_is_not_a_request():
    _, analysis, risk = analyse(
        [
            {"speaker": "VICTIM", "text": "Здравствуйте, почему не прошла оплата?"},
            {"speaker": "CALLER", "text": "Это Halyk Bank. Лимит по карте исчерпан."},
            {"speaker": "CALLER", "text": "Сотрудник банка никогда не спрашивает код из СМС. Никому не сообщайте его."},
        ],
        direction="inbound",
    )
    assert EventCategory.OTP_REQUEST.value not in analysis.event_categories
    assert EventCategory.SECRECY.value not in analysis.event_categories
    assert EventCategory.PROTECTIVE_ADVICE.value in analysis.event_categories
    assert risk.risk_score == 0


def test_victim_speech_never_adds_risk():
    """The same words from the victim must not be scored as an attack."""
    _, _, risk = analyse(
        [
            {"speaker": "VICTIM", "text": "Переведите деньги на безопасный счет, продиктуйте код из СМС."},
            {"speaker": "CALLER", "text": "Простите, я не понимаю, о чём вы."},
        ]
    )
    assert risk.risk_score == 0, [c.model_dump() for c in risk.contributions]


def test_code_word_is_not_an_otp():
    for text in ("Назовите кодовое слово по счёту.", "Шот бойынша кодтық сөзді растаңыз."):
        _, analysis, _ = analyse(
            [{"speaker": "VICTIM", "text": "Заблокируйте карту пожалуйста."},
             {"speaker": "CALLER", "text": text}],
            direction="inbound",
        )
        assert EventCategory.OTP_REQUEST.value not in analysis.event_categories, text


def test_scam_warning_is_not_a_scam():
    """'If someone asks you to transfer money, that's a scam' must not score."""
    _, analysis, risk = analyse([
        {"speaker": "CALLER", "text": "Это Kaspi, информационная рассылка о безопасности."},
        {"speaker": "CALLER", "text": "Если вам звонят и просят перевести деньги на безопасный счёт — это мошенники, положите трубку."},
    ])
    assert risk.risk_score < 30, [c.model_dump() for c in risk.contributions]
    assert EventCategory.SAFE_ACCOUNT.value not in analysis.event_categories


def test_organisation_name_alone_is_not_evidence():
    _, analysis, risk = analyse([
        {"speaker": "CALLER", "text": "Здравствуйте, я звоню из Halyk Bank по поводу вашей заявки."},
        {"speaker": "VICTIM", "text": "Да, я подавал заявку."},
        {"speaker": "CALLER", "text": "Заявка одобрена, подписать можно в приложении."},
    ])
    assert analysis.classification.value == "SAFE"
    assert EventCategory.IMPERSONATION.value not in analysis.event_categories
    assert risk.risk_score < 30


def test_code_switching_is_not_a_risk_signal():
    """Mixed Kazakh/Russian must not change the verdict on a benign call."""
    scores = []
    for text in (
        "Заявка одобрена, подписать можно в мобильном приложении.",
        "Өтінім мақұлданды, мобильді қолданбада қол қоюға болады.",
        "Өтінім одобрена, мобильді приложениеде қол қоюға болады.",
    ):
        _, _, risk = analyse(
            [{"speaker": "CALLER", "text": "Здравствуйте, это ForteBank по вашей заявке."},
             {"speaker": "CALLER", "text": text}]
        )
        scores.append(risk.risk_score)
    assert max(scores) < 30, scores


# ---------------------------------------------------------------------------
# Evidence integrity
# ---------------------------------------------------------------------------


def test_ungrounded_findings_are_discarded():
    transcript = transcript_from_turns(
        [{"speaker": "CALLER", "text": "Здравствуйте, это банк."}], call_id="g"
    )
    analysis = LLMAnalysis(
        classification="SCAM",
        confidence=0.9,
        risk_factors=[
            RiskFactor(
                timestamp="00:42",
                speaker=Speaker.CALLER,
                category=EventCategory.OTP_REQUEST.value,
                severity=Severity.CRITICAL,
                evidence="Продиктуйте код из SMS немедленно",  # never said
                reason="fabricated",
            )
        ],
    )
    grounded = analyzer.ground(analysis, transcript)
    assert grounded.risk_factors == []
    assert grounded.dropped_findings
    assert assess(grounded, transcript).risk_score == 0


def test_every_finding_carries_traceable_evidence():
    transcript, analysis, _ = analyse([
        {"speaker": "CALLER", "text": "Это служба безопасности банка, на вас оформляется кредит."},
        {"speaker": "CALLER", "text": "Срочно продиктуйте код из СМС."},
    ])
    assert analysis.risk_factors
    for factor in analysis.risk_factors:
        assert factor.evidence.strip(), factor
        index, score = transcript.locate(factor.evidence)
        assert index is not None and score >= 0.72, (factor.category, factor.evidence)


def test_normalizer_never_removes_protected_material():
    samples = [
        "ваш... ваш счет ээ будет заблокирован на 3 000 000 тенге",
        "[шум] переведите 250000 тенге на безопасный счет эээ срочно",
        "установите эни деск и назовите код из смс",
        "сіздің картаңыз ммм заблокирована, 4 цифры айтыңыз",
    ]
    normalizer = TranscriptNormalizer()
    for sample in samples:
        result = normalizer.normalize(sample)
        before, after = protected_counts(sample), protected_counts(result.text)
        for key, count in before.items():
            assert after.get(key, 0) >= count, f"{key} lost in {result.text!r}"


def test_masking_hides_credentials_but_keeps_amounts():
    masked, labels = mask_sensitive("Код 4821, переведите 3 000 000 тенге")
    assert "4821" not in masked
    assert "OTP_OR_PIN" in labels
    assert "3 000 000" in masked

    masked_card, labels_card = mask_sensitive("Номер карты 4400 4301 2345 6789")
    assert "4400" not in masked_card
    assert "CARD_NUMBER" in labels_card


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------


def test_safety_vocabulary_is_not_fear():
    """"служба безопасности" must not match a fear stem — it is in every bank greeting."""
    assert "FEAR" not in {c.value for c in find_concepts("Это служба безопасности банка.")}
    assert "FEAR" not in {c.value for c in find_concepts("Бұл қауіпсіздік қызметі.")}


def test_employee_is_not_a_court():
    assert "ORG_GOVERNMENT" not in {c.value for c in find_concepts("Я сотрудник банка.")}


def test_inflected_phrases_match():
    for text, expected in (
        ("У вас подозрительная операция на счете.", "PROBLEM_TRANSACTION"),
        ("Ақшаны қауіпсіз шотқа аудару керек.", "TARGET_SAFE_ACCOUNT"),
        ("На ваше имя оформляется кредит.", "PROBLEM_LOAN"),
    ):
        assert expected in {c.value for c in find_concepts(text)}, text


def test_language_detection():
    assert detect_language("Здравствуйте, ваш счет заблокирован.") == "ru"
    assert detect_language("Сіздің шотыңыз бұғатталды.") == "kk"
    assert detect_language("Сіздің картаңыз заблокирована.") == "mixed"


# ---------------------------------------------------------------------------
# Risk engine
# ---------------------------------------------------------------------------


def test_risk_engine_is_deterministic_and_bounded():
    turns = [
        {"speaker": "CALLER", "text": "Это служба безопасности банка Kaspi."},
        {"speaker": "CALLER", "text": "На вас оформляется кредит, срочно отмените."},
        {"speaker": "CALLER", "text": "Продиктуйте код из СМС и переведите деньги на безопасный счет."},
    ]
    scores = set()
    for _ in range(5):
        _, _, risk = analyse(turns)
        scores.add(risk.risk_score)
        assert 0 <= risk.risk_score <= 100
        assert risk.risk_level.value == risk_level(risk.risk_score)
    assert len(scores) == 1, scores


def test_repeated_pressure_is_capped():
    """Ten urgency lines must not outweigh a credential request."""
    urgency = [{"speaker": "CALLER", "text": "Срочно, немедленно, времени мало, быстрее!"}] * 10
    _, _, pressure_only = analyse(urgency)
    _, _, with_otp = analyse([
        {"speaker": "CALLER", "text": "Это служба безопасности банка, на вас оформляется кредит."},
        {"speaker": "CALLER", "text": "Продиктуйте код из СМС."},
    ])
    assert pressure_only.risk_score < with_otp.risk_score


def test_contributions_explain_the_score():
    _, _, risk = analyse([
        {"speaker": "CALLER", "text": "Это служба безопасности банка, на вас оформляется кредит."},
        {"speaker": "CALLER", "text": "Продиктуйте код из СМС."},
    ])
    assert risk.contributions
    assert risk.explanation
    kinds = {c.kind for c in risk.contributions}
    assert "event" in kinds


# ---------------------------------------------------------------------------
# Real-time behaviour
# ---------------------------------------------------------------------------


def test_realtime_risk_is_peak_hold_and_alerts_immediately():
    from realtime.session import LiveCallSession, growing_transcripts

    transcript = transcript_from_turns(
        [
            {"speaker": "CALLER", "text": "Здравствуйте, это служба безопасности банка Halyk."},
            {"speaker": "VICTIM", "text": "Да, слушаю."},
            {"speaker": "CALLER", "text": "На ваше имя оформляется кредит, нужно срочно отменить."},
            {"speaker": "CALLER", "text": "Продиктуйте код из СМС."},
            {"speaker": "CALLER", "text": "Хорошей погоды сегодня, кстати."},
        ],
        call_id="rt",
        call_direction="outbound",
    )
    session = LiveCallSession(call_id="rt", call_direction="outbound")
    peaks, alerts = [], []
    for update in session.replay(growing_transcripts(transcript)):
        peaks.append(update.state.risk_score)
        if update.alert:
            alerts.append(update.alert)

    assert peaks == sorted(peaks), f"peak risk decreased: {peaks}"
    assert peaks[-1] >= 60
    assert any(a.level == "CRITICAL" and a.triggered_by == "OTP_REQUEST" for a in alerts)
    # The pleasant closing line must not undo the earlier detection.
    assert session.state.risk_score >= 60


def test_stage_progression_moves_forward():
    from realtime.session import LiveCallSession, growing_transcripts

    transcript = transcript_from_turns(
        [
            {"speaker": "CALLER", "text": "Это служба безопасности банка."},
            {"speaker": "CALLER", "text": "На ваше имя оформляется кредит."},
            {"speaker": "CALLER", "text": "Срочно, иначе спишут деньги."},
            {"speaker": "CALLER", "text": "Продиктуйте код из СМС."},
        ],
        call_id="stage",
    )
    session = LiveCallSession(call_id="stage")
    list(session.replay(growing_transcripts(transcript)))
    timeline = session.stage_timeline()
    assert timeline
    assert timeline[-1]["stage"] in {"CREDENTIAL_EXTRACTION", "MONEY_TRANSFER", "PAYMENT"}


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------


def test_splits_share_no_script_family():
    from training.prepare_dataset import split_families

    assignment = split_families()
    by_split: dict[str, set[str]] = {}
    for family, split in assignment.items():
        by_split.setdefault(split, set()).add(family)
    assert by_split["train"] & by_split.get("test", set()) == set()
    assert by_split["train"] & by_split.get("validation", set()) == set()
    assert by_split.get("test")


def test_gold_evidence_is_verbatim_from_the_transcript():
    from training.corpus import ALL_FAMILIES, render_family
    from training.prepare_dataset import build_record

    for family in ALL_FAMILIES[:6]:
        rendered = render_family(family, language_mode="mixed", noisy=True, seed=3)
        record = build_record(rendered)
        texts = [segment["text"] for segment in record["segments"]]
        for factor in record["gold"]["risk_factors"]:
            assert factor["evidence"] in texts, (family.key, factor["evidence"])


def test_corpus_has_enough_hard_negatives():
    from training.corpus import LEGIT_FAMILIES, SCAM_FAMILIES

    share = len(LEGIT_FAMILIES) / (len(LEGIT_FAMILIES) + len(SCAM_FAMILIES))
    assert share >= 0.35, f"only {share:.0%} of families are legitimate"


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------


def test_schema_rejects_unknown_labels_and_clamps_confidence():
    analysis = LLMAnalysis.model_validate(
        {
            "classification": "SCAM",
            "confidence": 0.9,
            "scam_types": ["BANK_IMPERSONATION", "NOT_A_REAL_TYPE"],
            "tactics": ["OTP_REQUEST", "MIND_CONTROL"],
            "conversation_stage": "NONSENSE",
            "risk_factors": [],
        }
    )
    assert analysis.scam_types == ["BANK_IMPERSONATION"]
    assert analysis.tactics == ["OTP_REQUEST"]
    assert analysis.conversation_stage == "UNKNOWN"

    for bad in (-0.1, 1.4):
        try:
            LLMAnalysis.model_validate({"classification": "SAFE", "confidence": bad})
        except Exception:
            pass
        else:
            raise AssertionError(f"confidence {bad} should not validate")


def test_public_json_matches_the_required_schema():
    _, analysis, _ = analyse([
        {"speaker": "CALLER", "text": "Это банк, продиктуйте код из СМС."},
    ])
    payload = analysis.public_json()
    assert set(payload) == {
        "classification", "confidence", "scam_types", "tactics", "conversation_stage",
        "requested_actions", "risk_factors", "explanation", "recommended_action",
    }
    assert 0.0 <= payload["confidence"] <= 1.0
    json.dumps(payload)  # must be serialisable


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_repository_masks_credentials_and_audits(tmp_path=None):
    import tempfile

    from database.repository import CallRepository, RefusingEncryptor
    from transcription.schemas import CallAnalysis

    directory = Path(tmp_path or tempfile.mkdtemp())
    repository = CallRepository(
        database_url=f"sqlite:///{directory / 'test.db'}", encryptor=RefusingEncryptor()
    )
    transcript, analysis, risk = analyse([
        {"speaker": "CALLER", "text": "Это банк, продиктуйте код 4821 из СМС."},
        {"speaker": "CALLER", "text": "Переведите 3 000 000 тенге на безопасный счет."},
    ], call_id="store")
    repository.save(
        CallAnalysis(call_id="store", transcript=transcript, analysis=analysis, risk=risk)
    )

    stored = repository.get_call("store")
    assert stored is not None
    joined = " ".join(segment["text"] for segment in stored["segments"])
    assert "4821" not in joined, joined
    assert "3 000 000" in joined
    # No key configured -> verbatim text must not be on disk at all.
    assert all(segment["text_original"] is None for segment in stored["segments"])
    assert repository.stats()["calls"] == 1

    assert repository.delete_call("store")
    assert repository.get_call("store") is None


# ---------------------------------------------------------------------------
# Minimal runner for environments without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, test in tests:
        try:
            test()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
