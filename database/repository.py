"""Persistence, encryption at rest, retention and audit logging.

Security posture (spec section 27), stated plainly:

* Stored transcript text is **masked** — OTPs, PINs, CVVs and card numbers are
  replaced before they reach a queryable column.
* The verbatim ASR text is encrypted with Fernet (AES-128-CBC + HMAC) under a key
  supplied via `QORGAU_ENCRYPTION_KEY`. If no key or no `cryptography` install is
  available, QORGAU **declines to persist the unmasked text at all** rather than
  quietly writing plaintext or pretending to encrypt it.
* Every write, read and export is audit-logged.
* Retention deadlines are stored per call and enforced by `purge_expired()`.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Sequence

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from database.models import (
    Analysis,
    AuditLog,
    Base,
    Call,
    DetectedEvent,
    TranscriptSegmentRow,
    utcnow,
)
from models.fraud_llm.lexicon import mask_sensitive
from transcription.schemas import CallAnalysis, LLMAnalysis, Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encryption at rest
# ---------------------------------------------------------------------------


class Encryptor:
    """Encrypt/decrypt the verbatim transcript."""

    available = False

    def encrypt(self, text: str) -> bytes | None:  # pragma: no cover - interface
        return None

    def decrypt(self, blob: bytes | None) -> str | None:  # pragma: no cover - interface
        return None


class FernetEncryptor(Encryptor):
    available = True

    def __init__(self, key: bytes) -> None:
        from cryptography.fernet import Fernet

        self._fernet = Fernet(key)

    def encrypt(self, text: str) -> bytes | None:
        return self._fernet.encrypt(text.encode("utf-8"))

    def decrypt(self, blob: bytes | None) -> str | None:
        if not blob:
            return None
        return self._fernet.decrypt(blob).decode("utf-8")


class RefusingEncryptor(Encryptor):
    """No key configured — the unmasked original is not stored.

    This is deliberate. Writing raw call transcripts to disk unencrypted is worse
    than losing the verbatim copy, and silently downgrading to plaintext would
    make the storage claim in the report untrue.
    """

    available = False

    def encrypt(self, text: str) -> bytes | None:
        return None

    def decrypt(self, blob: bytes | None) -> str | None:
        return None


def build_encryptor() -> Encryptor:
    key = os.environ.get(settings.security.encryption_key_env)
    if not settings.security.encrypt_at_rest:
        logger.warning("encryption at rest disabled by configuration")
        return RefusingEncryptor()
    if not key:
        logger.warning(
            "%s is not set — verbatim transcripts will NOT be persisted "
            "(masked text still is). Generate a key with: python -c "
            "'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'",
            settings.security.encryption_key_env,
        )
        return RefusingEncryptor()
    try:
        return FernetEncryptor(key.encode() if isinstance(key, str) else key)
    except Exception as exc:  # pragma: no cover - misconfiguration
        logger.error("invalid encryption key (%s) — refusing to store verbatim text", exc)
        return RefusingEncryptor()


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class CallRepository:
    def __init__(self, database_url: str | None = None, encryptor: Encryptor | None = None) -> None:
        self.database_url = database_url or settings.database_url
        if self.database_url.startswith("sqlite"):
            settings.paths.storage.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(self.database_url, future=True)
        self._sessionmaker = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self.encryptor = encryptor or build_encryptor()
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -- audit --------------------------------------------------------
    def audit(self, action: str, *, call_id: str | None = None, actor: str = "system", detail: str = "") -> None:
        with self.session() as session:
            session.add(AuditLog(action=action, call_id=call_id, actor=actor, detail=detail))
        path = settings.security.audit_log_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{actor}\t"
                    f"{action}\t{call_id or '-'}\t{detail}\n"
                )
        except OSError:  # pragma: no cover - non-fatal
            logger.warning("could not append to audit log file %s", path)

    # -- writes -------------------------------------------------------
    def save(self, result: CallAnalysis, *, actor: str = "system", is_realtime: bool = False) -> str:
        """Upsert a call, its transcript, its events and one analysis row."""
        transcript, analysis, risk = result.transcript, result.analysis, result.risk
        with self.session() as session:
            call = session.get(Call, result.call_id)
            if call is None:
                call = Call(id=result.call_id, started_at=utcnow())
                session.add(call)
            call.duration = transcript.duration
            call.language = transcript.dominant_language
            call.language_profile = transcript.language_profile
            call.call_direction = transcript.call_direction
            call.final_risk = risk.risk_score
            call.risk_level = risk.risk_level.value
            call.peak_risk = max(call.peak_risk or 0, risk.risk_score)
            call.status = "analyzed"
            call.mean_asr_confidence = transcript.mean_confidence
            call.recording_ref = transcript.audio_path
            call.ended_at = call.ended_at or utcnow()
            call.apply_retention()

            # Transcript: replace wholesale so re-analysis cannot duplicate rows.
            for row in list(call.segments):
                session.delete(row)
            session.flush()
            for segment in transcript.segments:
                masked, labels = mask_sensitive(segment.text)
                session.add(
                    TranscriptSegmentRow(
                        call_id=call.id,
                        seq=segment.index,
                        speaker=segment.speaker.value,
                        start=segment.start,
                        end=segment.end,
                        timestamp=segment.timestamp,
                        language=segment.language.value,
                        text_masked=masked,
                        text_encrypted=self.encryptor.encrypt(segment.text_original),
                        masked_labels=labels,
                        asr_confidence=segment.confidence,
                        normalization_notes=segment.normalization_notes,
                    )
                )

            for row in list(call.events):
                session.delete(row)
            session.flush()
            for event in analysis.risk_factors:
                masked_evidence, _ = mask_sensitive(event.evidence)
                session.add(
                    DetectedEvent(
                        call_id=call.id,
                        timestamp=event.timestamp,
                        speaker=event.speaker.value,
                        category=event.category,
                        severity=event.severity.value,
                        evidence=masked_evidence,
                        reason=event.reason,
                        segment_seq=event.segment_index,
                        grounding_score=event.grounding_score,
                    )
                )

            session.add(
                Analysis(
                    call_id=call.id,
                    classification=analysis.classification.value,
                    confidence=analysis.confidence,
                    scam_types=analysis.scam_types,
                    tactics=analysis.tactics,
                    conversation_stage=analysis.conversation_stage,
                    requested_actions=analysis.requested_actions,
                    explanation=analysis.explanation,
                    recommended_action=analysis.recommended_action,
                    risk_score=risk.risk_score,
                    risk_level=risk.risk_level.value,
                    risk_contributions=[c.model_dump() for c in risk.contributions],
                    disagreement=risk.disagreement,
                    stage_timeline=result.stage_timeline,
                    model_backend=analysis.model_backend,
                    dropped_findings=analysis.dropped_findings,
                    is_realtime=is_realtime,
                )
            )
        self.audit(
            "save_analysis",
            call_id=result.call_id,
            actor=actor,
            detail=f"risk={risk.risk_score} level={risk.risk_level.value} backend={analysis.model_backend}",
        )
        return result.call_id

    # -- reads --------------------------------------------------------
    def list_calls(self, limit: int = 50, *, include_deleted: bool = False) -> list[dict]:
        with self.session() as session:
            stmt = select(Call).order_by(Call.started_at.desc()).limit(limit)
            if not include_deleted:
                stmt = stmt.where(Call.deleted_at.is_(None))
            rows = session.scalars(stmt).all()
            return [
                {
                    "id": row.id,
                    "started_at": row.started_at,
                    "duration": row.duration,
                    "language": row.language,
                    "call_direction": row.call_direction,
                    "final_risk": row.final_risk,
                    "risk_level": row.risk_level,
                    "peak_risk": row.peak_risk,
                    "status": row.status,
                }
                for row in rows
            ]

    def get_call(self, call_id: str, *, actor: str = "system", decrypt: bool = False) -> dict | None:
        """Full record. `decrypt=True` is an investigation action and is audited."""
        with self.session() as session:
            call = session.get(Call, call_id)
            if call is None or call.deleted_at is not None:
                return None
            analysis = (
                session.scalars(
                    select(Analysis)
                    .where(Analysis.call_id == call_id)
                    .order_by(Analysis.created_at.desc())
                    .limit(1)
                ).first()
            )
            payload = {
                "call": {
                    "id": call.id,
                    "started_at": call.started_at,
                    "duration": call.duration,
                    "language": call.language,
                    "language_profile": call.language_profile,
                    "call_direction": call.call_direction,
                    "final_risk": call.final_risk,
                    "risk_level": call.risk_level,
                    "peak_risk": call.peak_risk,
                    "mean_asr_confidence": call.mean_asr_confidence,
                    "recording_ref": call.recording_ref,
                    "recording_purge_after": call.recording_purge_after,
                    "transcript_purge_after": call.transcript_purge_after,
                },
                "segments": [
                    {
                        "seq": row.seq,
                        "speaker": row.speaker,
                        "start": row.start,
                        "end": row.end,
                        "timestamp": row.timestamp,
                        "language": row.language,
                        "text": row.text_masked,
                        "text_original": (
                            self.encryptor.decrypt(row.text_encrypted) if decrypt else None
                        ),
                        "masked_labels": row.masked_labels,
                        "confidence": row.asr_confidence,
                    }
                    for row in call.segments
                ],
                "events": [
                    {
                        "timestamp": row.timestamp,
                        "speaker": row.speaker,
                        "category": row.category,
                        "severity": row.severity,
                        "evidence": row.evidence,
                        "reason": row.reason,
                        "segment_seq": row.segment_seq,
                    }
                    for row in sorted(call.events, key=lambda e: e.timestamp)
                ],
                "analysis": (
                    {
                        "classification": analysis.classification,
                        "confidence": analysis.confidence,
                        "scam_types": analysis.scam_types,
                        "tactics": analysis.tactics,
                        "conversation_stage": analysis.conversation_stage,
                        "requested_actions": analysis.requested_actions,
                        "explanation": analysis.explanation,
                        "recommended_action": analysis.recommended_action,
                        "risk_score": analysis.risk_score,
                        "risk_level": analysis.risk_level,
                        "risk_contributions": analysis.risk_contributions,
                        "disagreement": analysis.disagreement,
                        "stage_timeline": analysis.stage_timeline,
                        "model_backend": analysis.model_backend,
                        "dropped_findings": analysis.dropped_findings,
                    }
                    if analysis
                    else None
                ),
            }
        self.audit(
            "read_call", call_id=call_id, actor=actor, detail=f"decrypt={decrypt}"
        )
        return payload

    def get_events(self, call_id: str) -> list[dict]:
        with self.session() as session:
            rows = session.scalars(
                select(DetectedEvent).where(DetectedEvent.call_id == call_id)
            ).all()
            return [
                {
                    "timestamp": row.timestamp,
                    "speaker": row.speaker,
                    "category": row.category,
                    "severity": row.severity,
                    "evidence": row.evidence,
                    "reason": row.reason,
                }
                for row in sorted(rows, key=lambda r: r.timestamp)
            ]

    # -- deletion & retention ----------------------------------------
    def delete_call(self, call_id: str, *, actor: str = "system", hard: bool = False) -> bool:
        """Erase a call. `hard=True` removes the rows; otherwise it is tombstoned."""
        with self.session() as session:
            call = session.get(Call, call_id)
            if call is None:
                return False
            if hard:
                session.delete(call)
            else:
                call.deleted_at = utcnow()
                for row in call.segments:
                    row.text_masked = "[deleted]"
                    row.text_encrypted = None
                for row in call.events:
                    row.evidence = "[deleted]"
        self.audit("delete_call", call_id=call_id, actor=actor, detail=f"hard={hard}")
        return True

    def purge_expired(self, *, now: datetime | None = None, actor: str = "retention-job") -> dict:
        """Enforce the retention policy. Returns what was purged."""
        moment = now or utcnow()
        purged = {"recordings": 0, "transcripts": 0}
        with self.session() as session:
            for call in session.scalars(select(Call)).all():
                if call.recording_purge_after and call.recording_purge_after <= moment and call.recording_ref:
                    call.recording_ref = None
                    purged["recordings"] += 1
                if call.transcript_purge_after and call.transcript_purge_after <= moment:
                    for row in call.segments:
                        if row.text_encrypted is not None or row.text_masked != "[purged]":
                            row.text_encrypted = None
                            row.text_masked = "[purged]"
                    purged["transcripts"] += 1
        self.audit("purge_expired", actor=actor, detail=str(purged))
        return purged

    # -- reporting ---------------------------------------------------
    def stats(self) -> dict:
        with self.session() as session:
            total = session.scalar(select(func.count()).select_from(Call)) or 0
            by_level = dict(
                session.execute(
                    select(Call.risk_level, func.count()).group_by(Call.risk_level)
                ).all()
            )
            events = dict(
                session.execute(
                    select(DetectedEvent.category, func.count())
                    .group_by(DetectedEvent.category)
                    .order_by(func.count().desc())
                ).all()
            )
        return {
            "calls": total,
            "by_risk_level": by_level,
            "events_by_category": events,
            "encryption_at_rest": self.encryptor.available,
        }


# ---------------------------------------------------------------------------
# Reconstruction helpers (DB row -> in-memory objects for the UI/API)
# ---------------------------------------------------------------------------


def transcript_from_rows(call_id: str, rows: Sequence[dict], direction: str = "unknown") -> Transcript:
    return Transcript(
        call_id=call_id,
        call_direction=direction,
        segments=[
            TranscriptSegment(
                index=row["seq"],
                speaker=row["speaker"],
                start=row["start"],
                end=row["end"],
                language=row["language"],
                text=row["text"],
                text_original=row.get("text_original") or row["text"],
                confidence=row.get("confidence", 1.0),
            )
            for row in rows
        ],
    )


def analysis_from_row(payload: dict) -> LLMAnalysis:
    return LLMAnalysis.model_validate(
        {
            "classification": payload["classification"],
            "confidence": payload["confidence"],
            "scam_types": payload["scam_types"],
            "tactics": payload["tactics"],
            "conversation_stage": payload["conversation_stage"],
            "requested_actions": payload["requested_actions"],
            "risk_factors": [],
            "explanation": payload["explanation"],
            "recommended_action": payload["recommended_action"],
            "model_backend": payload.get("model_backend", ""),
        }
    )


_repository: CallRepository | None = None


def get_repository() -> CallRepository:
    """Process-wide repository singleton."""
    global _repository
    if _repository is None:
        _repository = CallRepository()
    return _repository
