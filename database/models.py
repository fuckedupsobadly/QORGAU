"""SQLAlchemy schema (spec section 23) with the security rules from section 27.

Design decisions that follow from "treat this as sensitive financial data":

* Transcript rows carry **two** text columns: `text_masked` (what is stored and
  queried) and `text_encrypted` (the original, encrypted at rest and only
  decrypted for an authorised investigation). OTPs, PINs, CVVs and card numbers
  never reach the plaintext column.
* Recordings are referenced by an internal path plus a retention deadline. There
  is no public URL column, so a recording cannot be exposed by accident.
* Every read or export of a call is written to `audit_log`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config.settings import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    language: Mapped[str] = mapped_column(String(16), default="unknown")
    language_profile: Mapped[dict] = mapped_column(JSON, default=dict)
    call_direction: Mapped[str] = mapped_column(String(16), default="unknown")
    final_risk: Mapped[int] = mapped_column(Integer, default=0)
    risk_level: Mapped[str] = mapped_column(String(16), default="SAFE")
    peak_risk: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="new")  # new|analyzing|analyzed
    mean_asr_confidence: Mapped[float] = mapped_column(Float, default=1.0)

    #: Internal storage reference only — never a public URL (spec section 27).
    recording_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    recording_purge_after: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    transcript_purge_after: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    segments: Mapped[list["TranscriptSegmentRow"]] = relationship(
        back_populates="call", cascade="all, delete-orphan", order_by="TranscriptSegmentRow.seq"
    )
    events: Mapped[list["DetectedEvent"]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )
    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )

    def apply_retention(self) -> None:
        self.recording_purge_after = self.started_at + timedelta(
            days=settings.security.recording_retention_days
        )
        self.transcript_purge_after = self.started_at + timedelta(
            days=settings.security.transcript_retention_days
        )


class TranscriptSegmentRow(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint("call_id", "seq", name="uq_segment_seq"),
        Index("ix_segment_call_start", "call_id", "start"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str] = mapped_column(String(16))
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[str] = mapped_column(String(16))
    language: Mapped[str] = mapped_column(String(16), default="unknown")

    #: Query/display copy, with credentials masked.
    text_masked: Mapped[str] = mapped_column(Text)
    #: Verbatim ASR output, encrypted at rest. Evidence of record.
    text_encrypted: Mapped[bytes | None] = mapped_column(nullable=True)
    masked_labels: Mapped[list] = mapped_column(JSON, default=list)
    asr_confidence: Mapped[float] = mapped_column(Float, default=1.0)
    normalization_notes: Mapped[list] = mapped_column(JSON, default=list)

    call: Mapped[Call] = relationship(back_populates="segments")


class DetectedEvent(Base):
    __tablename__ = "detected_events"
    __table_args__ = (Index("ix_event_call_category", "call_id", "category"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    timestamp: Mapped[str] = mapped_column(String(16))
    speaker: Mapped[str] = mapped_column(String(16), default="UNKNOWN")
    category: Mapped[str] = mapped_column(String(48))
    severity: Mapped[str] = mapped_column(String(16))
    #: Masked evidence quote — the unmasked span lives in the encrypted segment.
    evidence: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text, default="")
    segment_seq: Mapped[int | None] = mapped_column(Integer, nullable=True)
    grounding_score: Mapped[float] = mapped_column(Float, default=0.0)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    call: Mapped[Call] = relationship(back_populates="events")


class Analysis(Base):
    __tablename__ = "analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    classification: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    scam_types: Mapped[list] = mapped_column(JSON, default=list)
    tactics: Mapped[list] = mapped_column(JSON, default=list)
    conversation_stage: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    requested_actions: Mapped[list] = mapped_column(JSON, default=list)
    explanation: Mapped[str] = mapped_column(Text, default="")
    recommended_action: Mapped[str] = mapped_column(Text, default="")
    risk_score: Mapped[int] = mapped_column(Integer, default=0)
    risk_level: Mapped[str] = mapped_column(String(16), default="SAFE")
    risk_contributions: Mapped[list] = mapped_column(JSON, default=list)
    disagreement: Mapped[str | None] = mapped_column(Text, nullable=True)
    stage_timeline: Mapped[list] = mapped_column(JSON, default=list)
    model_backend: Mapped[str] = mapped_column(String(48), default="")
    dropped_findings: Mapped[list] = mapped_column(JSON, default=list)
    is_realtime: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    call: Mapped[Call] = relationship(back_populates="analyses")


class AuditLog(Base):
    """Who touched which call, and when (spec section 27)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(128), default="system")
    action: Mapped[str] = mapped_column(String(64))
    call_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    detail: Mapped[str] = mapped_column(Text, default="")
