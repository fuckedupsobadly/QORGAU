"""FastAPI surface (spec section 28).

    POST   /api/calls                 upload / register a call
    POST   /api/calls/{id}/analyze    analyse a completed call
    GET    /api/calls/{id}            transcript + analysis
    GET    /api/calls/{id}/events     suspicious events
    WS     /api/calls/{id}/stream     live risk_update frames

Run: `uvicorn app.api.main:app --reload` from the repository root.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.websocket.manager import ConnectionManager
from config.settings import settings
from database.repository import get_repository
from models.inference import analyze_transcript, available_backends, get_backend
from realtime.session import LiveCallSession
from risk.engine import assess, render_report
from transcription.processor import AudioProcessor, transcript_from_turns
from transcription.schemas import CallAnalysis, Transcript

app = FastAPI(
    title="QORGAU",
    description="Kazakh / Russian telephone fraud intelligence",
    version="0.1.0",
)
manager = ConnectionManager()

#: Live sessions keyed by call id. A real deployment would use Redis so any
#: worker can serve any call; this keeps the prototype single-process.
SESSIONS: dict[str, LiveCallSession] = {}
TRANSCRIPTS: dict[str, Transcript] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TurnIn(BaseModel):
    speaker: Literal["CALLER", "VICTIM", "UNKNOWN"] = "UNKNOWN"
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float = 1.0
    language: str | None = None


class CreateCall(BaseModel):
    call_id: str | None = None
    call_direction: Literal["inbound", "outbound", "unknown"] = "unknown"
    turns: list[TurnIn] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    backend: str | None = None
    persist: bool = True
    realtime: bool = False


class AppendTurns(BaseModel):
    turns: list[TurnIn]


# ---------------------------------------------------------------------------
# Health / metadata
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": get_backend().name,
        "available_backends": available_backends(),
        "encryption_at_rest": get_repository().encryptor.available,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


@app.post("/api/calls", status_code=201)
def create_call(payload: CreateCall) -> dict:
    """Register a call from an already-transcribed conversation."""
    call_id = payload.call_id or f"call_{uuid.uuid4().hex[:10]}"
    transcript = transcript_from_turns(
        [turn.model_dump() for turn in payload.turns],
        call_id=call_id,
        call_direction=payload.call_direction,
    )
    TRANSCRIPTS[call_id] = transcript
    return {
        "call_id": call_id,
        "segments": len(transcript.segments),
        "duration": transcript.duration,
        "dominant_language": transcript.dominant_language,
    }


@app.post("/api/calls/upload", status_code=201)
async def upload_call(
    file: UploadFile,
    call_direction: Literal["inbound", "outbound", "unknown"] = "unknown",
) -> dict:
    """Upload a recording and run the audio pipeline over it."""
    call_id = f"call_{uuid.uuid4().hex[:10]}"
    settings.paths.storage.mkdir(parents=True, exist_ok=True)
    target = settings.paths.storage / f"{call_id}_{file.filename}"
    target.write_bytes(await file.read())

    processor = AudioProcessor()
    if processor.asr.name == "fixture" and target.suffix.lower() != ".json":
        raise HTTPException(
            status_code=503,
            detail=(
                "No speech-recognition engine installed. Run `pip install faster-whisper` "
                "(it is in requirements.txt) or POST /api/calls with a transcript."
            ),
        )
    try:
        transcript = processor.process(target, call_id=call_id, call_direction=call_direction)
    except (RuntimeError, ValueError) as exc:
        # An unsupported or corrupt upload is the client's problem, not a server fault.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    TRANSCRIPTS[call_id] = transcript
    return {
        "call_id": call_id,
        "segments": len(transcript.segments),
        "duration": transcript.duration,
        "pipeline": {
            "vad": processor.vad.name,
            "diarization": processor.diarizer.name,
            "asr": processor.asr.name,
        },
    }


@app.post("/api/calls/{call_id}/analyze")
def analyze_call(call_id: str, request: AnalyzeRequest = AnalyzeRequest()) -> dict:
    transcript = TRANSCRIPTS.get(call_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail=f"unknown call {call_id}")

    analysis = analyze_transcript(transcript, realtime=request.realtime, backend=request.backend)
    risk = assess(analysis, transcript)
    result = CallAnalysis(
        call_id=call_id,
        transcript=transcript,
        analysis=analysis,
        risk=risk,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if request.persist:
        get_repository().save(result, actor="api", is_realtime=request.realtime)
    return {
        "call_id": call_id,
        "analysis": analysis.public_json(),
        "risk": {
            "risk_score": risk.risk_score,
            "risk_level": risk.risk_level.value,
            "alert": risk.alert.model_dump(),
            "disagreement": risk.disagreement,
            "explanation": risk.explanation,
            "contributions": [c.model_dump() for c in risk.contributions],
        },
    }


@app.get("/api/calls")
def list_calls(limit: int = 50) -> dict:
    return {"calls": get_repository().list_calls(limit=limit)}


@app.get("/api/calls/{call_id}")
def get_call(call_id: str, decrypt: bool = False) -> dict:
    """Transcript + analysis. `decrypt=true` returns verbatim text and is audited."""
    payload = get_repository().get_call(call_id, actor="api", decrypt=decrypt)
    if payload is None:
        transcript = TRANSCRIPTS.get(call_id)
        if transcript is None:
            raise HTTPException(status_code=404, detail=f"unknown call {call_id}")
        return {"call": {"id": call_id, "status": "not analysed"},
                "segments": json.loads(transcript.model_dump_json())["segments"],
                "events": [], "analysis": None}
    return payload


@app.get("/api/calls/{call_id}/events")
def get_events(call_id: str) -> dict:
    return {"call_id": call_id, "events": get_repository().get_events(call_id)}


@app.get("/api/calls/{call_id}/report", response_model=None)
def get_report(call_id: str) -> dict:
    transcript = TRANSCRIPTS.get(call_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail=f"unknown call {call_id}")
    analysis = analyze_transcript(transcript)
    risk = assess(analysis, transcript)
    result = CallAnalysis(call_id=call_id, transcript=transcript, analysis=analysis, risk=risk)
    return {"call_id": call_id, "markdown": render_report(result)}


@app.delete("/api/calls/{call_id}")
def delete_call(call_id: str, hard: bool = False) -> dict:
    """Erase a call (spec section 27: ability to delete)."""
    TRANSCRIPTS.pop(call_id, None)
    SESSIONS.pop(call_id, None)
    deleted = get_repository().delete_call(call_id, actor="api", hard=hard)
    return {"call_id": call_id, "deleted": deleted, "hard": hard}


@app.post("/api/admin/purge-expired")
def purge_expired() -> dict:
    """Run the retention policy now."""
    return get_repository().purge_expired()


# ---------------------------------------------------------------------------
# Real-time streaming
# ---------------------------------------------------------------------------


@app.post("/api/calls/{call_id}/turns")
async def append_turns(call_id: str, payload: AppendTurns) -> dict:
    """Append live turns; every connected websocket receives a `risk_update`."""
    session = SESSIONS.get(call_id)
    transcript = TRANSCRIPTS.get(call_id)
    if session is None or transcript is None:
        raise HTTPException(status_code=404, detail=f"no live session for {call_id}")

    existing = [
        {
            "speaker": s.speaker.value,
            "text": s.text_original,
            "start": s.start,
            "end": s.end,
            "confidence": s.confidence,
        }
        for s in transcript.segments
    ]
    grown = transcript_from_turns(
        existing + [turn.model_dump() for turn in payload.turns],
        call_id=call_id,
        call_direction=transcript.call_direction,
    )
    TRANSCRIPTS[call_id] = grown
    update = session.ingest(grown)
    await manager.broadcast(call_id, update.to_ws_payload())
    return update.to_ws_payload()


@app.websocket("/api/calls/{call_id}/stream")
async def stream(websocket: WebSocket, call_id: str) -> None:
    """Live risk updates.

    Send `{"type": "turn", "speaker": "CALLER", "text": "..."}` frames; receive
    `risk_update` frames back. Sending `{"type": "finalize"}` returns the report.
    """
    await manager.connect(call_id, websocket)
    session = SESSIONS.setdefault(call_id, LiveCallSession(call_id=call_id))
    transcript = TRANSCRIPTS.setdefault(call_id, Transcript(call_id=call_id))
    try:
        await websocket.send_json(
            {"type": "connected", "call_id": call_id, "backend": get_backend().name}
        )
        while True:
            message = await websocket.receive_json()
            kind = message.get("type", "turn")

            if kind == "finalize":
                result = session.finalize()
                get_repository().save(result, actor="ws")
                await websocket.send_json(
                    {
                        "type": "final_report",
                        "call_id": call_id,
                        "risk_score": result.risk.risk_score,
                        "classification": result.risk.risk_level.value,
                        "analysis": result.analysis.public_json(),
                    }
                )
                continue

            if kind != "turn":
                await websocket.send_json({"type": "error", "detail": f"unknown frame {kind!r}"})
                continue

            transcript = TRANSCRIPTS[call_id]
            existing = [
                {
                    "speaker": s.speaker.value,
                    "text": s.text_original,
                    "start": s.start,
                    "end": s.end,
                    "confidence": s.confidence,
                }
                for s in transcript.segments
            ]
            existing.append(
                {
                    "speaker": message.get("speaker", "UNKNOWN"),
                    "text": message.get("text", ""),
                    "confidence": message.get("confidence", 1.0),
                }
            )
            grown = transcript_from_turns(
                existing, call_id=call_id, call_direction=transcript.call_direction
            )
            TRANSCRIPTS[call_id] = grown
            update = session.ingest(grown)
            await manager.broadcast(call_id, update.to_ws_payload())
    except WebSocketDisconnect:
        manager.disconnect(call_id, websocket)
    except Exception as exc:  # pragma: no cover - transport errors
        await websocket.send_json({"type": "error", "detail": str(exc)})
        manager.disconnect(call_id, websocket)
