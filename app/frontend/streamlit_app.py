"""QORGAU dashboard (spec section 29).

Run from the repository root:

    streamlit run app/frontend/streamlit_app.py

Three modes:

* **Analyse a call** — batch analysis of a corpus call, an uploaded recording, or
  a pasted transcript, with the evidence timeline and the risk breakdown.
* **Live call (real-time)** — replays a call turn by turn through the incremental
  session, so you can watch the risk score and stage move as the attack develops.
* **Model evaluation** — the held-out metrics, per slice.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.ontology import HARMFUL_ACTION_EVENTS  # noqa: E402
from config.settings import settings  # noqa: E402
from models.inference import available_backends, get_backend  # noqa: E402
from realtime.session import LiveCallSession, growing_transcripts  # noqa: E402
from risk.engine import assess, render_report  # noqa: E402
from transcription.processor import AudioProcessor, transcript_from_turns  # noqa: E402
from transcription.schemas import CallAnalysis, Transcript  # noqa: E402

st.set_page_config(page_title="QORGAU — voice scam detection", page_icon="🛡", layout="wide")

SEVERITY_DOT = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🟠", "CRITICAL": "🔴"}
LEVEL_BADGE = {
    "SAFE": "🟢 SAFE",
    "SUSPICIOUS": "🟡 SUSPICIOUS",
    "HIGH_RISK": "🚨 HIGH RISK",
    "CRITICAL": "🚨 CRITICAL",
}
CLASSIFICATION_BADGE = {"SAFE": "🟢 SAFE", "SUSPICIOUS": "🟡 SUSPICIOUS", "SCAM": "🚨 SCAM"}
LANG_LABEL = {"kk": "KZ", "ru": "RU", "mixed": "KZ/RU", "unknown": "??"}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def list_fixtures() -> list[dict]:
    directory = settings.paths.processed / "fixtures"
    if not directory.exists():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append(
            {
                "path": str(path),
                "call_id": payload.get("call_id", path.stem),
                "label": payload.get("family_label", path.stem),
                "language_mode": payload.get("language_mode", "?"),
                "noisy": payload.get("asr_noisy", False),
                "expected": payload.get("expected_classification", "?"),
                "segments": payload.get("segments", []),
                "call_direction": payload.get("call_direction", "unknown"),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def load_eval_reports() -> dict[str, dict]:
    reports = {}
    if settings.paths.reports.exists():
        for path in sorted(settings.paths.reports.glob("eval_*.json")):
            try:
                reports[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return reports


def transcript_from_fixture(fixture: dict) -> Transcript:
    return transcript_from_turns(
        [
            {
                "speaker": s["speaker"],
                "text": s["text"],
                "start": s.get("start"),
                "end": s.get("end"),
                "confidence": s.get("confidence", 1.0),
                "language": s.get("language"),
            }
            for s in fixture["segments"]
        ],
        call_id=fixture["call_id"],
        call_direction=fixture.get("call_direction", "unknown"),
    )


def parse_pasted(text: str, direction: str) -> Transcript:
    """`CALLER: ...` / `VICTIM: ...` per line."""
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        speaker, _, body = line.partition(":")
        speaker = speaker.strip().upper()
        if speaker not in {"CALLER", "VICTIM", "UNKNOWN"} or not body.strip():
            speaker, body = "UNKNOWN", line
        turns.append({"speaker": speaker, "text": body.strip()})
    return transcript_from_turns(turns, call_id="pasted_call", call_direction=direction)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def risk_bar(score: int, width: int = 20) -> str:
    filled = int(round(score / 100 * width))
    return "█" * filled + "░" * (width - filled)


def render_verdict(risk, analysis, transcript: Transcript) -> None:
    cols = st.columns([1.3, 1, 1, 1.4])
    with cols[0]:
        st.caption("RISK")
        st.markdown(f"### `{risk_bar(risk.risk_score)}`")
        st.markdown(f"## {risk.risk_score}/100")
    with cols[1]:
        st.caption("RISK LEVEL")
        st.markdown(f"## {LEVEL_BADGE.get(risk.risk_level.value, risk.risk_level.value)}")
        st.caption(f"alert: **{risk.alert.level}**")
    with cols[2]:
        st.caption("MODEL CLASSIFICATION")
        st.markdown(f"## {CLASSIFICATION_BADGE.get(analysis.classification.value, '?')}")
        st.caption(f"confidence {analysis.confidence:.2f} · `{analysis.model_backend}`")
    with cols[3]:
        st.caption("SCAM TYPE")
        if analysis.scam_types:
            st.markdown("\n".join(f"- {t.replace('_', ' ').title()}" for t in analysis.scam_types))
        else:
            st.markdown("_none detected_")
        st.caption(
            f"stage: **{analysis.conversation_stage}** · {LANG_LABEL.get(transcript.dominant_language, '?')}"
            f" · {int(transcript.duration)}s · ASR {transcript.mean_confidence:.2f}"
        )

    if risk.alert.level == "CRITICAL":
        st.error(f"**{risk.alert.headline}** — {risk.alert.detail}", icon="🚨")
    elif risk.alert.level == "WARNING":
        st.warning(f"**{risk.alert.headline}** — {risk.alert.detail}", icon="⚠️")
    elif risk.alert.level == "MONITOR":
        st.info(f"**{risk.alert.headline}** — {risk.alert.detail}", icon="👁")
    if risk.disagreement:
        st.warning(f"**Model / rule-engine disagreement.** {risk.disagreement}", icon="🔀")


def render_timeline(analysis, key_prefix: str = "tl") -> None:
    st.subheader("Timeline")
    events = sorted(analysis.risk_factors, key=lambda e: e.timestamp)
    if not events:
        st.caption("No events detected.")
        return
    for index, event in enumerate(events):
        dot = SEVERITY_DOT.get(event.severity.value, "⚪")
        label = event.category.replace("_", " ")
        if event.category in HARMFUL_ACTION_EVENTS:
            label = f"**{label}**"
        cols = st.columns([0.85, 3.2, 1])
        cols[0].markdown(f"`{event.timestamp}`")
        cols[1].markdown(f"{dot} {label} · _{event.speaker.value}_")
        if cols[2].button("jump", key=f"{key_prefix}_{index}", width="stretch"):
            st.session_state["focus_segment"] = event.segment_index
            st.session_state["focus_ts"] = event.timestamp


def render_transcript(transcript: Transcript, analysis) -> None:
    st.subheader("Transcript")
    focus = st.session_state.get("focus_segment")
    events_by_segment: dict[int, list] = {}
    for event in analysis.risk_factors:
        if event.segment_index is not None:
            events_by_segment.setdefault(event.segment_index, []).append(event)

    for segment in transcript.segments:
        marks = events_by_segment.get(segment.index, [])
        badge = " ".join(
            f"{SEVERITY_DOT.get(e.severity.value, '⚪')}`{e.category}`" for e in marks
        )
        lang = LANG_LABEL.get(segment.language.value, "??")
        low = " ⚠️low-ASR" if segment.confidence < settings.audio.low_confidence_threshold else ""
        header = f"`{segment.timestamp}` **{segment.speaker.value}** ·{lang}·{low} {badge}"
        body = segment.text
        if focus == segment.index:
            st.markdown(f"> {header}\n>\n> **➡ {body}**")
        else:
            st.markdown(f"{header}\n\n{body}")
        if segment.text_original != segment.text or segment.normalization_notes:
            with st.expander("original ASR + normalization notes", expanded=False):
                st.code(segment.text_original, language=None)
                for note in segment.normalization_notes:
                    st.caption(f"· {note}")


def render_breakdown(risk, analysis) -> None:
    st.subheader("Why this score")
    st.caption(risk.explanation)
    rows = [
        {
            "points": f"{c.points:+.1f}",
            "rule": c.label,
            "kind": c.kind,
            "detail": c.detail,
            "evidence": c.evidence[:90],
        }
        for c in sorted(risk.contributions, key=lambda c: -abs(c.points))
    ]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    if analysis.requested_actions:
        st.markdown("**Actions the caller requested**")
        for action in analysis.requested_actions:
            st.markdown(f"- {action}")
    st.markdown("**Model explanation**")
    st.write(analysis.explanation)
    st.markdown("**Recommended action**")
    st.info(analysis.recommended_action)
    if analysis.dropped_findings:
        with st.expander(f"{len(analysis.dropped_findings)} discarded finding(s) — evidence not in transcript"):
            for item in analysis.dropped_findings:
                st.caption(f"· {item}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🛡 QORGAU")
st.sidebar.caption("Kazakh / Russian voice scam detection")

backend_names = available_backends()
backend_choice = st.sidebar.selectbox(
    "Analysis backend",
    backend_names,
    index=backend_names.index(settings.model.backend) if settings.model.backend in backend_names else 0,
    help=(
        "`local_adapter` is the fine-tuned model (needs requirements-ml.txt + a trained adapter). "
        "`reference` is the offline behavioural analyser used when no weights are present. "
        "`anthropic` is a prompt-only API baseline."
    ),
)
active_backend = get_backend(backend_choice)
if active_backend.name != backend_choice:
    st.sidebar.warning(
        f"`{backend_choice}` unavailable — running on `{active_backend.name}`.", icon="⚠️"
    )
else:
    st.sidebar.success(f"backend: `{active_backend.name}`")

mode = st.sidebar.radio("Mode", ("Analyse a call", "Live call (real-time)", "Model evaluation"))

fixtures = list_fixtures()
if not fixtures:
    st.sidebar.error("No corpus found.")
    if st.sidebar.button("Build corpus now"):
        from training.prepare_dataset import prepare

        with st.spinner("generating conversations..."):
            prepare()
        list_fixtures.clear()
        st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    f"encryption at rest: "
    f"{'on' if settings.security.encrypt_at_rest else 'off'} · "
    f"retention {settings.security.recording_retention_days}d audio / "
    f"{settings.security.transcript_retention_days}d transcript"
)


# ---------------------------------------------------------------------------
# Mode: analyse a call
# ---------------------------------------------------------------------------


def source_picker(key: str) -> Transcript | None:
    source = st.radio(
        "Call source",
        ("Corpus call", "Upload recording", "Paste transcript"),
        horizontal=True,
        key=f"src_{key}",
    )
    if source == "Corpus call":
        if not fixtures:
            st.info("Build the corpus first (sidebar).")
            return None
        languages = sorted({f["language_mode"] for f in fixtures})
        cols = st.columns([1, 1, 3])
        language = cols[0].selectbox("Language", ["all"] + languages, key=f"lang_{key}")
        noisy = cols[1].selectbox("ASR", ["all", "clean", "noisy"], key=f"noise_{key}")
        pool = [
            f
            for f in fixtures
            if (language == "all" or f["language_mode"] == language)
            and (noisy == "all" or (f["noisy"] if noisy == "noisy" else not f["noisy"]))
        ]
        if not pool:
            st.info("No calls match that filter.")
            return None
        labels = {
            f"{f['expected']:10} · {f['label']} · {f['language_mode']}{' · noisy' if f['noisy'] else ''}": f
            for f in pool
        }
        picked = cols[2].selectbox("Call", list(labels), key=f"call_{key}")
        return transcript_from_fixture(labels[picked])

    if source == "Upload recording":
        upload = st.file_uploader("WAV / MP3 / M4A / OGG", type=["wav", "mp3", "m4a", "ogg", "flac"])
        direction = st.selectbox(
            "Call direction", ("outbound", "inbound", "unknown"),
            help="`outbound` = someone called the victim. Metadata only, never a verdict.",
            key=f"dir_{key}",
        )
        if upload is None:
            return None
        temp = settings.paths.storage / f"upload_{upload.name}"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(upload.getbuffer())
        processor = AudioProcessor()
        st.caption(
            f"pipeline: VAD `{processor.vad.name}` → diarization `{processor.diarizer.name}` "
            f"→ ASR `{processor.asr.name}`"
        )
        if processor.asr.name == "fixture":
            st.error(
                "No speech-recognition engine is installed, so an audio file cannot be "
                "transcribed. Install `faster-whisper` (see requirements-ml.txt), or use a "
                "corpus call / pasted transcript.",
                icon="🚫",
            )
            return None
        with st.spinner("VAD → diarization → ASR..."):
            return processor.process(temp, call_direction=direction)

    default = (
        "CALLER: Сәлеметсіз бе, это служба безопасности банка Halyk.\n"
        "VICTIM: Иә, тыңдап тұрмын.\n"
        "CALLER: На ваше имя оформляется кредит на 3 миллиона тенге.\n"
        "VICTIM: Но я ничего не оформлял.\n"
        "CALLER: Именно поэтому нужно срочно отменить операцию.\n"
        "CALLER: Қазір вам СМС келеді. Кодты айтып жіберіңіз.\n"
    )
    text = st.text_area("One turn per line, `CALLER:` / `VICTIM:`", value=default, height=200,
                        key=f"paste_{key}")
    direction = st.selectbox("Call direction", ("outbound", "inbound", "unknown"), key=f"pdir_{key}")
    return parse_pasted(text, direction) if text.strip() else None


if mode == "Analyse a call":
    st.title("Call analysis")
    transcript = source_picker("batch")
    if transcript and transcript.segments:
        with st.spinner("analysing..."):
            analysis = active_backend.analyze(transcript)
            risk = assess(analysis, transcript)
        st.caption("CALL STATUS 🟢 ANALYSED")
        render_verdict(risk, analysis, transcript)
        st.divider()
        left, right = st.columns([1.1, 2])
        with left:
            render_timeline(analysis)
        with right:
            render_transcript(transcript, analysis)
        st.divider()
        render_breakdown(risk, analysis)

        result = CallAnalysis(
            call_id=transcript.call_id, transcript=transcript, analysis=analysis, risk=risk
        )
        st.divider()
        cols = st.columns([1, 1, 2])
        cols[0].download_button(
            "⬇ Investigation report (.md)",
            render_report(result),
            file_name=f"qorgau_{transcript.call_id}.md",
            width="stretch",
        )
        cols[1].download_button(
            "⬇ Analysis JSON",
            json.dumps(analysis.public_json(), ensure_ascii=False, indent=2),
            file_name=f"qorgau_{transcript.call_id}.json",
            width="stretch",
        )
        if cols[2].button("💾 Save to case database", width="stretch"):
            from database.repository import get_repository

            repository = get_repository()
            repository.save(result, actor="streamlit-ui")
            st.success(
                f"Saved `{transcript.call_id}`. Credentials masked; "
                f"verbatim text {'encrypted' if repository.encryptor.available else 'NOT stored (no key set)'}."
            )
    elif transcript is not None:
        st.info("That call produced no speech segments.")


# ---------------------------------------------------------------------------
# Mode: live call
# ---------------------------------------------------------------------------

elif mode == "Live call (real-time)":
    st.title("Live call — incremental analysis")
    st.caption(
        "The same analysis stack, re-run on every new chunk of transcript. Risk uses a peak-hold: "
        "an OTP request at 00:42 still matters at 01:30."
    )
    transcript = source_picker("live")
    speed = st.slider("Replay delay per turn (seconds)", 0.0, 2.0, 0.35, 0.05)

    if transcript and transcript.segments and st.button("▶ Start live analysis", type="primary"):
        session = LiveCallSession(
            call_id=transcript.call_id,
            call_direction=transcript.call_direction,
            backend=backend_choice,
        )
        status = st.empty()
        meter = st.empty()
        alert_box = st.empty()
        state_cols = st.columns(3)
        stage_box, tactics_box, events_box = (c.empty() for c in state_cols)
        st.divider()
        feed = st.container()
        rendered = 0

        for update in session.replay(growing_transcripts(transcript)):
            status.markdown(
                f"**CALL STATUS** 🟢 ANALYSING · turn {update.state.segments_seen}/"
                f"{len(transcript.segments)}"
            )
            meter.markdown(
                f"### `{risk_bar(update.risk.risk_score)}` **{update.risk.risk_score}/100** — "
                f"{LEVEL_BADGE.get(update.risk.risk_level.value, '')}"
                f"  ·  peak **{update.state.risk_score}**"
            )
            stage_box.metric("Stage", update.state.current_stage)
            tactics_box.metric("Tactics", len(update.state.detected_tactics))
            events_box.metric("Events", len(update.state.events))

            if update.alert and update.alert.level in {"CRITICAL", "WARNING"}:
                render = alert_box.error if update.alert.level == "CRITICAL" else alert_box.warning
                render(f"**{update.alert.headline}** — {update.alert.detail}", icon="🚨")

            for segment in transcript.segments[rendered : update.state.segments_seen]:
                new = [
                    e for e in update.new_events if e.segment_index == segment.index
                ]
                marks = " ".join(
                    f"{SEVERITY_DOT.get(e.severity.value, '⚪')}`{e.category}`" for e in new
                )
                feed.markdown(
                    f"`{segment.timestamp}` **{segment.speaker.value}** "
                    f"·{LANG_LABEL.get(segment.language.value, '??')}· {marks}\n\n{segment.text}"
                )
            rendered = update.state.segments_seen
            if speed:
                time.sleep(speed)

        st.divider()
        st.subheader("Final report")
        final = session.finalize()
        render_verdict(final.risk, final.analysis, final.transcript)
        if final.stage_timeline:
            st.markdown("**Stage progression**")
            st.markdown(
                "  →  ".join(f"`{e['timestamp']}` {e['stage']}" for e in final.stage_timeline)
            )
        render_breakdown(final.risk, final.analysis)
        st.download_button(
            "⬇ Investigation report (.md)",
            render_report(final),
            file_name=f"qorgau_{final.call_id}.md",
        )


# ---------------------------------------------------------------------------
# Mode: evaluation
# ---------------------------------------------------------------------------

else:
    st.title("Model evaluation")
    st.caption(
        "Held-out results. Splits are by **script family**, so nothing in test is a paraphrase "
        "of something in train. Priority metrics are scam recall and false-positive rate."
    )
    reports = load_eval_reports()
    if not reports:
        st.info(
            "No evaluation reports yet. Run:\n\n"
            "```bash\npython -m training.prepare_dataset\npython -m training.evaluate --split test\n```"
        )
    else:
        choice = st.selectbox("Report", list(reports))
        report = reports[choice]
        overall = report["overall"]
        cols = st.columns(4)
        cols[0].metric("Scam recall (risk ≥ 60)", f"{overall['scam_recall_risk_engine']:.3f}")
        cols[1].metric("False-positive rate", f"{overall['false_positive_rate_risk_engine']:.3f}")
        cols[2].metric("Exact classification", f"{overall['exact_classification_accuracy']:.3f}")
        cols[3].metric("Evidence grounding", f"{overall['evidence_grounding_rate']:.3f}")
        cols = st.columns(4)
        cols[0].metric("Scam-type F1", f"{overall['scam_type']['f1']:.3f}")
        cols[1].metric("Tactic F1", f"{overall['tactic']['f1']:.3f}")
        cols[2].metric("JSON validity", f"{overall['json_validity']:.3f}")
        cols[3].metric("Mean latency", f"{overall['mean_latency_ms']:.0f} ms")

        st.subheader("Per slice")
        st.dataframe(
            [
                {
                    "slice": name,
                    "n": scores["n"],
                    "scam recall": round(scores["scam_recall_risk_engine"], 3),
                    "false-pos rate": round(scores["false_positive_rate_risk_engine"], 3),
                    "exact acc": round(scores["exact_classification_accuracy"], 3),
                    "tactic F1": round(scores["tactic"]["f1"], 3),
                    "mean risk (scam)": scores["mean_risk_scam"],
                    "mean risk (legit)": scores["mean_risk_legitimate"],
                }
                for name, scores in report["by_slice"].items()
            ],
            width="stretch",
            hide_index=True,
        )
        st.subheader("Risk band distribution")
        st.dataframe(
            [
                {"band": band, **counts}
                for band, counts in overall.get("risk_band_distribution", {}).items()
            ],
            width="stretch",
            hide_index=True,
        )
        if report.get("failures"):
            st.subheader(f"Misses and false alarms ({len(report['failures'])})")
            st.dataframe(report["failures"], width="stretch", hide_index=True)
        else:
            st.success("No misses or false alarms on this split.")
