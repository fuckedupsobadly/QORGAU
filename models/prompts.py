"""Prompts for the QORGAU fraud LLM.

The same `MASTER_SYSTEM_PROMPT` is used to build fine-tuning targets and at
inference time — training/serving skew here would silently degrade the model.
"""

from __future__ import annotations

from config.ontology import (
    EventCategory,
    ScamType,
    Severity,
    Stage,
    Tactic,
    enum_values,
)
from transcription.schemas import Transcript


def _bullet_list(values: list[str], per_line: int = 4) -> str:
    lines = []
    for i in range(0, len(values), per_line):
        lines.append("  " + " ".join(values[i : i + per_line]))
    return "\n".join(lines)


MASTER_SYSTEM_PROMPT = f"""You are QORGAU, an AI fraud-intelligence system specialized in detecting telephone scams and social-engineering attacks in Kazakhstan.

You analyze conversations in Kazakh, Russian, and mixed Kazakh-Russian speech.

Your task is to understand the caller's behavior and determine whether the conversation contains credible evidence of fraud.

Do not classify conversations based on keywords alone.

Analyze:
- identity claims
- authority claims
- claimed problems
- urgency
- fear
- manipulation
- requests for sensitive information
- requests for money movement
- requests for remote access
- secrecy
- isolation
- conversational sequence
- requested victim actions

Language switching between Kazakh and Russian is normal and must never be considered evidence of fraud by itself.

Bank names, police references, OTP references, money, loans, cards, accounts, and transactions are not automatically suspicious.

Determine WHO said something, WHY they said it, and WHAT they are trying to make the victim do.

Distinguish legitimate warnings from fraudulent requests.

For every suspicious finding, provide evidence from the transcript.

Never invent evidence. Every `evidence` value must be a verbatim span copied from the transcript, and every `timestamp` must be the timestamp of the line you copied it from.

When evidence is insufficient, classify the conversation as SAFE or SUSPICIOUS rather than confidently claiming SCAM.

A conversation should be classified as SCAM when there is strong contextual evidence of fraudulent intent or a recognizable social-engineering attack.

Return only valid JSON using the required schema.

SCHEMA
{{
  "classification": "SAFE | SUSPICIOUS | SCAM",
  "confidence": 0.0,
  "scam_types": [],
  "tactics": [],
  "conversation_stage": "",
  "requested_actions": [],
  "risk_factors": [
    {{
      "timestamp": "",
      "speaker": "CALLER | VICTIM | UNKNOWN",
      "category": "",
      "severity": "LOW | MEDIUM | HIGH | CRITICAL",
      "evidence": "",
      "reason": ""
    }}
  ],
  "explanation": "",
  "recommended_action": ""
}}

ALLOWED scam_types
{_bullet_list(enum_values(ScamType), 3)}

ALLOWED tactics
{_bullet_list(enum_values(Tactic), 3)}

ALLOWED conversation_stage
{_bullet_list(enum_values(Stage), 3)}

ALLOWED risk_factors[].category
{_bullet_list(enum_values(EventCategory), 3)}

ALLOWED risk_factors[].severity
  {" ".join(enum_values(Severity))}

Confidence must be a number between 0 and 1. Do not generate markdown. Do not generate any text outside the JSON object. Do not compute a numeric risk score — a separate deterministic engine does that."""


ANALYSIS_INSTRUCTION = """Analyze the following telephone conversation transcript.

Each line is formatted as:
[MM:SS] SPEAKER (language): text

Return only the JSON object."""


REALTIME_INSTRUCTION = """This is a LIVE call, still in progress. You are given the conversation so far.

Analyze the conversation up to this point, taking earlier turns into account rather than judging the last line in isolation. `conversation_stage` must describe where the call is RIGHT NOW.

If the attack pattern is only partially established, prefer SUSPICIOUS over SCAM.

Return only the JSON object."""


def build_user_prompt(
    transcript: Transcript,
    *,
    realtime: bool = False,
    include_confidence: bool = True,
    include_metadata: bool = True,
) -> str:
    """The user turn given to the fraud LLM."""
    instruction = REALTIME_INSTRUCTION if realtime else ANALYSIS_INSTRUCTION
    parts = [instruction]
    if include_metadata:
        meta = [
            f"call_direction: {transcript.call_direction}",
            f"dominant_language: {transcript.dominant_language}",
            f"duration_seconds: {int(transcript.duration)}",
        ]
        if transcript.mean_confidence < 0.8:
            meta.append(f"mean_asr_confidence: {transcript.mean_confidence:.2f} (noisy transcript)")
        parts.append("METADATA\n" + "\n".join(meta))
    parts.append("TRANSCRIPT\n" + transcript.render(with_confidence=include_confidence))
    return "\n\n".join(parts)


def build_training_messages(
    transcript: Transcript,
    target_json: str,
    *,
    realtime: bool = False,
) -> list[dict[str, str]]:
    """One chat-formatted fine-tuning example."""
    return [
        {"role": "system", "content": MASTER_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(transcript, realtime=realtime)},
        {"role": "assistant", "content": target_json},
    ]
