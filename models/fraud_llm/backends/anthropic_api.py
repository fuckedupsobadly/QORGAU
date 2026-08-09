"""Claude API backend — a prompt-only comparison baseline.

Useful for two things:

1. **Bootstrapping the corpus.** Running this over raw call transcripts produces
   draft annotations for human review, which is much cheaper than annotating
   from scratch (`training/prepare_dataset.py` keeps the human-verified label).
2. **A ceiling to measure against.** The fine-tuned adapter should approach or
   beat this on the held-out sets while being far cheaper per call and runnable
   on-premise — which matters, because call transcripts are sensitive financial
   data that many operators cannot send to a third party at all.

It is deliberately NOT the default backend: QORGAU's specification calls for a
fine-tuned model as the contextual intelligence layer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config.ontology import (
    Classification,
    EventCategory,
    ScamType,
    Severity,
    Speaker,
    Stage,
    Tactic,
    enum_values,
)
from config.settings import settings
from models.fraud_llm.backends.base import FraudLLMBackend
from models.prompts import MASTER_SYSTEM_PROMPT, build_user_prompt
from transcription.schemas import LLMAnalysis, Transcript, extract_json

logger = logging.getLogger(__name__)


def _analysis_schema() -> dict[str, Any]:
    """JSON schema for spec section 17, in the subset structured outputs accepts."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "classification",
            "confidence",
            "scam_types",
            "tactics",
            "conversation_stage",
            "requested_actions",
            "risk_factors",
            "explanation",
            "recommended_action",
        ],
        "properties": {
            "classification": {"type": "string", "enum": enum_values(Classification)},
            "confidence": {"type": "number"},
            "scam_types": {
                "type": "array",
                "items": {"type": "string", "enum": enum_values(ScamType)},
            },
            "tactics": {
                "type": "array",
                "items": {"type": "string", "enum": enum_values(Tactic)},
            },
            "conversation_stage": {"type": "string", "enum": enum_values(Stage)},
            "requested_actions": {"type": "array", "items": {"type": "string"}},
            "risk_factors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "timestamp",
                        "speaker",
                        "category",
                        "severity",
                        "evidence",
                        "reason",
                    ],
                    "properties": {
                        "timestamp": {"type": "string"},
                        "speaker": {"type": "string", "enum": enum_values(Speaker)},
                        "category": {"type": "string", "enum": enum_values(EventCategory)},
                        "severity": {"type": "string", "enum": enum_values(Severity)},
                        "evidence": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
            "explanation": {"type": "string"},
            "recommended_action": {"type": "string"},
        },
    }


class AnthropicBackend(FraudLLMBackend):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.model.anthropic_model
        self._api_key = api_key
        self._client: Any = None

    def warmup(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError(
                "AnthropicBackend needs the `anthropic` package: pip install anthropic"
            ) from exc
        # Zero-arg construction resolves ANTHROPIC_API_KEY / auth-token / CLI profile.
        self._client = (
            anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        )

    def analyze(self, transcript: Transcript, *, realtime: bool = False) -> LLMAnalysis:
        self.warmup()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=[
                {
                    "type": "text",
                    "text": MASTER_SYSTEM_PROMPT,
                    # The system prompt is byte-stable across every call, so it
                    # caches; the transcript goes after it as the volatile part.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(transcript, realtime=realtime),
                }
            ],
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": _analysis_schema()},
            },
        )

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            logger.warning("Claude declined the analysis (category=%s)", category)
            return LLMAnalysis(
                explanation=(
                    "The analysis model declined this request, so no findings are available. "
                    "The deterministic risk engine still evaluated the transcript."
                ),
                recommended_action="Review the call manually.",
                dropped_findings=[f"refusal: {category}"],
                model_backend=self.name,
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            payload = extract_json(text)
        except ValueError as exc:
            logger.error("invalid JSON from %s: %s", self.model, exc)
            return LLMAnalysis(
                explanation=f"Model output was not valid JSON ({exc}).",
                dropped_findings=[f"invalid_json: {text[:200]}"],
                model_backend=self.name,
            )
        analysis = LLMAnalysis.model_validate(payload)
        return self.ground(analysis, transcript)

    # -- corpus bootstrapping -------------------------------------------
    def draft_annotation(self, transcript: Transcript) -> str:
        """Pretty-printed JSON for a human annotator to correct (spec section 24)."""
        return json.dumps(self.analyze(transcript).public_json(), ensure_ascii=False, indent=2)
