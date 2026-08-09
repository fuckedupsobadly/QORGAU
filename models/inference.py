"""Backend selection + the single entry point the rest of QORGAU calls."""

from __future__ import annotations

import logging
from functools import lru_cache

from config.settings import settings
from models.fraud_llm.backends.base import FraudLLMBackend
from transcription.schemas import LLMAnalysis, Transcript

logger = logging.getLogger(__name__)

#: name -> factory. Imports are lazy so a missing ML dependency never breaks the app.
_BACKENDS: dict[str, str] = {
    "reference": "models.fraud_llm.backends.reference:ReferenceAnalyzer",
    "local_adapter": "models.fraud_llm.backends.local_adapter:LocalAdapterBackend",
    "anthropic": "models.fraud_llm.backends.anthropic_api:AnthropicBackend",
}


def available_backends() -> list[str]:
    return list(_BACKENDS)


def build_backend(name: str | None = None) -> FraudLLMBackend:
    """Instantiate a backend, falling back to `reference` with a loud warning."""
    target = (name or settings.model.backend).strip()
    if target not in _BACKENDS:
        raise ValueError(f"unknown LLM backend {target!r}; choose from {available_backends()}")
    module_path, class_name = _BACKENDS[target].split(":")
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)()
    except Exception as exc:
        if target == "reference":
            raise
        logger.warning(
            "backend %r unavailable (%s) — falling back to the reference analyser. "
            "Findings will be lexicon-bound rather than model-generated.",
            target,
            exc,
        )
        from models.fraud_llm.backends.reference import ReferenceAnalyzer

        return ReferenceAnalyzer()


@lru_cache(maxsize=4)
def get_backend(name: str | None = None) -> FraudLLMBackend:
    """Process-wide singleton per backend name (weights load once)."""
    return build_backend(name)


def analyze_transcript(
    transcript: Transcript,
    *,
    realtime: bool = False,
    backend: str | None = None,
) -> LLMAnalysis:
    """Run the fraud LLM over a transcript.

    Everything downstream (risk engine, API, UI, evaluator) goes through here, so
    swapping the intelligence layer is a one-line configuration change.
    """
    return get_backend(backend).analyze(transcript, realtime=realtime)
