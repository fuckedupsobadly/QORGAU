"""Central configuration. Every module reads paths/thresholds from here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Risk thresholds (spec sections 20 & 22)
# ---------------------------------------------------------------------------

RISK_BANDS: list[tuple[int, int, str]] = [
    (0, 29, "SAFE"),
    (30, 59, "SUSPICIOUS"),
    (60, 79, "HIGH_RISK"),
    (80, 100, "CRITICAL"),
]

ALERT_MONITOR_THRESHOLD = 30
ALERT_WARNING_THRESHOLD = 60
ALERT_CRITICAL_THRESHOLD = 80


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    datasets: Path = ROOT / "datasets"
    raw: Path = ROOT / "datasets" / "raw"
    processed: Path = ROOT / "datasets" / "processed"
    train: Path = ROOT / "datasets" / "train"
    validation: Path = ROOT / "datasets" / "validation"
    test: Path = ROOT / "datasets" / "test"
    artifacts: Path = ROOT / "artifacts"
    adapters: Path = ROOT / "artifacts" / "adapters"
    reports: Path = ROOT / "artifacts" / "reports"
    storage: Path = ROOT / "artifacts" / "storage"

    def ensure(self) -> None:
        for value in vars(self).values():
            if isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    #: Real-time analysis window. The pipeline emits a transcript chunk roughly
    #: every `chunk_seconds`; the LLM re-analyses on every chunk.
    chunk_seconds: float = 8.0
    vad_backend: str = os.environ.get("QORGAU_VAD", "energy")  # energy | silero
    diarization_backend: str = os.environ.get("QORGAU_DIAR", "heuristic")  # heuristic | pyannote
    asr_backend: str = os.environ.get("QORGAU_ASR", "auto")  # auto | faster_whisper | whisper | fixture
    asr_model: str = os.environ.get("QORGAU_ASR_MODEL", "large-v3")
    #: Segments below this ASR confidence are still analysed, but the risk engine
    #: dampens the score and the UI marks them as low confidence.
    low_confidence_threshold: float = 0.55


@dataclass(frozen=True)
class ModelConfig:
    #: reference | local_adapter | anthropic
    backend: str = os.environ.get("QORGAU_LLM_BACKEND", "reference")
    base_model: str = os.environ.get("QORGAU_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    adapter_path: str = os.environ.get("QORGAU_ADAPTER", str(ROOT / "artifacts" / "adapters" / "qorgau-lora"))
    anthropic_model: str = os.environ.get("QORGAU_ANTHROPIC_MODEL", "claude-sonnet-5")
    max_new_tokens: int = 1024
    temperature: float = 0.0
    #: How many previous turns of context the incremental analyser keeps.
    context_window_turns: int = 40


@dataclass(frozen=True)
class SecurityConfig:
    mask_credentials_in_storage: bool = _env_flag("QORGAU_MASK_PII", True)
    encrypt_at_rest: bool = _env_flag("QORGAU_ENCRYPT_AT_REST", True)
    #: Days before recordings/transcripts are purged by the retention job.
    recording_retention_days: int = int(os.environ.get("QORGAU_RECORDING_RETENTION_DAYS", "30"))
    transcript_retention_days: int = int(os.environ.get("QORGAU_TRANSCRIPT_RETENTION_DAYS", "180"))
    audit_log_path: Path = ROOT / "artifacts" / "storage" / "audit.log"
    #: Fernet key material lives outside the repo in real deployments.
    encryption_key_env: str = "QORGAU_ENCRYPTION_KEY"


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get(
        "QORGAU_DATABASE_URL", f"sqlite:///{ROOT / 'artifacts' / 'storage' / 'qorgau.db'}"
    )
    paths: Paths = field(default_factory=Paths)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    random_seed: int = 20260809


settings = Settings()


def risk_level(score: int) -> str:
    """Map a 0-100 integer score onto a risk band label."""
    for low, high, label in RISK_BANDS:
        if low <= score <= high:
            return label
    return "CRITICAL" if score > 100 else "SAFE"
