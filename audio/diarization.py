"""Speaker diarization — who said what (spec section 5).

This is not a nice-to-have. "Никому не сообщайте код" and "Сообщите мне код" share
most of their vocabulary and mean opposite things; without a speaker label the
analysis layer cannot tell a warning from an attack. Every downstream component
therefore refuses to attribute a harmful request to an unlabelled voice.

Two speakers are assumed (CALLER + VICTIM), which is the shape of a phone call.
Additional voices are labelled UNKNOWN rather than forced into a role.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from audio.ingestion import AudioBuffer
from audio.vad import SpeechRegion
from config.ontology import Speaker

logger = logging.getLogger(__name__)


@dataclass
class DiarizedRegion:
    start: float
    end: float
    speaker: Speaker
    cluster: int = 0
    confidence: float = 0.5


class Diarizer(ABC):
    name = "base"

    @abstractmethod
    def diarize(
        self,
        audio: AudioBuffer,
        regions: list[SpeechRegion],
        *,
        call_direction: str = "unknown",
    ) -> list[DiarizedRegion]:
        ...

    # -- shared role assignment ----------------------------------------
    @staticmethod
    def assign_roles(
        clusters: list[int], call_direction: str
    ) -> dict[int, Speaker]:
        """Map acoustic clusters onto conversational roles.

        On an inbound call the victim dialled, so the victim usually speaks second
        (after the institution's greeting is heard); on an unsolicited inbound call
        *to* the victim, the caller speaks first. `call_direction` disambiguates:
        for `outbound` (someone called the victim) the first voice is the CALLER.
        Anything beyond two clusters stays UNKNOWN.
        """
        ordered: list[int] = []
        for cluster in clusters:
            if cluster not in ordered:
                ordered.append(cluster)
        mapping: dict[int, Speaker] = {}
        if not ordered:
            return mapping
        if call_direction == "inbound":
            # The victim placed the call, so the victim speaks first.
            roles = [Speaker.VICTIM, Speaker.CALLER]
        else:
            roles = [Speaker.CALLER, Speaker.VICTIM]
        for index, cluster in enumerate(ordered):
            mapping[cluster] = roles[index] if index < len(roles) else Speaker.UNKNOWN
        return mapping


class HeuristicDiarizer(Diarizer):
    """Two-speaker clustering on cheap acoustic features.

    Each speech region is described by mean log-energy and a zero-crossing-rate
    proxy for pitch, then 2-means-clustered. This is genuinely weaker than a
    neural diarizer on cross-talk — it exists so the pipeline runs anywhere, and
    it reports a per-region confidence so the UI can flag uncertain turns.
    """

    name = "heuristic"

    def diarize(
        self,
        audio: AudioBuffer,
        regions: list[SpeechRegion],
        *,
        call_direction: str = "unknown",
    ) -> list[DiarizedRegion]:
        if not regions:
            return []
        features = [self._features(audio, region) for region in regions]
        if len(regions) == 1:
            mapping = self.assign_roles([0], call_direction)
            return [DiarizedRegion(regions[0].start, regions[0].end, mapping[0], 0, 0.4)]

        labels, separation = self._two_means(features)
        mapping = self.assign_roles(labels, call_direction)
        confidence = min(0.9, 0.45 + separation)
        return [
            DiarizedRegion(
                region.start,
                region.end,
                mapping.get(label, Speaker.UNKNOWN),
                label,
                round(confidence, 2),
            )
            for region, label in zip(regions, labels)
        ]

    @staticmethod
    def _features(audio: AudioBuffer, region: SpeechRegion) -> tuple[float, float]:
        chunk = audio.slice_seconds(region.start, region.end).samples
        if not chunk:
            return 0.0, 0.0
        rms = math.sqrt(sum(value * value for value in chunk) / len(chunk))
        crossings = sum(
            1 for a, b in zip(chunk, chunk[1:]) if (a >= 0) != (b >= 0)
        ) / max(1, len(chunk) - 1)
        return 20 * math.log10(rms + 1e-9), crossings * 100

    @staticmethod
    def _two_means(points: list[tuple[float, float]], iterations: int = 25) -> tuple[list[int], float]:
        lo = min(points, key=lambda p: (p[0], p[1]))
        hi = max(points, key=lambda p: (p[0], p[1]))
        centroids = [list(lo), list(hi)]
        labels = [0] * len(points)
        for _ in range(iterations):
            changed = False
            for index, point in enumerate(points):
                distances = [
                    (point[0] - c[0]) ** 2 + (point[1] - c[1]) ** 2 for c in centroids
                ]
                best = 0 if distances[0] <= distances[1] else 1
                if labels[index] != best:
                    labels[index] = best
                    changed = True
            for cluster in (0, 1):
                members = [p for p, label in zip(points, labels) if label == cluster]
                if members:
                    centroids[cluster] = [
                        sum(m[0] for m in members) / len(members),
                        sum(m[1] for m in members) / len(members),
                    ]
            if not changed:
                break
        spread = math.dist(centroids[0], centroids[1])
        scale = max(1e-6, max(p[0] for p in points) - min(p[0] for p in points))
        return labels, min(0.45, spread / (scale * 4))


class PyannoteDiarizer(Diarizer):
    """`pyannote.audio` speaker diarization (needs a HF token and a GPU to be fast)."""

    name = "pyannote"

    def __init__(self, model: str = "pyannote/speaker-diarization-3.1", token: str | None = None) -> None:
        self.model = model
        self.token = token
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(self.model, use_auth_token=self.token)

    def diarize(
        self,
        audio: AudioBuffer,
        regions: list[SpeechRegion],
        *,
        call_direction: str = "unknown",
    ) -> list[DiarizedRegion]:
        try:
            self._load()
            import torch

            waveform = torch.tensor([list(audio.samples)], dtype=torch.float32)
            annotation = self._pipeline(
                {"waveform": waveform, "sample_rate": audio.sample_rate}, num_speakers=2
            )
            spans: list[tuple[float, float, str]] = [
                (turn.start, turn.end, label)
                for turn, _, label in annotation.itertracks(yield_label=True)
            ]
            clusters = {label: index for index, label in enumerate(dict.fromkeys(s[2] for s in spans))}
            mapping = self.assign_roles([clusters[s[2]] for s in spans], call_direction)
            return [
                DiarizedRegion(start, end, mapping.get(clusters[label], Speaker.UNKNOWN),
                               clusters[label], 0.85)
                for start, end, label in spans
            ]
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("pyannote unavailable (%s); falling back to heuristic diarizer", exc)
            return HeuristicDiarizer().diarize(audio, regions, call_direction=call_direction)


def build_diarizer(backend: str | None = None) -> Diarizer:
    from config.settings import settings

    choice = (backend or settings.audio.diarization_backend).lower()
    return PyannoteDiarizer() if choice == "pyannote" else HeuristicDiarizer()
