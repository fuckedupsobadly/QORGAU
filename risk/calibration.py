"""Threshold and weight calibration against the labelled corpus.

The risk engine's numbers are policy, but policy should be checked against data.
This module sweeps the decision threshold over a labelled set and reports the
recall / false-positive trade-off, because for this application those two metrics
are what matter (spec section 26) — not overall accuracy.

Nothing here changes behaviour automatically. It produces a report; a human
decides whether to edit `risk/rules.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from config.settings import RISK_BANDS


@dataclass
class ScoredCall:
    call_id: str
    risk_score: int
    is_scam: bool
    slice_name: str = "all"


@dataclass
class ThresholdReport:
    threshold: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def scam_recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.scam_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
            "scam_recall": round(self.scam_recall, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class CalibrationResult:
    thresholds: list[ThresholdReport] = field(default_factory=list)
    band_distribution: dict[str, dict[str, int]] = field(default_factory=dict)

    def best_threshold(self, *, max_false_positive_rate: float = 0.05) -> ThresholdReport | None:
        """Highest-recall threshold that stays under the FPR budget."""
        eligible = [t for t in self.thresholds if t.false_positive_rate <= max_false_positive_rate]
        if not eligible:
            return None
        return max(eligible, key=lambda t: (t.scam_recall, -t.false_positive_rate))

    def as_dict(self) -> dict[str, object]:
        return {
            "thresholds": [t.as_dict() for t in self.thresholds],
            "band_distribution": self.band_distribution,
        }


def evaluate_thresholds(
    calls: Sequence[ScoredCall], thresholds: Iterable[int] | None = None
) -> CalibrationResult:
    grid = list(thresholds) if thresholds is not None else list(range(10, 100, 5))
    result = CalibrationResult()
    for threshold in grid:
        report = ThresholdReport(threshold, 0, 0, 0, 0)
        for call in calls:
            flagged = call.risk_score >= threshold
            if call.is_scam and flagged:
                report.true_positives += 1
            elif call.is_scam:
                report.false_negatives += 1
            elif flagged:
                report.false_positives += 1
            else:
                report.true_negatives += 1
        result.thresholds.append(report)

    for _, _, label in RISK_BANDS:
        result.band_distribution[label] = {"scam": 0, "legitimate": 0}
    for call in calls:
        label = next(
            (name for low, high, name in RISK_BANDS if low <= call.risk_score <= high), "CRITICAL"
        )
        result.band_distribution[label]["scam" if call.is_scam else "legitimate"] += 1
    return result


def write_report(result: CalibrationResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return path
