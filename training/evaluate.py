"""Evaluate any analysis backend on the held-out sets (spec sections 24 & 26).

Metric priorities are set by the application, not by convention: **scam recall**
and **false-positive rate** first. A detector that misses scams is useless; a
detector that flags every legitimate bank call is also useless, and would be
switched off by its users within a week. Overall accuracy is reported but is not
the number to optimise.

Everything is also reported **per slice** — Russian, Kazakh, mixed, legitimate,
obvious, subtle, ASR-noisy, unseen-pattern — because an aggregate number hides
exactly the failures that matter (e.g. good Russian, poor Kazakh).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from config.settings import ALERT_WARNING_THRESHOLD, settings
from models.inference import get_backend
from risk.calibration import ScoredCall, evaluate_thresholds
from risk.engine import assess
from transcription.processor import transcript_from_turns
from transcription.schemas import Transcript


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _set_micro(pairs: Sequence[tuple[set[str], set[str]]]) -> dict[str, float]:
    """Micro-averaged P/R/F1 over set-valued predictions, plus exact-set match."""
    tp = fp = fn = 0
    exact = 0
    for predicted, gold in pairs:
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        exact += int(predicted == gold)
    scores = _prf(tp, fp, fn)
    scores["exact_set_match"] = round(exact / len(pairs), 4) if pairs else 0.0
    return scores


@dataclass
class CaseResult:
    call_id: str
    slices: list[str]
    gold_classification: str
    predicted_classification: str
    risk_score: int
    risk_level: str
    gold_is_scam: bool
    flagged_by_model: bool
    flagged_by_risk: bool
    json_valid: bool
    gold_scam_types: set[str] = field(default_factory=set)
    predicted_scam_types: set[str] = field(default_factory=set)
    gold_tactics: set[str] = field(default_factory=set)
    predicted_tactics: set[str] = field(default_factory=set)
    gold_categories: set[str] = field(default_factory=set)
    predicted_categories: set[str] = field(default_factory=set)
    grounded_findings: int = 0
    total_findings: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def load_split(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m training.prepare_dataset` first"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transcript_of(record: dict) -> Transcript:
    return transcript_from_turns(
        [
            {
                "speaker": s["speaker"],
                "text": s["text_original"],
                "start": s["start"],
                "end": s["end"],
                "confidence": s["confidence"],
                "language": s.get("language"),
            }
            for s in record["segments"]
        ],
        call_id=record["call_id"],
        call_direction=record.get("call_direction", "unknown"),
    )


def evaluate_records(records: Sequence[dict], *, backend: str | None = None) -> list[CaseResult]:
    engine = get_backend(backend)
    results: list[CaseResult] = []
    for record in records:
        transcript = transcript_of(record)
        started = time.perf_counter()
        analysis = engine.analyze(transcript, realtime=False)
        risk = assess(analysis, transcript)
        latency = (time.perf_counter() - started) * 1000

        gold = record["gold"]
        gold_categories = {factor["category"] for factor in gold.get("risk_factors", [])}
        predicted_categories = {factor.category for factor in analysis.risk_factors}
        json_valid = not any(item.startswith("invalid_json") for item in analysis.dropped_findings)

        results.append(
            CaseResult(
                call_id=record["call_id"],
                slices=record.get("slices", []),
                gold_classification=gold["classification"],
                predicted_classification=analysis.classification.value,
                risk_score=risk.risk_score,
                risk_level=risk.risk_level.value,
                gold_is_scam=gold["classification"] == "SCAM",
                flagged_by_model=analysis.classification.value in {"SCAM", "SUSPICIOUS"},
                flagged_by_risk=risk.risk_score >= ALERT_WARNING_THRESHOLD,
                json_valid=json_valid,
                gold_scam_types=set(gold.get("scam_types", [])),
                predicted_scam_types=set(analysis.scam_types),
                gold_tactics=set(gold.get("tactics", [])),
                predicted_tactics=set(analysis.tactics),
                gold_categories=gold_categories,
                predicted_categories=predicted_categories,
                grounded_findings=len(analysis.grounded_factors()),
                total_findings=len(analysis.risk_factors) + len(analysis.dropped_findings),
                latency_ms=round(latency, 1),
            )
        )
    return results


def summarise(results: Sequence[CaseResult]) -> dict:
    if not results:
        return {}

    scams = [r for r in results if r.gold_is_scam]
    legit = [r for r in results if not r.gold_is_scam]

    strict_tp = sum(1 for r in scams if r.predicted_classification == "SCAM")
    strict_fp = sum(1 for r in legit if r.predicted_classification == "SCAM")
    strict_fn = len(scams) - strict_tp

    alert_tp = sum(1 for r in scams if r.flagged_by_model)
    alert_fp = sum(1 for r in legit if r.flagged_by_model)
    alert_fn = len(scams) - alert_tp

    risk_tp = sum(1 for r in scams if r.flagged_by_risk)
    risk_fp = sum(1 for r in legit if r.flagged_by_risk)
    risk_fn = len(scams) - risk_tp

    payload: dict = {
        "n": len(results),
        "n_scam": len(scams),
        "n_legitimate": len(legit),
        # --- the two metrics that matter most (spec section 26) ---
        "scam_recall_strict": round(strict_tp / len(scams), 4) if scams else 0.0,
        "false_positive_rate_strict": round(strict_fp / len(legit), 4) if legit else 0.0,
        "scam_recall_flagged": round(alert_tp / len(scams), 4) if scams else 0.0,
        "false_positive_rate_flagged": round(alert_fp / len(legit), 4) if legit else 0.0,
        "scam_recall_risk_engine": round(risk_tp / len(scams), 4) if scams else 0.0,
        "false_positive_rate_risk_engine": round(risk_fp / len(legit), 4) if legit else 0.0,
        # --- classification quality ---
        "classification_f1_strict": _prf(strict_tp, strict_fp, strict_fn),
        "classification_f1_flagged": _prf(alert_tp, alert_fp, alert_fn),
        "classification_f1_risk_engine": _prf(risk_tp, risk_fp, risk_fn),
        "exact_classification_accuracy": round(
            sum(1 for r in results if r.predicted_classification == r.gold_classification)
            / len(results),
            4,
        ),
        # --- structured-output quality ---
        "scam_type": _set_micro([(r.predicted_scam_types, r.gold_scam_types) for r in scams]),
        "tactic": _set_micro([(r.predicted_tactics, r.gold_tactics) for r in results]),
        "event_category": _set_micro([(r.predicted_categories, r.gold_categories) for r in results]),
        "json_validity": round(sum(1 for r in results if r.json_valid) / len(results), 4),
        "evidence_grounding_rate": round(
            sum(r.grounded_findings for r in results) / max(1, sum(r.total_findings for r in results)),
            4,
        ),
        "mean_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 1),
        "mean_risk_scam": round(sum(r.risk_score for r in scams) / len(scams), 1) if scams else 0.0,
        "mean_risk_legitimate": round(sum(r.risk_score for r in legit) / len(legit), 1) if legit else 0.0,
    }

    # Calibration sweep over the risk threshold.
    calibration = evaluate_thresholds(
        [ScoredCall(r.call_id, r.risk_score, r.gold_is_scam) for r in results]
    )
    best = calibration.best_threshold(max_false_positive_rate=0.05)
    payload["risk_band_distribution"] = calibration.band_distribution
    payload["best_threshold_at_5pct_fpr"] = best.as_dict() if best else None
    return payload


def summarise_by_slice(results: Sequence[CaseResult]) -> dict[str, dict]:
    buckets: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        for name in result.slices:
            buckets[name].append(result)
    return {
        name: {
            key: value
            for key, value in summarise(subset).items()
            if key
            in {
                "n", "n_scam", "n_legitimate",
                "scam_recall_strict", "false_positive_rate_strict",
                "scam_recall_risk_engine", "false_positive_rate_risk_engine",
                "exact_classification_accuracy", "tactic", "json_validity",
                "evidence_grounding_rate", "mean_risk_scam", "mean_risk_legitimate",
            }
        }
        for name, subset in sorted(buckets.items())
    }


def run(
    *, split: str = "test", backend: str | None = None, limit: int | None = None
) -> dict:
    paths = settings.paths
    directory = {"train": paths.train, "validation": paths.validation, "test": paths.test}[split]
    records = load_split(directory / "dataset.jsonl")
    if limit:
        records = records[:limit]
    results = evaluate_records(records, backend=backend)
    report = {
        "backend": get_backend(backend).name,
        "split": split,
        "overall": summarise(results),
        "by_slice": summarise_by_slice(results),
        "failures": [
            {
                "call_id": r.call_id,
                "gold": r.gold_classification,
                "predicted": r.predicted_classification,
                "risk_score": r.risk_score,
                "slices": r.slices,
            }
            for r in results
            if (r.gold_is_scam and not r.flagged_by_risk)
            or (not r.gold_is_scam and r.flagged_by_risk)
        ],
    }
    return report


def print_report(report: dict) -> None:
    overall = report["overall"]
    print(f"\nQORGAU evaluation — backend `{report['backend']}`, split `{report['split']}`")
    print("=" * 78)
    print(f"conversations: {overall['n']}  ({overall['n_scam']} scam / {overall['n_legitimate']} legitimate)")
    print("\nPriority metrics (spec section 26)")
    print(f"  scam recall     model-strict {overall['scam_recall_strict']:.3f} | "
          f"model-flagged {overall['scam_recall_flagged']:.3f} | "
          f"risk>=60 {overall['scam_recall_risk_engine']:.3f}")
    print(f"  false pos rate  model-strict {overall['false_positive_rate_strict']:.3f} | "
          f"model-flagged {overall['false_positive_rate_flagged']:.3f} | "
          f"risk>=60 {overall['false_positive_rate_risk_engine']:.3f}")
    print("\nOther metrics")
    print(f"  exact classification accuracy : {overall['exact_classification_accuracy']:.3f}")
    print(f"  scam-type micro F1            : {overall['scam_type']['f1']:.3f} "
          f"(exact set {overall['scam_type']['exact_set_match']:.3f})")
    print(f"  tactic micro F1               : {overall['tactic']['f1']:.3f}")
    print(f"  event-category micro F1       : {overall['event_category']['f1']:.3f}")
    print(f"  JSON validity                 : {overall['json_validity']:.3f}")
    print(f"  evidence grounding rate       : {overall['evidence_grounding_rate']:.3f}")
    print(f"  mean risk  scam / legitimate  : {overall['mean_risk_scam']:.1f} / {overall['mean_risk_legitimate']:.1f}")
    print(f"  mean latency                  : {overall['mean_latency_ms']:.1f} ms")

    print("\nPer slice")
    header = f"  {'slice':16} {'n':>4} {'recall':>7} {'FPR':>6} {'acc':>6} {'tacticF1':>9} {'ground':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, scores in report["by_slice"].items():
        print(
            f"  {name:16} {scores['n']:4} {scores['scam_recall_risk_engine']:7.3f} "
            f"{scores['false_positive_rate_risk_engine']:6.3f} "
            f"{scores['exact_classification_accuracy']:6.3f} {scores['tactic']['f1']:9.3f} "
            f"{scores['evidence_grounding_rate']:7.3f}"
        )
    if report["failures"]:
        print(f"\nMisses / false alarms ({len(report['failures'])})")
        for failure in report["failures"][:15]:
            print(f"  {failure['call_id']:52} gold={failure['gold']:10} "
                  f"pred={failure['predicted']:10} risk={failure['risk_score']}")
    else:
        print("\nNo misses or false alarms on this split.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a QORGAU analysis backend")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--backend", default=None, help="reference | local_adapter | anthropic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    args = parser.parse_args()

    report = run(split=args.split, backend=args.backend, limit=args.limit)
    print_report(report)

    out = Path(args.out) if args.out else (
        settings.paths.reports / f"eval_{report['backend']}_{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
