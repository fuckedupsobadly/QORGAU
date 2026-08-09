"""Build the fine-tuning dataset (spec sections 13-16, 24, 25).

    script families → rendered conversations → transcripts → gold JSON
      → family-level split → chat-formatted JSONL

Split policy (spec section 25): **conversations are never split; families are.**
Every language variant, ASR-noise variant and paraphrase of a script stays inside
one split, and two families are reserved entirely for the test set as an
"unseen pattern" slice. That is stricter than splitting by conversation and it is
the only way the reported numbers mean "generalises to a new scam script" rather
than "recognises a script it has seen".

Real annotated calls can be mixed in by dropping JSON files into `datasets/raw/`
(same shape as the fixtures this script writes, plus a `gold` object). Real data
always goes to the training split unless it declares `"split": "test"`, and never
displaces a synthetic family's assignment.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from config.settings import settings
from models.prompts import MASTER_SYSTEM_PROMPT, build_user_prompt
from training.corpus import (
    ALL_FAMILIES,
    Family,
    LEGIT_FAMILIES,
    RenderedCall,
    SCAM_FAMILIES,
    gold_analysis,
    iter_corpus,
)
from transcription.processor import transcript_from_turns
from transcription.schemas import Transcript

SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Family-level split
# ---------------------------------------------------------------------------


def split_families(
    *, seed: int = 20260809, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> dict[str, str]:
    """family key -> split. Held-out families always land in test."""
    rng = random.Random(seed)
    assignment: dict[str, str] = {}

    for group in (SCAM_FAMILIES, LEGIT_FAMILIES):
        holdout = [f for f in group if f.holdout]
        pool = [f for f in group if not f.holdout]
        rng.shuffle(pool)
        for family in holdout:
            assignment[family.key] = "test"

        total = len(pool)
        n_val = max(1, round(total * ratios[1]))
        n_test = max(1, round(total * ratios[2]))
        n_train = total - n_val - n_test
        if n_train < 1:  # tiny group: keep at least one training family
            n_train, n_val, n_test = max(1, total - 2), min(1, total - 1), max(0, total - 2)
        for index, family in enumerate(pool):
            if index < n_train:
                assignment[family.key] = "train"
            elif index < n_train + n_val:
                assignment[family.key] = "validation"
            else:
                assignment[family.key] = "test"
    return assignment


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def build_record(rendered: RenderedCall) -> dict:
    """One dataset row: transcript, gold analysis, and the chat-format example."""
    transcript: Transcript = transcript_from_turns(
        [
            {
                "speaker": turn["speaker"],
                "text": turn["text"],
                "start": turn["start"],
                "end": turn["end"],
                "confidence": turn["confidence"],
            }
            for turn in rendered.turns
        ],
        call_id=rendered.call_id,
        call_direction=rendered.family.call_direction,
    )
    # Evidence must be verbatim what the model reads, so it is taken from the
    # *normalized* segment text rather than from the script.
    texts = [segment.text for segment in transcript.segments]
    timestamps = [segment.timestamp for segment in transcript.segments]
    gold = gold_analysis(rendered, texts=texts, timestamps=timestamps)

    target = json.dumps(gold, ensure_ascii=False)
    return {
        "call_id": rendered.call_id,
        "family": rendered.family.key,
        "family_label": rendered.family.label,
        "language_mode": rendered.language_mode,
        "asr_noisy": rendered.noisy,
        "difficulty": rendered.family.difficulty,
        "slices": rendered.slices,
        "call_direction": rendered.family.call_direction,
        "dominant_language": transcript.dominant_language,
        "segments": [
            {
                "speaker": segment.speaker.value,
                "start": segment.start,
                "end": segment.end,
                "timestamp": segment.timestamp,
                "language": segment.language.value,
                "text": segment.text,
                "text_original": segment.text_original,
                "confidence": segment.confidence,
            }
            for segment in transcript.segments
        ],
        "gold": gold,
        "messages": [
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(transcript)},
            {"role": "assistant", "content": target},
        ],
    }


def load_real_calls(raw_dir: Path) -> list[dict]:
    """Pick up human-annotated real calls, if any have been added."""
    records: list[dict] = []
    if not raw_dir.exists():
        return records
    for path in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ! skipping unparseable {path.name}")
            continue
        if "gold" not in payload or "segments" not in payload:
            print(f"  ! skipping {path.name}: needs both 'segments' and 'gold'")
            continue
        transcript = transcript_from_turns(
            payload["segments"],
            call_id=payload.get("call_id", path.stem),
            call_direction=payload.get("call_direction", "unknown"),
        )
        records.append(
            {
                "call_id": payload.get("call_id", path.stem),
                "family": payload.get("family", f"real::{path.stem}"),
                "family_label": payload.get("family_label", "real annotated call"),
                "language_mode": payload.get("language_mode", transcript.dominant_language),
                "asr_noisy": payload.get("asr_noisy", transcript.mean_confidence < 0.8),
                "difficulty": payload.get("difficulty", "real"),
                "slices": payload.get("slices", ["real"]),
                "call_direction": transcript.call_direction,
                "dominant_language": transcript.dominant_language,
                "segments": [
                    {
                        "speaker": s.speaker.value,
                        "start": s.start,
                        "end": s.end,
                        "timestamp": s.timestamp,
                        "language": s.language.value,
                        "text": s.text,
                        "text_original": s.text_original,
                        "confidence": s.confidence,
                    }
                    for s in transcript.segments
                ],
                "gold": payload["gold"],
                "messages": [
                    {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(transcript)},
                    {
                        "role": "assistant",
                        "content": json.dumps(payload["gold"], ensure_ascii=False),
                    },
                ],
                "_split": payload.get("split", "train"),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_fixtures(records: Sequence[dict], out_dir: Path) -> int:
    """Fixture files the demo UI and `FixtureSource` can replay directly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        payload = {
            "call_id": record["call_id"],
            "call_direction": record["call_direction"],
            "family": record["family"],
            "family_label": record["family_label"],
            "language_mode": record["language_mode"],
            "asr_noisy": record["asr_noisy"],
            "expected_classification": record["gold"]["classification"],
            "segments": [
                {
                    "speaker": s["speaker"],
                    "start": s["start"],
                    "end": s["end"],
                    "language": s["language"],
                    "text": s["text_original"],
                    "confidence": s["confidence"],
                }
                for s in record["segments"]
            ],
        }
        (out_dir / f"{record['call_id']}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return len(records)


def summarise(records_by_split: dict[str, list[dict]], assignment: dict[str, str]) -> dict:
    manifest: dict = {
        "family_split": assignment,
        "counts": {},
        "slices": {},
        "families": {},
        "leakage_check": {},
    }
    for split, records in records_by_split.items():
        manifest["counts"][split] = len(records)
        slice_counts: Counter[str] = Counter()
        for record in records:
            slice_counts.update(record["slices"])
        manifest["slices"][split] = dict(sorted(slice_counts.items()))
        manifest["families"][split] = sorted({record["family"] for record in records})

    train_families = set(manifest["families"].get("train", []))
    for split in ("validation", "test"):
        overlap = train_families & set(manifest["families"].get(split, []))
        manifest["leakage_check"][f"train_vs_{split}_family_overlap"] = sorted(overlap)
    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def prepare(
    *,
    seed: int = 20260809,
    variants: int = 2,
    include_noisy: bool = True,
    out_root: Path | None = None,
) -> dict:
    paths = settings.paths
    root = out_root or paths.datasets
    root.mkdir(parents=True, exist_ok=True)

    assignment = split_families(seed=seed)
    records_by_split: dict[str, list[dict]] = defaultdict(list)
    all_records: list[dict] = []

    for rendered in iter_corpus(seed=seed, variants=variants, include_noisy=include_noisy):
        record = build_record(rendered)
        split = assignment[rendered.family.key]
        record["split"] = split
        records_by_split[split].append(record)
        all_records.append(record)

    real = load_real_calls(paths.raw)
    for record in real:
        split = record.pop("_split", "train")
        record["split"] = split if split in SPLITS else "train"
        records_by_split[record["split"]].append(record)
        all_records.append(record)
    if real:
        print(f"  + mixed in {len(real)} real annotated call(s) from {paths.raw}")

    write_jsonl(paths.processed / "corpus.jsonl", all_records)
    for split in SPLITS:
        rows = records_by_split.get(split, [])
        target_dir = {"train": paths.train, "validation": paths.validation, "test": paths.test}[split]
        write_jsonl(target_dir / "dataset.jsonl", rows)
        # Slice files make per-slice evaluation (spec section 25) a one-liner.
        if split == "test":
            by_slice: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                for name in row["slices"]:
                    by_slice[name].append(row)
            for name, subset in by_slice.items():
                write_jsonl(target_dir / f"slice_{name}.jsonl", subset)

    write_fixtures(all_records, paths.processed / "fixtures")
    manifest = summarise(records_by_split, assignment)
    (paths.processed / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the QORGAU fine-tuning dataset")
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--variants", type=int, default=2, help="paraphrase variants per family+language")
    parser.add_argument("--no-noisy", action="store_true", help="skip ASR-noise variants")
    args = parser.parse_args()

    print("Building QORGAU corpus...")
    manifest = prepare(seed=args.seed, variants=args.variants, include_noisy=not args.no_noisy)
    print(f"  families: {len(ALL_FAMILIES)} ({len(SCAM_FAMILIES)} scam / {len(LEGIT_FAMILIES)} legitimate)")
    for split in SPLITS:
        print(f"  {split:11} {manifest['counts'].get(split, 0):4} conversations, "
              f"{len(manifest['families'].get(split, []))} families")
    for key, overlap in manifest["leakage_check"].items():
        status = "clean" if not overlap else f"LEAK: {overlap}"
        print(f"  {key}: {status}")
    print(f"  test slices: {manifest['slices'].get('test', {})}")
    print(f"  written to {settings.paths.datasets}")


if __name__ == "__main__":
    main()
