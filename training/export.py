"""Export a trained adapter for serving (spec section 24, last two stages).

Three modes:

* `adapter`  — validate + package the LoRA adapter (small, base model stays shared)
* `merged`   — merge LoRA into the base weights (one self-contained model; needed
               by inference servers that do not support adapters)
* `card`     — write a model card recording what was trained, on what, and how it
               scored, so a deployed adapter is traceable to its data and metrics
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings

REQUIRED_ADAPTER_FILES = ("adapter_config.json",)


def validate_adapter(adapter_dir: Path) -> dict:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"{adapter_dir} does not exist — train first")
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{adapter_dir} is not a PEFT adapter (missing {missing})")
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    weights = [p.name for p in adapter_dir.iterdir() if p.suffix in {".safetensors", ".bin"}]
    size_mb = sum((adapter_dir / name).stat().st_size for name in weights) / 1e6
    return {
        "base_model": config.get("base_model_name_or_path"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": config.get("target_modules"),
        "weight_files": weights,
        "size_mb": round(size_mb, 2),
    }


def merge_adapter(adapter_dir: Path, out_dir: Path, base_model: str | None = None) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info = validate_adapter(adapter_dir)
    base = base_model or info["base_model"] or settings.model.base_model
    print(f"merging {adapter_dir.name} into {base} (this loads the full model in fp16/bf16)")

    model = AutoModelForCausalLM.from_pretrained(base, dtype="auto", device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(str(out_dir))
    return out_dir


def write_model_card(adapter_dir: Path, out_path: Path | None = None) -> Path:
    """Record what this adapter is, so a deployed model is auditable."""
    info = validate_adapter(adapter_dir)
    manifest_path = settings.paths.processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    metrics_path = adapter_dir / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    eval_reports = {}
    for report in sorted(settings.paths.reports.glob("eval_*.json")):
        payload = json.loads(report.read_text(encoding="utf-8"))
        eval_reports[report.stem] = {
            "backend": payload.get("backend"),
            "split": payload.get("split"),
            "scam_recall_risk_engine": payload.get("overall", {}).get("scam_recall_risk_engine"),
            "false_positive_rate_risk_engine": payload.get("overall", {}).get(
                "false_positive_rate_risk_engine"
            ),
            "json_validity": payload.get("overall", {}).get("json_validity"),
        }

    lines = [
        "# QORGAU fraud LLM — adapter card",
        "",
        f"Exported {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Adapter",
        "",
        f"- base model: `{info['base_model']}`",
        f"- LoRA rank / alpha: {info['r']} / {info['lora_alpha']}",
        f"- target modules: {', '.join(info['target_modules'] or [])}",
        f"- adapter size: {info['size_mb']} MB",
        "",
        "## Task",
        "",
        "Input: a diarized, normalized Kazakh/Russian/code-switched call transcript.",
        "Output: the QORGAU analysis JSON (spec section 17) — classification, scam types,",
        "tactics, conversation stage, requested actions, and evidence-bearing risk factors.",
        "The model does **not** compute the risk score; `risk/engine.py` does, deterministically.",
        "",
        "## Training data",
        "",
        f"- families: {len(manifest.get('family_split', {}))}",
        f"- conversations per split: {manifest.get('counts', {})}",
        f"- split policy: by script family, so no test conversation paraphrases a training one",
        f"- family overlap check: {manifest.get('leakage_check', {})}",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(eval_reports, indent=2),
        "```",
        "",
        "## Known limitations",
        "",
        "- The bundled corpus is synthetic; real call distributions are wider and noisier.",
        "- Kazakh ASR quality is the dominant error source end-to-end, ahead of this model.",
        "- Scores are system risk estimates, not calibrated fraud probabilities.",
        "",
        "## Intended use",
        "",
        "Decision support for anti-fraud analysts and live caller warnings. Not an automated",
        "authority to block accounts, move money, or contact law enforcement.",
    ]
    if metrics.get("generation_check"):
        lines += ["", "## Generation check at end of training", "",
                  f"```json\n{json.dumps(metrics['generation_check'], indent=2)}\n```"]

    target = out_path or (adapter_dir / "MODEL_CARD.md")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained QORGAU adapter")
    parser.add_argument("--adapter", default=str(settings.paths.adapters / "qorgau-lora"))
    parser.add_argument("--mode", choices=("adapter", "merged", "card"), default="adapter")
    parser.add_argument("--out", default=None)
    parser.add_argument("--base-model", default=None)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    try:
        validate_adapter(adapter_dir)
    except FileNotFoundError as exc:
        # A missing adapter is the normal state before training, not a crash.
        raise SystemExit(
            f"{exc}\n\ntrain one first:\n"
            "  pip install -r requirements-ml.txt\n"
            "  python -m training.prepare_dataset\n"
            "  python -m training.train"
        ) from None

    if args.mode == "card":
        print(f"model card written to {write_model_card(adapter_dir, Path(args.out) if args.out else None)}")
        return
    if args.mode == "merged":
        out = Path(args.out or (settings.paths.adapters / "qorgau-merged"))
        print(f"merged model written to {merge_adapter(adapter_dir, out, args.base_model)}")
        write_model_card(adapter_dir)
        return

    info = validate_adapter(adapter_dir)
    print(json.dumps(info, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        archive = shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=adapter_dir)
        print(f"packaged {archive}")
    write_model_card(adapter_dir)
    print("adapter validated; serve with QORGAU_LLM_BACKEND=local_adapter")


if __name__ == "__main__":
    main()
