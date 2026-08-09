"""QLoRA fine-tuning loop for the QORGAU fraud LLM (spec section 24).

    datasets/train/dataset.jsonl → 4-bit base + LoRA adapters → artifacts/adapters/

Why LoRA/QLoRA rather than full fine-tuning: the task is to teach an existing
multilingual model a *behaviour* (read a call, emit this JSON) rather than new
language competence. Adapters get there on one consumer GPU, keep the base model
swappable, and are small enough to ship and roll back per bank.

Design choices that matter for this task:

* **Loss on the answer only.** The system prompt is ~2 kB of fixed instructions
  and the user turn is the transcript; training the model to predict those wastes
  capacity and biases it toward reciting the prompt. Prompt tokens are masked to
  -100 so gradient flows only through the JSON.
* **No packing.** Conversations must not bleed into each other — a transcript
  boundary is semantically load-bearing here.
* **Greedy eval during training.** The metric that matters is whether the emitted
  JSON is valid and correct, not perplexity, so `--eval-generate` samples a few
  validation calls and reports JSON validity and classification agreement.

Requires `requirements-ml.txt` (transformers, peft, trl, bitsandbytes, accelerate).
Nothing else in QORGAU imports this module, so the app runs without them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from config.settings import settings

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    base_model: str = settings.model.base_model
    output_dir: str = str(settings.paths.adapters / "qorgau-lora")
    train_file: Path = settings.paths.train / "dataset.jsonl"
    eval_file: Path = settings.paths.validation / "dataset.jsonl"

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    #: All attention + MLP projections. Attention-only adapters underfit the
    #: JSON-structure part of this task in practice.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )

    # optimisation
    epochs: float = 3.0
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 4096
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    seed: int = settings.random_seed

    # quantisation
    load_in_4bit: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True

    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 100
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m training.prepare_dataset` first"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SupervisedJSONDataset:
    """Chat-formatted examples with the prompt masked out of the loss."""

    def __init__(self, records: Sequence[dict], tokenizer: Any, max_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: list[dict] = []
        skipped = 0
        for record in records:
            messages = record["messages"]
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            full_text = prompt_text + messages[-1]["content"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if len(full_ids) > max_len:
                # Truncating the answer would teach the model to emit invalid JSON.
                skipped += 1
                continue
            labels = list(full_ids)
            for index in range(min(len(prompt_ids), len(labels))):
                labels[index] = -100
            self.examples.append(
                {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}
            )
        if skipped:
            print(f"  ! skipped {skipped} example(s) longer than max_seq_length={max_len}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def collate(batch: Sequence[dict], pad_token_id: int) -> dict:
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, mask = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        mask.append(item["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(config: TrainConfig):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quant_config,
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
    )
    model.config.use_cache = False
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation check — the metric that actually matters
# ---------------------------------------------------------------------------


def generation_check(model, tokenizer, records: Sequence[dict], *, limit: int = 8) -> dict:
    """Sample a few validation calls: is the JSON valid, is the verdict right?"""
    import torch

    from transcription.schemas import LLMAnalysis, extract_json

    model.eval()
    valid = agree = 0
    total = min(limit, len(records))
    for record in records[:total]:
        prompt = tokenizer.apply_chat_template(
            record["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=settings.model.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        try:
            analysis = LLMAnalysis.model_validate(extract_json(completion))
        except Exception:
            continue
        valid += 1
        if analysis.classification.value == record["gold"]["classification"]:
            agree += 1
    model.train()
    return {
        "sampled": total,
        "json_validity": round(valid / total, 3) if total else 0.0,
        "classification_agreement": round(agree / total, 3) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: TrainConfig, *, eval_generate: bool = True) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from transformers import get_scheduler, set_seed

    set_seed(config.seed)
    print(f"base model      : {config.base_model}")
    print(f"train file      : {config.train_file}")
    train_records = load_jsonl(config.train_file)
    eval_records = load_jsonl(config.eval_file) if config.eval_file.exists() else []

    model, tokenizer = build_model(config)
    train_ds = SupervisedJSONDataset(train_records, tokenizer, config.max_seq_length)
    eval_ds = SupervisedJSONDataset(eval_records, tokenizer, config.max_seq_length) if eval_records else None
    print(f"train examples  : {len(train_ds)}   eval examples: {len(eval_ds) if eval_ds else 0}")

    loader = DataLoader(
        train_ds,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    eval_loader = (
        DataLoader(
            eval_ds,
            batch_size=config.per_device_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
        )
        if eval_ds
        else None
    )

    steps_per_epoch = max(1, len(loader) // config.gradient_accumulation_steps)
    total_steps = int(steps_per_epoch * config.epochs)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    step = 0
    running = 0.0
    model.train()

    print(f"optimiser steps : {total_steps} ({steps_per_epoch}/epoch)")
    for epoch in range(math.ceil(config.epochs)):
        for index, batch in enumerate(loader, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loss = model(**batch).loss / config.gradient_accumulation_steps
            loss.backward()
            running += loss.item()

            if index % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % config.logging_steps == 0:
                    entry = {
                        "step": step,
                        "epoch": round(step / steps_per_epoch, 3),
                        "loss": round(running / config.logging_steps, 4),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    history.append(entry)
                    print(f"  step {step:5d}/{total_steps}  loss {entry['loss']:.4f}  lr {entry['lr']:.2e}")
                    running = 0.0

                if eval_loader and step % config.eval_steps == 0:
                    model.eval()
                    losses = []
                    with torch.no_grad():
                        for eval_batch in eval_loader:
                            eval_batch = {k: v.to(model.device) for k, v in eval_batch.items()}
                            losses.append(model(**eval_batch).loss.item())
                    model.train()
                    eval_loss = sum(losses) / max(1, len(losses))
                    print(f"  step {step:5d} eval_loss {eval_loss:.4f}")
                    history.append({"step": step, "eval_loss": round(eval_loss, 4)})

                if step % config.save_steps == 0:
                    model.save_pretrained(str(output_dir))
                    tokenizer.save_pretrained(str(output_dir))

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics: dict[str, Any] = {"history": history, "config": {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(config).items()
    }}
    if eval_generate and eval_records:
        print("  running generation check on validation calls...")
        metrics["generation_check"] = generation_check(model, tokenizer, eval_records)
        print(f"  {metrics['generation_check']}")

    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nadapter saved to {output_dir}")
    print("serve it with:  QORGAU_LLM_BACKEND=local_adapter streamlit run app/frontend/streamlit_app.py")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune the QORGAU fraud LLM")
    parser.add_argument("--base-model", default=TrainConfig.base_model)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--epochs", type=float, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--lora-r", type=int, default=TrainConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.per_device_batch_size)
    parser.add_argument("--grad-accum", type=int, default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument("--max-seq-length", type=int, default=TrainConfig.max_seq_length)
    parser.add_argument("--no-4bit", action="store_true", help="load in bf16 instead of 4-bit")
    parser.add_argument("--no-eval-generate", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
    )
    try:
        train(config, eval_generate=not args.no_eval_generate)
    except ImportError as exc:
        raise SystemExit(
            f"missing ML dependency ({exc}).\n"
            "install them with:  pip install -r requirements-ml.txt\n"
            "(the app itself runs without them, on the reference backend)"
        ) from exc


if __name__ == "__main__":
    main()
