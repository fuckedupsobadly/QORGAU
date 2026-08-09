"""The production backend: base model + QORGAU QLoRA adapter.

This is the component the whole architecture is built around. It loads the
fine-tuned adapter produced by `training/train.py` and runs greedy decoding with
the exact `MASTER_SYSTEM_PROMPT` used during training, so there is no
training/serving skew.

Requires `transformers`, `peft`, `torch` and (for 4-bit) `bitsandbytes`. Those are
not installed by default — see `requirements-ml.txt`. Until they are, the app
runs on the `reference` backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config.settings import settings
from models.fraud_llm.backends.base import FraudLLMBackend
from models.prompts import MASTER_SYSTEM_PROMPT, build_user_prompt
from transcription.schemas import LLMAnalysis, Transcript, extract_json

logger = logging.getLogger(__name__)


class LocalAdapterBackend(FraudLLMBackend):
    """Fine-tuned Kazakh/Russian fraud LLM served locally."""

    name = "local_adapter"

    def __init__(
        self,
        base_model: str | None = None,
        adapter_path: str | None = None,
        *,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:
        self.base_model = base_model or settings.model.base_model
        self.adapter_path = adapter_path or settings.model.adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError(
                "LocalAdapterBackend needs `transformers` and `torch`: "
                "pip install -r requirements-ml.txt"
            ) from exc

        quant_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except Exception:  # pragma: no cover - optional
                logger.warning("bitsandbytes unavailable; loading in bf16 instead")

        logger.info("loading base model %s", self.base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map=self.device_map,
            dtype=torch.bfloat16,
            quantization_config=quant_config,
        )

        adapter_dir = Path(self.adapter_path)
        if adapter_dir.exists():
            from peft import PeftModel

            logger.info("attaching QORGAU adapter %s", adapter_dir)
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model = model.eval()
        else:
            logger.warning(
                "adapter %s not found — running the BASE model. Findings will be far "
                "weaker than the fine-tuned system; train with training/train.py.",
                adapter_dir,
            )
        self._model = model

    # ------------------------------------------------------------------
    def analyze(self, transcript: Transcript, *, realtime: bool = False) -> LLMAnalysis:
        self.warmup()
        import torch

        messages = [
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(transcript, realtime=realtime)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=settings.model.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        completion = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        analysis = self._parse(completion)
        return self.ground(analysis, transcript)

    def _parse(self, completion: str) -> LLMAnalysis:
        try:
            payload = extract_json(completion)
        except ValueError as exc:
            # A JSON-validity failure is a real model defect: surface it rather
            # than inventing an analysis.
            logger.error("fine-tuned model emitted invalid JSON: %s", exc)
            return LLMAnalysis(
                explanation=f"Model output was not valid JSON ({exc}). No findings can be trusted.",
                recommended_action="Re-run the analysis; if this repeats, the adapter needs retraining.",
                dropped_findings=[f"invalid_json: {completion[:200]}"],
            )
        return LLMAnalysis.model_validate(payload)
