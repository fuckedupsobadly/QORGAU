# QORGAU — implementation notes

How the system described in [README.md](README.md) is actually built, how to run
it, and what it measures. The README is the specification; this is the build.

QORGAU analyses telephone conversations in **Kazakh, Russian and mixed
Kazakh-Russian speech** and detects financial scams and social-engineering
attacks — while the call is still happening.

It is built around one idea: **a scam is a behavioural sequence, not a set of
keywords.**

```
IDENTITY CLAIM → PROBLEM CLAIM → EMOTIONAL MANIPULATION
              → PROPOSED SOLUTION → REQUESTED ACTION
              → FINANCIAL / CREDENTIAL CONSEQUENCE
```

"OTP", "bank", "police" and "money" appear in almost every *legitimate* bank call
too. What distinguishes an attack is **who** said something, **why**, and **what
they are trying to make the other person do**. So:

| | |
|---|---|
| `Никому не сообщайте код из СМС` (bank employee) | protective advice — **0 risk** |
| `Продиктуйте мне код из СМС` (unknown caller, after inventing a loan) | OTP theft — **critical** |

Same vocabulary. Opposite meaning. QORGAU is designed around telling them apart.

---

## Quick start

```bash
pip install -r requirements.txt
python -m training.prepare_dataset          # build the annotated corpus
python -m training.evaluate --split test    # held-out metrics
streamlit run app/frontend/streamlit_app.py # the dashboard
```

Or just `./run.sh`. Nothing above needs a GPU, an API key, or an ML dependency.

The dashboard has three modes:

* **Analyse a call** — a corpus call, an uploaded recording, or a pasted
  transcript; verdict, evidence timeline, and the full risk breakdown. Clicking a
  timeline event jumps to that moment in the transcript.
* **Live call (real-time)** — replays a call turn by turn so you can watch the
  risk score, conversation stage and alerts move as the attack develops.
* **Model evaluation** — the held-out numbers, per slice.

---

## Measured results

Full detail in [`METRICS.md`](METRICS.md). Held-out **test** split, 72
conversations (48 scam / 24 legitimate):

| metric | value |
|---|---|
| **scam recall** | **1.000** |
| **false-positive rate** | **0.000** |
| exact classification accuracy | 0.944 |
| tactic micro F1 | 0.741 |
| evidence grounding rate | 1.000 |
| latency per call | ~4 ms |

Per language slice: Kazakh 1.000 recall, Russian 1.000, mixed 1.000, ASR-noisy
1.000 — and 0.000 false positives on every legitimate slice.

**Read these numbers with the following caveats, which matter:**

1. **The corpus is synthetic.** It is generated from 24 hand-annotated script
   families (14 scam / 10 legitimate). Splits are by *family*, so no test
   conversation paraphrases a training one, and two scam families plus one
   legitimate family are withheld entirely as an `unseen_pattern` slice. That is
   a real generalisation test, but it is still a narrower distribution than real
   calls. Expect these numbers to drop on production audio.
2. **These are the numbers for the `reference` backend, not a trained adapter.**
   Training the LLM needs a GPU (see below). The reference backend is a
   transparent, inspectable implementation of the same sequence model, and it is
   there so the whole system runs and is measurable without weights — and so the
   fine-tuned model has a real baseline to beat.
3. **The score is a system risk estimate, not a calibrated probability.** There
   is no `97.384829%` anywhere in this codebase, by design.

---

## Architecture

Each stage is an injected component behind an interface, so any one of them can
be replaced without touching the others.

```
                  TELEPHONE CALL
                        │
   audio/ingestion.py   │  WebRTC / SIP frames, uploaded file, or fixture
                        ▼
   audio/vad.py         │  energy VAD (no deps) │ Silero
                        ▼
   audio/diarization.py │  CALLER / VICTIM / UNKNOWN  ← load-bearing, see below
                        ▼
   audio/asr.py         │  faster-whisper │ whisper │ fixture
                        ▼
   transcription/normalizer.py   two transcripts: verbatim + normalized
                        ▼
   models/fraud_llm/    ★ FINE-TUNED FRAUD LLM  → JSON findings + evidence
                        ▼
   risk/engine.py       ★ DETERMINISTIC RISK ENGINE → 0-100 + audit trail
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  realtime/alerts.py            risk/engine.py::render_report
  live warning                  investigation report
```

**The LLM understands the conversation. The risk engine decides the number.**
The model never computes a score; the engine never calls a model. That split is
what makes the output both context-aware *and* auditable — every point in a score
traces to a named rule in [`risk/rules.py`](risk/rules.py).

### Why diarization is not optional

`Не говорите никому свой SMS-код` and `Назовите мне ваш SMS-код` share most of
their vocabulary and mean opposite things. Without a speaker label the analysis
layer cannot tell a warning from an attack, so the risk engine **refuses to score
a harmful request attributed to anyone but the caller**
([`risk/engine.py`](risk/engine.py), `CALLER_ONLY_CATEGORIES`).

### Evidence can't be invented

Every finding must quote the transcript. `FraudLLMBackend.ground()` locates each
quote in the actual segments; anything it cannot find is **removed from the
analysis and recorded in `dropped_findings`**, and contributes nothing to the
score. A hallucinated `SCAM` verdict with no grounded evidence scores **0**.

### Code-switching is normal, never a signal

`Сіздің картаңыз заблокирована` is ordinary Kazakhstani speech. Language mixing is
detected only so the UI and the evaluator can slice by it. There is a test for
this ([`test_code_switching_is_not_a_risk_signal`](tests/test_qorgau.py)).

---

## The risk engine

Weights come from the spec; the combination rule is the interesting part.

```
score = 100 × (1 − Π(1 − cᵢ/100))     over capped group contributions
```

* **Group caps** — ten urgency mentions are not ten times one urgency mention.
* **Interaction effects** — `IMPERSONATION + OTP_REQUEST` is worth more than
  either alone, because the authority claim is what makes the request credible.
* **Mitigations** — protective advice, or a call the customer placed themselves,
  subtract.
* **Policy floors** — a CRITICAL credential/money/device request inside an
  impersonation frame is CRITICAL *regardless* of how the arithmetic lands.
* **Bands** — 0-29 SAFE · 30-59 SUSPICIOUS · 60-79 HIGH RISK · 80-100 CRITICAL.

Model and engine can disagree, and when they do QORGAU **says so** rather than
silently picking one (`RiskAssessment.disagreement`).

---

## Real-time mode

Analysis runs on every transcript chunk, over a context window of previous turns
— never on the newest sentence alone. Two properties:

* **Peak-hold.** A caller who asks for an OTP at 00:42 and then chats pleasantly
  is still dangerous at 01:30.
* **Immediate escalation.** A critical event alerts *at once*, without waiting for
  the total to reach 80. Waiting while the victim reads a code aloud defeats the
  purpose.

```json
{"type": "risk_update", "risk_score": 82, "classification": "CRITICAL",
 "current_stage": "CREDENTIAL_EXTRACTION",
 "event": {"category": "OTP_REQUEST", "severity": "CRITICAL"}}
```

---

## Fine-tuning the model

```bash
pip install -r requirements-ml.txt
python -m training.prepare_dataset                        # → datasets/{train,validation,test}
python -m training.train --base-model Qwen/Qwen2.5-7B-Instruct
python -m training.evaluate --split test --backend local_adapter
python -m training.export --mode card
QORGAU_LLM_BACKEND=local_adapter streamlit run app/frontend/streamlit_app.py
```

QLoRA (4-bit base + rank-32 adapters), because the task is teaching a *behaviour*
rather than new language competence — that fits on one consumer GPU and keeps the
base model swappable per deployment.

Two details that matter for this task:

* **Loss is computed on the JSON answer only.** The system prompt is ~2 kB of
  fixed instructions; training the model to predict it wastes capacity.
* **No sequence packing.** A transcript boundary is semantically load-bearing
  here, so conversations are never allowed to bleed into each other.

Training reports **JSON validity and classification agreement** from greedy
generation on validation calls, not just loss — the deliverable is parseable,
correct JSON.

### The corpus

`training/corpus.py` holds the annotated script families. The annotation lives on
the *script*, and rendering produces Russian / Kazakh / code-switched surface
forms, optionally degraded with realistic ASR noise (dropped vowels, stutters,
fillers, lost punctuation). Gold evidence is taken from the **rendered** text, so
it is always verbatim.

Adding real annotated calls: drop JSON files into `datasets/raw/` with `segments`
and a `gold` object; `prepare_dataset.py` mixes them in.

---

## Privacy and security

Call transcripts are sensitive financial data, and the code treats them that way.

* **Masking before storage** — OTPs, PINs, CVVs and card numbers are replaced in
  the queryable column. Money amounts are *not* masked; they are evidence.
* **Encryption at rest** — verbatim text is Fernet-encrypted under
  `QORGAU_ENCRYPTION_KEY`. **With no key configured, QORGAU declines to store the
  verbatim text at all** rather than silently writing plaintext or pretending to
  encrypt it.
* **No public recording URLs** — recordings are internal references only; there
  is no column that could leak one.
* **Audit log** — every save, read, decrypt, delete and purge is recorded.
* **Retention + deletion** — per-call deadlines enforced by `purge_expired()`;
  `DELETE /api/calls/{id}` erases a call.

```bash
export QORGAU_ENCRYPTION_KEY=$(python -c \
  'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')
```

---

## API

```
POST   /api/calls                 register a transcribed call
POST   /api/calls/upload          upload a recording (runs VAD→diarization→ASR)
POST   /api/calls/{id}/analyze    analyse a completed call
GET    /api/calls/{id}            transcript + analysis
GET    /api/calls/{id}/events     suspicious events
GET    /api/calls/{id}/report     investigation report (markdown)
DELETE /api/calls/{id}            erase a call
WS     /api/calls/{id}/stream     live risk_update frames
```

```bash
uvicorn app.api.main:app --reload
```

---

## Tests

```bash
python tests/test_qorgau.py     # or: pytest tests/
```

28 behavioural tests, all passing. They target the properties that make the
system trustworthy rather than implementation details — every legitimate family
staying below the alarm threshold, every scam family reaching it, victim speech
never adding risk, normalization never deleting evidence, ungrounded findings
being discarded, splits sharing no script family.

---

## Layout

```
qorgau/
├── app/            api/ · websocket/ · frontend/ (Streamlit)
├── audio/          ingestion · vad · diarization · asr
├── transcription/  schemas · normalizer · processor
├── models/         prompts · inference · fraud_llm/{lexicon, backends}
├── risk/           engine · rules · calibration
├── realtime/       session · alerts
├── database/       models · repository
├── training/       corpus · prepare_dataset · train · evaluate · export
├── datasets/        raw · processed · train · validation · test
├── config/         settings · ontology
└── tests/
```

## Configuration

| variable | default | purpose |
|---|---|---|
| `QORGAU_LLM_BACKEND` | `reference` | `reference` · `local_adapter` · `anthropic` |
| `QORGAU_BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | base for fine-tuning |
| `QORGAU_ADAPTER` | `artifacts/adapters/qorgau-lora` | trained adapter |
| `QORGAU_ASR` | `auto` | `faster_whisper` · `whisper` · `fixture` |
| `QORGAU_VAD` / `QORGAU_DIAR` | `energy` / `heuristic` | `silero` / `pyannote` |
| `QORGAU_ENCRYPTION_KEY` | — | Fernet key for transcripts at rest |
| `QORGAU_DATABASE_URL` | SQLite in `artifacts/storage` | any SQLAlchemy URL |

---

## Known limitations

Stated plainly, because they determine what this is ready for:

* **The corpus is synthetic.** Real calls are messier, longer, more interrupted,
  and contain scam scripts nobody has written down yet.
* **Kazakh ASR is the weakest link end-to-end**, ahead of the analysis layer.
  Whisper's Kazakh is materially worse than its Russian, and it emits a single
  language tag per segment — which is wrong for code-switched speech, so QORGAU
  re-derives language per utterance itself.
* **The reference backend is lexicon-bound.** It generalises across inflection,
  agglutination and noise, but a paraphrase built from vocabulary outside
  `lexicon.py` will be missed. That is exactly the gap the fine-tuned model
  closes, and why `unseen_pattern` is a reported slice.
* **The heuristic diarizer struggles with cross-talk.** It reports per-turn
  confidence so uncertain turns are visible; use `pyannote` for production.
* **The risk weights are policy, not learned.** `risk/calibration.py` sweeps them
  against labelled data, but a human sets them.

## What the system will not do

It analyses and warns. It does not block accounts, move money, contact
authorities, or decide anything financial — and it says "SUSPICIOUS" instead of
"SCAM" when the evidence is thin, because a fraud detector that cries wolf gets
switched off in a week.
