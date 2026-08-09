# QORGAU

### AI fraud intelligence for telephone calls in Kazakh, Russian and mixed speech

> **қорғау** — *to protect*

QORGAU listens to a phone call and works out whether someone is being scammed —
while the call is still happening.

It is built on one idea: **a scam is a behavioural sequence, not a set of
keywords.**

```
IDENTITY CLAIM  →  PROBLEM CLAIM  →  EMOTIONAL MANIPULATION
                →  PROPOSED SOLUTION  →  REQUESTED ACTION
                →  FINANCIAL / CREDENTIAL CONSEQUENCE
```

`банк`, `код`, `деньги`, `полиция` appear in almost every *legitimate* bank call
too. What separates an attack from customer service is **who** said something,
**why**, and **what they are trying to make the other person do**:

| utterance | speaker | verdict |
|---|---|---|
| `Никому не сообщайте код из СМС` | bank employee | protective advice — **risk 0** |
| `Продиктуйте мне код из СМС` | unknown caller, after inventing a loan | OTP theft — **risk 80, CRITICAL** |

Nearly the same words. Opposite meaning. Telling those two apart is the whole
job, and it is what QORGAU is designed and measured around.

---

## Run it

```bash
pip install -r requirements.txt
python -m training.prepare_dataset            # build the annotated corpus
python -m training.evaluate --split test      # held-out metrics
streamlit run app/frontend/streamlit_app.py   # dashboard on :8501
```

Or `./run.sh`, which does all four. **No GPU, no API key and no ML dependency is
required** for any of that.

The dashboard has three modes:

| mode | what it does |
|---|---|
| **Analyse a call** | A corpus call, an uploaded recording, or a pasted transcript. Verdict, evidence timeline, and the full risk breakdown. Clicking a timeline event jumps to that moment in the transcript. |
| **Live call (real-time)** | Replays a call turn by turn so you can watch the risk score, conversation stage and alerts move as the attack develops. |
| **Model evaluation** | The held-out numbers, per language and per difficulty slice. |

REST + WebSocket API: `uvicorn app.api.main:app --reload`.

---

## Results

Held-out **test** split — 72 conversations, 48 scam / 24 legitimate. Full detail
and reproduction steps in [`METRICS.md`](METRICS.md).

| metric | value |
|---|---|
| **scam recall** | **1.000** |
| **false-positive rate** | **0.000** |
| exact classification accuracy | 0.944 |
| tactic micro F1 | 0.741 |
| evidence grounding rate | 1.000 |
| JSON validity | 1.000 |
| latency per call | ~4 ms |

Per slice: Kazakh 1.000 recall · Russian 1.000 · code-switched 1.000 ·
ASR-noisy 1.000 · unseen scam patterns 1.000 — with **0.000 false positives on
every legitimate slice**. No legitimate call reaches even SUSPICIOUS, and no scam
call falls below HIGH RISK.

Across all three splits (288 conversations): recall 0.979 / 0.958 / 1.000, false
positives 0.000 everywhere.

### What these numbers are, and are not

Three caveats, because they decide what this is ready for:

1. **The corpus is synthetic.** It is generated from 24 hand-annotated script
   families (14 scam / 10 legitimate) rendered into Russian, Kazakh and
   code-switched variants with realistic ASR noise. Splits are **by family**, so
   no test conversation is a paraphrase of a training one, and two scam families
   plus one legitimate family are withheld entirely as an `unseen_pattern` slice.
   That is a real generalisation test — but it is still a narrower distribution
   than production audio. Expect these numbers to drop on real calls.
2. **These are the `reference` backend's numbers, not a trained adapter's.**
   Fine-tuning needs a GPU. The reference backend is a transparent implementation
   of the same sequence model, and it exists so the whole system runs and is
   measurable without weights — and so the fine-tuned model has a real baseline
   to beat rather than a strawman.
3. **The score is a system risk estimate, not a calibrated probability.** There
   is no `97.384829%` anywhere in this codebase, deliberately.

---

## How it works

Every stage sits behind an interface, so any one can be replaced without touching
the others.

```
                    TELEPHONE CALL
                          │
  audio/ingestion.py      │  WebRTC / SIP frames · uploaded file · fixture
                          ▼
  audio/vad.py            │  energy VAD (no deps) │ Silero
                          ▼
  audio/diarization.py    │  CALLER / VICTIM / UNKNOWN     ← load-bearing
                          ▼
  audio/asr.py            │  faster-whisper │ whisper │ fixture
                          ▼
  transcription/          │  two transcripts: verbatim (evidence)
    normalizer.py         │  + normalized (model input)
                          ▼
  models/fraud_llm/       ★  FINE-TUNED FRAUD LLM
                          │  → scam types · tactics · stage · requested actions
                          │  → risk events, each carrying transcript evidence
                          ▼
  risk/engine.py          ★  DETERMINISTIC RISK ENGINE
                          │  → 0-100 score + full audit trail
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
  realtime/alerts.py                risk/engine.py::render_report
  live victim warning               final investigation report
```

**The LLM understands the conversation. The risk engine decides the number.**
The model never computes a score; the engine never calls a model. That split is
what makes the output both context-aware *and* auditable — every point in a score
traces back to a named rule in [`risk/rules.py`](risk/rules.py).

### Four properties that carry the design

**Speaker attribution is not optional.** `Не говорите никому свой SMS-код` and
`Назовите мне ваш SMS-код` share most of their vocabulary and mean opposite
things. Without a speaker label, the analysis layer cannot tell a warning from an
attack — so the risk engine **refuses to score a harmful request attributed to
anyone but the caller**, and records why it declined.

**Evidence cannot be invented.** Every finding must quote the transcript.
`FraudLLMBackend.ground()` locates each quote in the actual segments; anything it
cannot find is removed from the analysis, recorded in `dropped_findings`, and
contributes nothing. A hallucinated `SCAM` verdict with no grounded evidence
scores **0**.

**Normalization can never delete evidence.** Each cleanup rule is applied
speculatively and reverted if it would drop a protected token — a digit, an
amount, an OTP reference, an organisation, a code-switch. The verbatim ASR text is
kept untouched alongside the cleaned version.

**Code-switching is normal, never a signal.** `Сіздің картаңыз заблокирована` is
ordinary Kazakhstani speech. Language mixing is detected only so the UI and the
evaluator can slice by it, and there is a test asserting it does not move the
score.

### The risk engine

Weights are policy, set in one auditable file. The combination rule is the
interesting part:

```
score = 100 × (1 − Π(1 − cᵢ/100))     over capped group contributions
```

* **Group caps** — ten urgency mentions are not ten times one urgency mention.
* **Interaction effects** — `IMPERSONATION + OTP_REQUEST` scores higher than
  either alone, because the authority claim is what makes the request credible.
* **Mitigations** — protective advice, or a call the customer placed themselves,
  subtract.
* **Policy floors** — a CRITICAL credential/money/device request inside an
  impersonation frame is CRITICAL *regardless* of how the arithmetic lands.
* **Bands** — 0-29 SAFE · 30-59 SUSPICIOUS · 60-79 HIGH RISK · 80-100 CRITICAL.

When the model and the engine disagree, QORGAU **says so** (`disagreement`)
instead of silently picking one.

### Real-time mode

Analysis re-runs on every transcript chunk over a window of previous turns —
never on the newest sentence alone.

* **Peak-hold.** A caller who demands an OTP at 00:42 and then chats pleasantly is
  still dangerous at 01:30.
* **Immediate escalation.** A critical event alerts *at once*, without waiting for
  the total to cross 80. Waiting while the victim reads a code aloud defeats the
  point.

Observed live progression on a bank-impersonation call:

```
turn 1-3   risk   0  SAFE        greeting, identity claim, invented loan
turn 4     risk  35  SUSPICIOUS  urgency + threat         → MONITOR
turn 5     risk  80  CRITICAL    "Кодты айтып жіберіңіз"  → CRITICAL alert
```

```json
{"type": "risk_update", "risk_score": 82, "classification": "CRITICAL",
 "current_stage": "CREDENTIAL_EXTRACTION",
 "event": {"category": "OTP_REQUEST", "severity": "CRITICAL"}}
```

---

## Fine-tuning the model

```bash
pip install -r requirements-ml.txt
python -m training.prepare_dataset
python -m training.train --base-model Qwen/Qwen2.5-7B-Instruct
python -m training.evaluate --split test --backend local_adapter
python -m training.export --mode card
QORGAU_LLM_BACKEND=local_adapter streamlit run app/frontend/streamlit_app.py
```

QLoRA — 4-bit base, rank-32 adapters on all attention and MLP projections —
because the task is teaching a *behaviour* (read a call, emit this JSON) rather
than new language competence. That fits one consumer GPU and keeps the base model
swappable per deployment.

Two details that matter here:

* **Loss is computed on the JSON answer only.** The system prompt is ~2 kB of
  fixed instructions and the user turn is the transcript; training the model to
  predict those wastes capacity and biases it toward reciting the prompt.
* **No sequence packing.** A transcript boundary is semantically load-bearing, so
  conversations are never allowed to bleed into each other.

Training reports **JSON validity and classification agreement** from greedy
generation on validation calls, not just loss — the deliverable is parseable,
correct JSON.

### The corpus

[`training/corpus.py`](training/corpus.py) holds the annotated script families.
Annotation lives on the *script*, and rendering produces the surface forms — so a
family yields Russian, Kazakh and code-switched conversations, each optionally
degraded with dropped vowels, stutters, fillers and lost punctuation. Gold
evidence is taken from the **rendered** text, so it is always verbatim.

Hard negatives are ~40% of families by design: a customer asking to freeze a
card, a bank explaining that it never asks for codes, a police officer explaining
how to report a scam, a real courier confirming an address, a bank verifying a
genuine transaction *and refusing to take an OTP*. Without those, a model learns
`bank + money + OTP = scam`, which is useless in production.

To add real annotated calls, drop JSON files into `datasets/raw/` with `segments`
and a `gold` object; `prepare_dataset.py` mixes them in.

---

## Privacy and security

Call transcripts are sensitive financial data and the code treats them that way.

* **Masking before storage** — OTPs, PINs, CVVs and card numbers are replaced
  everywhere they can appear: the transcript column, event evidence, the model's
  explanation, and the risk engine's audit trail. There is a test that scans the
  raw database file for plaintext credentials.
* **Money amounts are *not* masked.** They are evidence, not secrets.
* **Encryption at rest** — verbatim text is Fernet-encrypted under
  `QORGAU_ENCRYPTION_KEY`. **With no key configured, QORGAU declines to store the
  verbatim text at all** rather than silently writing plaintext or pretending to
  encrypt it.
* **No public recording URLs** — recordings are internal references only; there is
  no column that could leak one.
* **Audit log** — every save, read, decrypt, delete and purge is recorded, with
  decryption flagged as its own action.
* **Retention and deletion** — per-call deadlines enforced by `purge_expired()`;
  `DELETE /api/calls/{id}` erases a call.

```bash
export QORGAU_ENCRYPTION_KEY=$(python -c \
  'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')
```

---

## API

```
POST   /api/calls                 register a transcribed call
POST   /api/calls/upload          upload a recording (VAD → diarization → ASR)
POST   /api/calls/{id}/analyze    analyse a completed call
GET    /api/calls/{id}            transcript + analysis
GET    /api/calls/{id}/events     suspicious events
GET    /api/calls/{id}/report     investigation report (markdown)
DELETE /api/calls/{id}            erase a call
WS     /api/calls/{id}/stream     live risk_update frames
POST   /api/admin/purge-expired   run the retention policy
```

---

## Tests

```bash
pytest tests/            # or: python tests/test_qorgau.py
```

29 behavioural tests, all passing. They assert the properties that make the
system trustworthy rather than implementation details:

* every legitimate family stays below the alarm threshold, every scam family
  reaches it
* a bank's warning about an OTP is not scored as a request for one
* victim speech never adds risk
* normalization never removes protected material
* ungrounded findings are discarded and score nothing
* the score is deterministic and bounded; repeated pressure is capped
* real-time risk is peak-hold and alerts immediately on critical events
* train and test share no script family; gold evidence is verbatim
* no credential reaches the database file in plaintext

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
├── datasets/       raw · processed · train · validation · test
├── config/         settings · ontology
└── tests/
```

## Configuration

| variable | default | purpose |
|---|---|---|
| `QORGAU_LLM_BACKEND` | `reference` | `reference` · `local_adapter` · `anthropic` |
| `QORGAU_BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | base model for fine-tuning |
| `QORGAU_ADAPTER` | `artifacts/adapters/qorgau-lora` | trained adapter path |
| `QORGAU_ASR` | `auto` | `faster_whisper` · `whisper` · `fixture` |
| `QORGAU_ASR_MODEL` | `small` | any Whisper size — `medium` / `large-v3` are better at Kazakh, and download several GB on first use |
| `QORGAU_ASR_LANGUAGE` | — | unset = detect per utterance (needed for code-switching); `ru` or `kk` pins a monolingual line |
| `QORGAU_VAD` | `energy` | `silero` |
| `QORGAU_DIAR` | `heuristic` | `pyannote` |
| `QORGAU_ENCRYPTION_KEY` | — | Fernet key for transcripts at rest |
| `QORGAU_DATABASE_URL` | SQLite under `artifacts/storage` | any SQLAlchemy URL |

---

## Known limitations

Stated plainly, because they determine what this is ready for:

* **The corpus is synthetic.** Real calls are messier, longer, more interrupted,
  and contain scripts nobody has written down yet.
* **Kazakh ASR is the weakest link end-to-end**, ahead of the analysis layer.
  Whisper's Kazakh is materially worse than its Russian, and it emits one language
  tag per segment — wrong for code-switched speech, so QORGAU re-derives language
  per utterance itself. It also mis-detects the language outright on short, noisy
  clips and transliterates instead of transcribing; QORGAU constrains detection to
  Kazakh and Russian and re-runs the clip when Whisper strays outside them. The
  default `small` model keeps the first run fast — raise `QORGAU_ASR_MODEL` to
  `medium` or `large-v3` for production Kazakh accuracy.
* **The reference backend is lexicon-bound.** It generalises across inflection,
  agglutination and noise, but a paraphrase built from vocabulary outside
  `lexicon.py` will be missed. That is precisely the gap the fine-tuned model
  closes, and why `unseen_pattern` is a reported slice.
* **The heuristic diarizer struggles with cross-talk.** It reports per-turn
  confidence so uncertain turns are visible; use `pyannote` in production.
* **Risk weights are policy, not learned.** `risk/calibration.py` sweeps them
  against labelled data, but a human sets them.
* **Policy floors outrank ASR dampening.** A critical request on a very
  low-confidence segment still alerts. That is a deliberate recall-first choice
  for a fraud detector; the uncertainty is surfaced rather than hidden.

## What the system will not do

It analyses and warns. It does not block accounts, move money, contact
authorities, or make any financial decision — and it says `SUSPICIOUS` rather
than `SCAM` when the evidence is thin, because a fraud detector that cries wolf
gets switched off within a week.
