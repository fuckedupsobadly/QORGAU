# QORGAU

### AI-powered detection of telephone scams in Kazakh and Russian

QORGAU is an AI-powered fraud detection system designed to analyze telephone conversations in **Kazakh, Russian, and mixed Kazakh-Russian speech** and identify social-engineering and financial scam patterns in real time.

Instead of relying on keywords such as `"банк"`, `"код"` or `"деньги"`, QORGAU analyzes the **context, intent, conversational sequence, manipulation tactics, and actions requested from the victim**.

> From keyword detection to behavioral fraud intelligence.

---

## Problem

Modern telephone scams are becoming increasingly sophisticated.

A scammer may impersonate:

* a bank employee
* a police officer
* a government official
* a delivery service
* an investment company
* technical support

The attack usually happens as a sequence:

```text
Impersonation
      ↓
False Problem
      ↓
Fear / Authority
      ↓
Urgency
      ↓
Trust Manipulation
      ↓
OTP / Credential Request
      ↓
Money Transfer
```

Traditional keyword-based systems struggle with these attacks because individual phrases are not necessarily suspicious.

For example:

> "Никому не сообщайте код из SMS."

and

> "Продиктуйте мне код из SMS."

contain almost the same keywords but have completely different intent.

QORGAU focuses on **who said something, why they said it, and what they are trying to make the victim do.**

---

# Goals

QORGAU is designed to:

* Detect telephone scams in real time
* Understand Kazakh and Russian speech
* Handle natural Kazakh-Russian code-switching
* Identify social-engineering tactics
* Detect suspicious victim requests
* Extract evidence from conversations
* Provide an explainable risk score
* Identify the stage of an ongoing scam
* Minimize false positives on legitimate conversations

---

# Architecture

```text
                         PHONE CALL
                              │
                              ▼
                    ┌───────────────────┐
                    │  Audio Ingestion  │
                    │   WebRTC / SIP    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │       VAD         │
                    │ Voice Activity    │
                    │ Detection         │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │    Diarization    │
                    │                   │
                    │ CALLER / VICTIM   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Speech-to-Text  │
                    │                   │
                    │ KZ / RU / Mixed   │
                    └─────────┬─────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │      Structured Transcript  │
              │                              │
              │ speaker                      │
              │ timestamp                    │
              │ language                     │
              │ text                         │
              │ ASR confidence               │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   Fine-Tuned Fraud LLM       │
              │                              │
              │ Classification               │
              │ Intent Detection              │
              │ Tactic Detection              │
              │ Event Extraction              │
              │ Evidence Extraction           │
              │ Conversation Stage            │
              └──────────────┬───────────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │    Risk Engine    │
                    │                   │
                    │ LLM signals       │
                    │ + deterministic   │
                    │   rules           │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             REAL-TIME ALERT          REPORT
```

## Core principle

The **LLM understands the conversation**.

The **risk engine calculates the final risk**.

This separation makes the system more explainable, deterministic, and easier to evaluate.

---

# AI Pipeline

## 1. Audio Ingestion

QORGAU accepts a live audio stream or recorded call.

Planned interfaces:

```text
WebRTC
SIP
WAV
MP3
M4A
OGG
```

---

## 2. Voice Activity Detection

VAD identifies periods of speech and removes unnecessary silence.

```text
Audio
  ↓
VAD
  ↓
Speech segments
```

This reduces unnecessary ASR computation and helps with real-time processing.

---

## 3. Speaker Diarization

The system identifies who is speaking:

```text
CALLER
VICTIM
UNKNOWN
```

Example:

```text
[00:03] CALLER:
Сәлеметсіз бе, это служба безопасности банка.

[00:08] VICTIM:
Иә, тыңдап тұрмын.

[00:12] CALLER:
На ваше имя оформляется кредит.
```

Speaker attribution is essential for intent detection.

For example:

```text
VICTIM:
Не сообщайте никому код из SMS.
```

is very different from:

```text
CALLER:
Сообщите мне код из SMS.
```

---

# Multilingual Speech Recognition

QORGAU is designed around:

* Kazakh
* Russian
* Mixed Kazakh-Russian speech

Real conversations frequently contain code-switching:

```text
"Сіздің картаңыз заблокирована."

"Қазір вам SMS келеді."

"Кодты айтып жіберіңіз, я сейчас проверю."

"Ақшаңызды безопасный счетқа аудару керек."
```

The system must treat this as normal language behavior rather than a fraud signal.

---

# Fine-Tuned Fraud LLM

The core intelligence layer is a **fine-tuned multilingual language model**.

The model is trained on conversational fraud examples rather than isolated keywords.

### Model responsibilities

The LLM extracts:

* Scam classification
* Scam type
* Manipulation tactics
* Conversation stage
* Requested victim actions
* Suspicious events
* Evidence
* Explanation

---

# Fraud Ontology

## Scam Types

```text
BANK_IMPERSONATION
POLICE_IMPERSONATION
GOVERNMENT_IMPERSONATION
INVESTIGATOR_IMPERSONATION

ACCOUNT_COMPROMISE
FAKE_LOAN
FAKE_TRANSACTION
FAKE_REFUND

OTP_THEFT
CARD_DATA_THEFT
CREDENTIAL_THEFT

SAFE_ACCOUNT_SCAM
MONEY_TRANSFER_SCAM

REMOTE_ACCESS_SCAM

INVESTMENT_SCAM
CRYPTO_SCAM

MARKETPLACE_SCAM
DELIVERY_SCAM

ROMANCE_SCAM
JOB_SCAM

OTHER_SOCIAL_ENGINEERING
```

## Manipulation Tactics

```text
IMPERSONATION
URGENCY
FEAR
AUTHORITY
SECRECY
ISOLATION
FALSE_PROBLEM
FALSE_SOLUTION

OTP_REQUEST
CARD_REQUEST
PASSWORD_REQUEST

MONEY_TRANSFER_REQUEST
REMOTE_ACCESS_REQUEST
INSTALLATION_REQUEST
SCREEN_SHARING

CRYPTO_TRANSFER
```

---

# Conversation Stages

QORGAU models the progression of an attack:

```text
INTRODUCTION
      ↓
IDENTITY_CLAIM
      ↓
PROBLEM_CREATION
      ↓
FEAR_ESCALATION
      ↓
TRUST_BUILDING
      ↓
INFORMATION_EXTRACTION
      ↓
CREDENTIAL_EXTRACTION
      ↓
MONEY_TRANSFER
      ↓
REMOTE_ACCESS
      ↓
EXIT
```

This enables the system to detect an attack **before the final financial loss occurs**.

---

# Risk Events

The model extracts individual suspicious events.

Example:

```json
{
  "timestamp": "00:42",
  "speaker": "CALLER",
  "category": "OTP_REQUEST",
  "severity": "CRITICAL",
  "evidence": "Продиктуйте код из SMS",
  "reason": "Caller requests a one-time authentication code."
}
```

Possible categories:

```text
IMPERSONATION
THREAT
URGENCY
FEAR
SECRECY
OTP_REQUEST
CARD_REQUEST
PASSWORD_REQUEST
MONEY_TRANSFER
SAFE_ACCOUNT
REMOTE_ACCESS
SCREEN_SHARING
APP_INSTALLATION
CRYPTO_TRANSFER
```

Every event must contain evidence.

The model must never fabricate evidence.

---

# Training Strategy

The model should be fine-tuned using conversational examples.

Each example contains:

```text
Conversation
      ↓
Classification
      ↓
Scam Type
      ↓
Tactics
      ↓
Conversation Stage
      ↓
Victim Action
      ↓
Evidence
```

Example:

```json
{
  "conversation": [
    {
      "speaker": "CALLER",
      "text": "Сәлеметсіз бе, это служба безопасности банка."
    },
    {
      "speaker": "VICTIM",
      "text": "Иә."
    },
    {
      "speaker": "CALLER",
      "text": "На ваше имя оформляется кредит."
    },
    {
      "speaker": "VICTIM",
      "text": "Я ничего не оформлял."
    },
    {
      "speaker": "CALLER",
      "text": "Сейчас вам придет SMS. Продиктуйте код."
    }
  ],

  "label": {
    "classification": "SCAM",
    "scam_types": [
      "BANK_IMPERSONATION",
      "OTP_THEFT"
    ],
    "tactics": [
      "IMPERSONATION",
      "FEAR",
      "URGENCY",
      "OTP_REQUEST"
    ]
  }
}
```

---

# Hard Negatives

Reducing false positives is a major part of the training strategy.

The dataset must contain legitimate conversations such as:

```text
Customer calling a bank about a loan

Customer asking about a failed transaction

Bank employee explaining OTP security

Customer reporting suspicious activity

Police explaining how to report fraud

Customer asking how to freeze a card
```

For example:

> "Никому не сообщайте код из SMS."

must **not** be classified as OTP theft.

The model must understand **who requested the information and why**.

---

# Kazakhstan-First Threat Intelligence

QORGAU is optimized for local fraud scenarios involving:

* Banks
* eGov
* Government services
* Police
* Delivery companies
* Marketplaces
* Investment platforms
* Cryptocurrency
* Fake loans
* Fake transactions
* "Safe account" scams

The system should understand local context without assuming that mentioning a specific organization means fraud.

---

# Real-Time Mode

QORGAU is designed to support streaming analysis.

```text
Audio
 ↓
5–10 second chunk
 ↓
ASR
 ↓
Conversation State
 ↓
Fine-Tuned LLM
 ↓
Risk Engine
 ↓
Risk Update
 ↓
Alert
```

The model should maintain context across multiple chunks instead of classifying each sentence independently.

Example:

```text
00:05 → Identity claim
00:18 → False problem
00:32 → Fear
00:47 → OTP request
00:55 → Money transfer
```

The risk score evolves as the attack develops.

---

# Explainable Detection

QORGAU does not simply output:

```text
SCAM
```

It provides evidence.

Example:

```json
{
  "classification": "SCAM",
  "risk_score": 94,
  "scam_types": [
    "BANK_IMPERSONATION",
    "OTP_THEFT",
    "SAFE_ACCOUNT_SCAM"
  ],
  "risk_factors": [
    {
      "timestamp": "00:12",
      "category": "IMPERSONATION",
      "severity": "HIGH",
      "evidence": "Это служба безопасности банка."
    },
    {
      "timestamp": "00:42",
      "category": "OTP_REQUEST",
      "severity": "CRITICAL",
      "evidence": "Продиктуйте код из SMS."
    },
    {
      "timestamp": "01:04",
      "category": "MONEY_TRANSFER",
      "severity": "CRITICAL",
      "evidence": "Переведите деньги на безопасный счет."
    }
  ]
}
```

This allows the user or investigator to understand **why the system generated the alert**.

---

# Risk Engine

The LLM does not directly determine the final production risk score.

Instead, detected events are passed to a deterministic risk engine.

Initial signals may include:

```text
Bank impersonation       +20
OTP request               +35
Money transfer request   +35
Remote access             +35
Urgency                   +10
Fear                      +10
Secrecy                   +15
Safe account              +40
```

Signals can also interact.

For example:

```text
BANK_IMPERSONATION
+
OTP_REQUEST
+
MONEY_TRANSFER_REQUEST
```

should result in a substantially higher risk than any individual signal.

### Risk Levels

|  Score | Level      |
| -----: | ---------- |
|   0–29 | SAFE       |
|  30–59 | SUSPICIOUS |
|  60–79 | HIGH RISK  |
| 80–100 | CRITICAL   |

The score is a **system risk estimate**, not a calibrated probability.

---

# Evaluation

QORGAU should not be evaluated using accuracy alone.

Important metrics include:

* Scam Recall
* Precision
* F1 Score
* False Positive Rate
* Scam Type F1
* Tactic Detection F1
* Evidence Extraction Accuracy
* JSON Validity
* Kazakh performance
* Russian performance
* Mixed-language performance
* ASR-noise robustness
* Unseen-scam generalization

The test set should contain **previously unseen scam scripts** to measure whether the model actually generalizes.

---

# Dataset Design

The dataset should contain:

```text
SCAM
├── Bank impersonation
├── Police impersonation
├── Fake loan
├── Fake transaction
├── OTP theft
├── Safe account
├── Remote access
├── Investment
├── Cryptocurrency
├── Marketplace
└── Delivery

SAFE
├── Legitimate bank support
├── Loan inquiry
├── Transaction support
├── Card replacement
├── Fraud reporting
└── Security education
```

A significant portion of the dataset should contain naturally mixed Kazakh-Russian conversations.

Training data should also include:

* ASR errors
* filler words
* interruptions
* incomplete sentences
* colloquial speech
* numbers
* background noise
* code-switching

---

# Project Structure

```text
qorgau/
│
├── app/
│   ├── api/
│   ├── websocket/
│   └── frontend/
│
├── audio/
│   ├── ingestion.py
│   ├── vad.py
│   ├── diarization.py
│   └── asr.py
│
├── transcription/
│   ├── processor.py
│   ├── normalizer.py
│   └── schemas.py
│
├── models/
│   ├── fraud_llm/
│   ├── inference.py
│   └── prompts.py
│
├── risk/
│   ├── engine.py
│   ├── rules.py
│   └── calibration.py
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── training/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
│
├── database/
│   ├── models.py
│   └── repository.py
│
├── tests/
│
├── config/
│
├── requirements.txt
└── README.md
```

---

# Tech Stack

## AI / ML

* Python
* PyTorch
* Hugging Face Transformers
* PEFT
* LoRA / QLoRA
* Multilingual ASR
* Fine-tuned LLM

## Backend

* FastAPI
* WebSockets
* PostgreSQL

## Frontend

* React
* Next.js

## Infrastructure

* Docker
* GPU inference
* REST API
* WebSocket streaming

---

# Privacy and Security

Call recordings and transcripts may contain sensitive financial information.

QORGAU follows a **Privacy-by-Design** approach.

Planned protections:

* Encryption in transit
* Encryption at rest
* Access control
* Audit logging
* Data retention policies
* Secure deletion
* PII masking
* Protection of authentication credentials
* No public access to raw recordings

Sensitive information such as OTPs, PINs, CVVs, and passwords should not be retained unnecessarily.

---

# Roadmap

## Phase 1 — MVP

* [ ] Audio upload
* [ ] Speech-to-text
* [ ] Kazakh/Russian transcription
* [ ] Fine-tuned fraud LLM
* [ ] Scam classification
* [ ] Risk engine
* [ ] Explainable alerts

## Phase 2 — Real-Time

* [ ] Streaming audio
* [ ] Real-time ASR
* [ ] Speaker diarization
* [ ] Incremental LLM inference
* [ ] Real-time risk updates
* [ ] Live alerts

## Phase 3 — Production

* [ ] Large-scale fraud dataset
* [ ] Model evaluation framework
* [ ] Model monitoring
* [ ] Continuous fine-tuning
* [ ] Telecom integration
* [ ] Banking API integration
* [ ] Fraud intelligence dashboard

---

# Vision

QORGAU aims to become an **AI security layer for voice communication**.

Instead of detecting fraud after money has been stolen, the goal is to detect the attack while the conversation is still happening.

```text
Attacker
   ↓
Phone Call
   ↓
┌──────────────────┐
│      QORGAU      │
│   AI Voice Shield│
└────────┬─────────┘
         ↓
      Warning
         ↓
       Victim
```

## The mission

> **Detect the manipulation before it becomes a financial loss.**

---

# QORGAU

**Real-time conversational fraud intelligence for Kazakhstan.**

**Қорғау начинается с разговора.**
