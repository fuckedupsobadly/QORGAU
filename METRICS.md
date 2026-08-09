# QORGAU — measured results

Reproduce with:

```bash
python -m training.prepare_dataset
python -m training.evaluate --split test
```

Backend: `reference` — the offline behavioural analyser. See
[README](README.md#what-these-numbers-are-and-are-not) for why this, and not a
trained adapter, is the number reported here.

## Held-out test split

| metric | value |
|---|---|
| conversations | 72 (48 scam / 24 legitimate) |
| **scam recall** (risk ≥ 60) | **1.000** |
| **false-positive rate** (risk ≥ 60) | **0.000** |
| exact classification accuracy | 0.944 |
| scam-type micro F1 | 0.650 |
| tactic micro F1 | 0.741 |
| event-category micro F1 | 0.782 |
| JSON validity | 1.000 |
| evidence grounding rate | 1.000 |
| mean risk — scam / legitimate | 77.3 / 0.0 |
| mean latency per call | 4.4 ms |

Three detection thresholds are reported because they answer different questions:

| | scam recall | false-pos rate |
|---|---|---|
| model says exactly `SCAM` | 0.917 | 0.000 |
| model says `SCAM` or `SUSPICIOUS` | 1.000 | 0.000 |
| **risk engine ≥ 60** (what actually alerts) | **1.000** | **0.000** |

## All splits

| split | conversations | script families | scam recall | false-pos rate | exact acc |
|---|---|---|---|---|---|
| train | 180 | 15 | 0.979 | 0.000 | 0.906 |
| validation | 36 | 3 | 0.958 | 0.000 | 0.972 |
| test | 72 | 6 | 1.000 | 0.000 | 0.944 |

Splits are by **script family**. Family overlap between train and
validation/test: `{'train_vs_validation_family_overlap': [], 'train_vs_test_family_overlap': []}` — empty, i.e. no test conversation is a
paraphrase of a training one.

## Per slice (test)

| slice | n | scam recall | false-pos rate | exact acc | tactic F1 |
|---|---|---|---|---|---|
| `asr_noisy` | 36 | 1.000 | 0.000 | 0.944 | 0.735 |
| `kk` | 24 | 1.000 | 0.000 | 0.833 | 0.667 |
| `legitimate` | 24 | 0.000 | 0.000 | 1.000 | 0.000 |
| `mixed` | 24 | 1.000 | 0.000 | 1.000 | 0.792 |
| `obvious` | 24 | 1.000 | 0.000 | 1.000 | 0.757 |
| `ru` | 24 | 1.000 | 0.000 | 1.000 | 0.762 |
| `scam` | 48 | 1.000 | 0.000 | 0.917 | 0.741 |
| `subtle` | 24 | 1.000 | 0.000 | 0.833 | 0.721 |
| `unseen_pattern` | 36 | 1.000 | 0.000 | 1.000 | 0.778 |

`unseen_pattern` = two scam families and one legitimate family withheld from
training entirely; the closest thing here to a genuinely novel scam script.
`legitimate` has no recall figure by construction (no scams in it) — the number
that matters there is its false-positive rate.

## Risk band separation (test)

| band | scam calls | legitimate calls |
|---|---|---|
| SAFE | 0 | 24 |
| SUSPICIOUS | 0 | 0 |
| HIGH_RISK | 8 | 0 |
| CRITICAL | 40 | 0 |

No legitimate call reaches SUSPICIOUS; no scam call falls below HIGH RISK.
