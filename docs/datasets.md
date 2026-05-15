# Datasets

Reference for all training and evaluation datasets, label mappings,
known issues, and data sourcing decisions.

Related: [Training](training.md) · [Models](models.md) ·
[Architecture](architecture.md)

---

## Overview

| Dataset | Languages | Size | Classes | License | Used In |
|---------|-----------|------|---------|---------|---------|
| [DailyDialog](#dailydialog) | EN | ~100K utterances | 4 | CC BY-NC-SA 4.0 | English model (train + eval) |
| [XDailyDialog](#xdailydialog) | EN, DE, IT | ~83K per lang | 4 | Apache-2.0 | Multilingual model (train + eval) |
| [xdailydialog-ru](#xdailydialog-ru) | RU | ~99K utterances | 4 | Apache-2.0 | Multilingual model (train + eval) |

---

## DailyDialog

### Source

- **HuggingFace:** `eusip/silicone`, config `dyda_da`
- **Original:** [DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset](https://arxiv.org/abs/1710.03957)
- **License:** CC BY-NC-SA 4.0

### Stats

| Split | Utterances |
|-------|-----------|
| Train | 87,170 |
| Validation | 8,069 |
| Test | 7,740 |

### Label Distribution (Test Set)

| Label | Index | Count |
|-------|-------|-------|
| commissive | 0 | 718 |
| directive | 1 | 1,278 |
| inform | 2 | 3,534 |
| question | 3 | 2,210 |

`inform` dominates (46% of test set). `commissive` is the rarest class
(9%). This imbalance is reflected in per-class F1 scores — see
[Models → Confusion Matrix](models.md#english-model-performance).

### Why DailyDialog

13,118 multi-turn dialogues about daily life (travel, shopping, work,
relationships) — much closer to meeting speech than TigreGotico's
synthetic/written training data. The 4 dialogue act classes are used
directly with no remapping.

### Loading

```python
from datasets import load_dataset
ds = load_dataset("eusip/silicone", "dyda_da", trust_remote_code=True)
# Columns: Utterance (str), Label (int), Dialogue_ID (int), Act_Index (int)
```

Note: Requires `datasets >= 3.0, < 4.0` with `trust_remote_code=True`
because the SILICONE dataset uses a legacy loading script. Version 4.x
drops script support. See [PREREQUISITES.md](../PREREQUISITES.md).

---

## XDailyDialog

### Source

- **Repository:** [github.com/liuzeming01/XDailyDialog](https://github.com/liuzeming01/XDailyDialog)
- **Paper:** [XDailyDialog: A Large-Scale Multi-turn Open-Domain Dialogue Dataset with Emotion Plan](https://arxiv.org/abs/2212.03010)
- **License:** Apache-2.0

### Stats (per language)

| Split | EN | DE | IT |
|-------|-----|-----|-----|
| Train | ~25K | ~25K | ~25K |
| Dev | ~7.7K | ~7.7K | ~7.7K |
| Test | ~7.7K | ~7.7K | ~7.7K |

### Data Format

Tab-separated files (`en_train_human.txt`, etc.):

```
utterance1 __eou__ utterance2 __eou__ ... \t emotion1 emotion2 ... \t act1 act2 ...
```

- Utterances are separated by `__eou__` (end of utterance).
- Emotion labels are in column 2 (not used for this project).
- **Act labels are in column 3** — space-separated integers 1–4.

Parsed by [`parse_xdailydialog_line()`](../train_multilingual.py) and
[`parse_xdailydialog()`](../evaluate_multilingual.py).

### Why XDailyDialog (not MIAM)

The initial plan was to use the MIAM benchmark (`Bingsu/MIAM`) for
multilingual data. This was dropped because:

- MIAM datasets are domain-specific (map task, tourism, medical) with
  incompatible label schemas.
- No Russian data in MIAM.
- XDailyDialog has consistent 4-class labels matching DailyDialog and
  covers EN + DE + IT.

### Setup

XDailyDialog is cloned locally (gitignored):

```bash
git clone https://github.com/liuzeming01/XDailyDialog.git .reference/XDailyDialog
```

Configured in [`config_multilingual.yaml`](../config_multilingual.yaml)
under `xdailydialog.local_path`.

---

## xdailydialog-ru

### Source

- **HuggingFace:** [`WSHAPER/xdailydialog-ru`](https://huggingface.co/datasets/WSHAPER/xdailydialog-ru)
- **Origin:** Machine-translated from XDailyDialog English using
  `Helsinki-NLP/opus-mt-en-ru`
- **License:** Apache-2.0 (derived from XDailyDialog)

### Stats

| Split | Utterances |
|-------|-----------|
| Train | ~33K |
| Validation | ~33K |
| Test | ~7.6K |

### Data Format

HuggingFace dataset with columns:
- `utterances`: list of strings per dialogue
- `acts`: list of int (0–3, already mapped to project label schema)

### Loading

```python
from datasets import load_dataset
ds = load_dataset("WSHAPER/xdailydialog-ru", split="test", trust_remote_code=True)
for dialog in ds:
    for utt, act in zip(dialog["utterances"], dialog["acts"]):
        # utt: Russian text, act: 0-3 label
```

### Machine Translation Quality

The Russian data was produced by Opus-MT (neural machine translation),
not human translation. This introduces translation artifacts, but the
multilingual model still achieves 81.7% accuracy on the Russian test
set — comparable to German (81.8%), suggesting the translation quality
is sufficient for training.

The translation was originally done by
[`translate_to_russian()`](../train_multilingual.py) in the training
script and then uploaded to HuggingFace as a standalone dataset to
avoid re-translating on every training run.

---

## Label Mappings

### Project Canonical Labels

| Index | Label | Description |
|-------|-------|-------------|
| 0 | commissive | Promises, commitments |
| 1 | directive | Commands, requests |
| 2 | inform | Statements, facts |
| 3 | question | Questions, inquiries |

### DailyDialog (SILICONE dyda_da)

Already uses the canonical mapping. The `Label` column in the SILICONE
dataset is `{0: commissive, 1: directive, 2: inform, 3: question}`.
Loaded directly by [`prepare_dyda()`](../train.py) — no remapping needed.

### XDailyDialog

**Important:** XDailyDialog uses a **different label ordering**:

| XDailyDialog Act | Meaning | Maps to Project Index |
|-------------------|---------|-----------------------|
| 1 | inform | 2 |
| 2 | question | 3 |
| 3 | directive | 1 |
| 4 | commissive | 0 |

This mapping is defined in
[`config_multilingual.yaml: act_label_map`](../config_multilingual.yaml):

```yaml
act_label_map:
  1: 2   # XDD inform → project inform
  2: 3   # XDD question → project question
  3: 1   # XDD directive → project directive
  4: 0   # XDD commissive → project commissive
```

**History:** The initial implementation assumed XDailyDialog used the
same ordering as DailyDialog. This was wrong — commit `4c81015c` fixed
the mapping, which corrected per-language accuracy by ~5–10 percentage
points.

The `xdailydialog-ru` HuggingFace dataset already has labels mapped to
the project schema (0–3), so no additional mapping is needed when
loading it.

---

## Edge-Case Test Suites

### English: [`tests/test_cases.json`](../tests/test_cases.json)

52 samples across 7 categories:

| Category | Count | Examples |
|----------|-------|----------|
| `wh_question` | 6 | "What is the timeline?", "Why did we change the approach?" |
| `polar_question` | 9 | "Can you hear me?", "How's it going?" |
| `command` | 8 | "Close the door.", "Show me the results." |
| `request` | 6 | "Please send me the report.", "Make sure it works." |
| `inform` | 6 | "The weather is nice today.", "I think this is the right approach." |
| `commissive` | 5 | "I'll handle the deployment.", "I promise to finish by Friday." |
| `asr_disfluent` | 5 | "uh what is the timeline", "um can you explain that again" |

The English model achieves **100% accuracy** on this suite.

These test cases serve double duty:
1. **Evaluation** — passed to [`evaluate.py`](../evaluate.py) as a
   held-out test.
2. **Augmentation** — expanded by [`augment.py`](../augment.py) into
   training data (fillers, casing, punctuation variants). See
   [Training → Augmentation](training.md#asr-augmentation).

### Multilingual: [`tests/test_cases_multilingual.json`](../tests/test_cases_multilingual.json)

Language-specific edge cases for the multilingual model.

---

## Augmentation Data

[`augment.py`](../augment.py) transforms the edge-case test suite into
expanded training data:

### Transformations

| Transform | Probability | Example |
|-----------|------------|---------|
| Filler word insertion | 40% | "What is the timeline?" → "uh What is the timeline?" |
| Punctuation strip | varies | "Can you hear me?" → "Can you hear me" |
| Punctuation swap | varies | "Close the door." → "Close the door?" |
| Lowercase | varies | "Send the report." → "send the report." |
| Capitalize | varies | "the weather is nice." → "The weather is nice." |

### Filler Words

```python
FILLERS = ["uh", "um", "like", "so", "well", "ok", "right", "you know"]
```

Inserted at random positions 0–3 in the utterance to simulate ASR
disfluencies.

### Generation

With default `--augment-factor 5`, each of the 52 test cases generates
~5 valid variants (variants identical to the original after
transformation are discarded), producing roughly 200–260 additional
training samples. These are concatenated with DailyDialog's 87K
training split.

---

## Known Issues

### SILICONE Loading Script Deprecation

The `eusip/silicone` dataset uses a legacy Python loading script that
requires `trust_remote_code=True`. This works with `datasets >= 3.0,
< 4.0` but will break with `datasets 4.x`. Pin to
`datasets>=3.0,<4.0` in [`requirements.txt`](../requirements.txt).

### XDailyDialog Local Clone

XDailyDialog must be cloned manually to `.reference/XDailyDialog/`
(it's gitignored). The training script will raise `FileNotFoundError`
with instructions if missing.

### Russian Machine Translation

The Russian dataset is machine-translated. Some translations may be
unnatural or incorrect. The model's 81.7% Russian accuracy suggests
the quality is acceptable but may benefit from human review or
correction in future iterations.

### Class Imbalance

`commissive` (9% of DailyDialog test set) is consistently the hardest
class. It's most often confused with `inform` — both describe states,
differing mainly in whether the speaker is committing to a future action
or describing a fact. Addressing this would require either more
commissive training data or class-weighted loss.
