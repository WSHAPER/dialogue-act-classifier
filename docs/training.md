# Training

English and multilingual fine-tuning pipelines, data augmentation,
hyperparameters, and evaluation results.

Related: [Architecture](architecture.md) · [Models](models.md) ·
[Datasets](datasets.md)

---

## English Pipeline

### Script

[`train.py`](../train.py) — fine-tunes `distilbert-base-uncased` on
DailyDialog dialogue acts.

### Configuration

From [`config.yaml`](../config.yaml):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `base_model` | `distilbert-base-uncased` | 67M params, fits alongside Whisper in memory |
| `max_seq_length` | 128 | Training padding length (p95 of utterances is ~35 tokens) |
| `inference_max_length` | 48 | Decoupled from training — covers p95 of inputs while avoiding wasted compute |
| `batch_size` | 32 | Fills RTX A3000 without OOM |
| `learning_rate` | 2e-5 | Standard for BERT fine-tuning |
| `num_epochs` | 4 | Early stopping via `load_best_model_at_end` (F1 macro) |
| `warmup_ratio` | 0.1 | Linear warmup for 10% of steps |
| `weight_decay` | 0.01 | Regularization |
| `seed` | 42 | Reproducibility |

### Data Loading

[`load_dailydialog()`](../train.py) loads `eusip/silicone` config
`dyda_da` — the SILICONE-hosted DailyDialog dataset with official
train/validation/test splits. Columns `Utterance` → `text`, `Label` →
`label`. Labels are used directly (no remapping):

| Index | Label | Description |
|-------|-------|-------------|
| 0 | commissive | "I'll handle the deployment." |
| 1 | directive | "Send the report." |
| 2 | inform | "The deadline is Friday." |
| 3 | question | "What is the timeline?" |

See [Datasets → DailyDialog](datasets.md#dailydialog) for dataset stats.

### ASR Augmentation

[`augment.py`](../augment.py) expands the 52 samples in
[`tests/test_cases.json`](../tests/test_cases.json) into thousands of
training examples via:

1. **Filler insertion** — randomly inserts `"uh"`, `"um"`, `"like"`,
   `"so"`, `"well"`, `"ok"`, `"right"`, `"you know"` at positions 0–3.
2. **Punctuation variants** — strips, adds `.`, `?`, `!` to test
   robustness to ASR transcript inconsistency.
3. **Casing variants** — `lower()`, `capitalize()`.

Each test case generates ~5 augmented variants (configurable via
`--augment-factor`). The augmented dataset is concatenated with
DailyDialog training split before tokenization.

The augmentation targets conversational edge cases that DailyDialog
doesn't cover well — things like `"Can you hear me?" → question` and
`"Do something for me." → directive` that the previous TigreGotico
model got wrong. See [Models → Baseline Comparison](models.md#baseline-comparison).

### Training Loop

Standard `transformers.Trainer` with:
- `fp16=True` when CUDA is available
- `eval_strategy="epoch"`, `save_strategy="epoch"`
- `metric_for_best_model="f1_macro"`, `load_best_model_at_end=True`
- `dataloader_num_workers=0` (avoid fork issues)

### English Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 83.4 % |
| **F1 macro** | 0.7617 |
| **Edge-case accuracy** | 45/45 (100 %) |

The 100% edge-case score is the key result — the model handles all
conversational/ASR patterns that the TigreGotico TF-IDF model failed on.
See [`eval_results/evaluation_results.json`](../eval_results/evaluation_results.json).

---

## Multilingual Pipeline

### Script

[`train_multilingual.py`](../train_multilingual.py) — fine-tunes
`distilbert-base-multilingual-cased` on combined EN + DE + RU data.

### Configuration

From [`config_multilingual.yaml`](../config_multilingual.yaml):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `base_model` | `distilbert-base-multilingual-cased` | 277M params, 104 languages |
| `num_epochs` | 5 | One more epoch than English (more diverse data) |
| `max_seq_length` | 128 | Same as English |
| Batch/LR/warmup | Same as English | Proven hyperparameters |

### Data Sources

The multilingual pipeline merges three sources — see
[Datasets](datasets.md) for full details:

1. **SILICONE dyda_da** (EN) — same DailyDialog as the English model.
   Loaded by [`load_silicone_dyda()`](../train_multilingual.py).
2. **XDailyDialog** (EN + DE + IT) — cloned from GitHub into
   `.reference/XDailyDialog/`. Loaded by
   [`load_xdailydialog()`](../train_multilingual.py) which parses the
   tab-separated `__eou__`-delimited format and applies the
   [`act_label_map`](datasets.md#label-mappings).
3. **xdailydialog-ru** (RU) — loaded directly from HuggingFace
   `WSHAPER/xdailydialog-ru`. Originally machine-translated from
   XDailyDialog English using Opus-MT (`Helsinki-NLP/opus-mt-en-ru`).

The training script also supports live translation via
[`translate_to_russian()`](../train_multilingual.py) if the HF dataset
is unavailable, but this path is rarely needed now.

### Multilingual Results

Per-language performance on test sets:

| Language | Test Set | N | Accuracy | F1 Macro |
|----------|----------|---|----------|----------|
| English | SILICONE dyda_da | 7,740 | 80.8 % | 0.725 |
| English | XDailyDialog | 7,716 | 82.5 % | 0.750 |
| German | XDailyDialog | 7,716 | 81.8 % | 0.738 |
| Russian | xdailydialog-ru | 7,623 | 81.7 % | 0.734 |

See [`eval_results/multilingual_per_language.json`](../eval_results/multilingual_per_language.json).

The multilingual model trades ~3 percentage points of English accuracy
for covering DE and RU with a single model. If you only need English,
use the monolingual model — see [Models → Which Model](models.md#which-model-should-i-use).

---

## Evaluation

### English Evaluation

[`evaluate.py`](../evaluate.py) runs three evaluation modes:

1. **DailyDialog test set** — full classification report + confusion matrix
   saved to [`eval_results/confusion_matrix.png`](../eval_results/confusion_matrix.png).
2. **Edge-case test suite** — 52 samples from
   [`tests/test_cases.json`](../tests/test_cases.json) covering WH-questions,
   polar questions, commands, requests, inform, commissive, and ASR disfluent
   categories. All must pass.
3. **TigreGotico baseline comparison** — if the TigreGotico ONNX model is
   available locally, runs the same edge cases through it for a direct
   accuracy comparison.

### Multilingual Evaluation

[`evaluate_multilingual.py`](../evaluate_multilingual.py) evaluates on
each language's test set independently, using the same model checkpoint.

### Quantized Model Benchmarking

Both `evaluate.py` (`--quantized-model`) and `quantize.py` (default)
measure ONNX Runtime latency percentiles (p50/p95/p99) alongside accuracy.
See [Optimization → Results](optimization.md#results-summary) for the
full latency progression.

---

## CLI Reference

```bash
# English training (defaults from config.yaml)
python train.py
python train.py --epochs 6 --lr 3e-5    # override hyperparams
python train.py --no-augment             # skip ASR augmentation
python train.py --push-to-hub WSHAPER/...  # publish to HuggingFace

# Multilingual training
python train_multilingual.py
python train_multilingual.py --languages en de   # subset
python train_multilingual.py --skip-ru           # no Russian

# Evaluation
python evaluate.py                              # all modes
python evaluate.py --model-path models/en/run_20260513_120000
python evaluate.py --quantized-model models/en/final/quantized/fp16

# Multilingual evaluation
python evaluate_multilingual.py
python evaluate_multilingual.py --xdailydialog-path .reference/XDailyDialog/data

# Export
python export.py --onnx                         # ONNX + candle
python export.py --quantize --quantize-mode fp16

# Augmentation only (inspect what gets generated)
python augment.py --factor 10 --output tests/augmented_cases.json
```
