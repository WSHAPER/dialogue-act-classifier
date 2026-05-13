# Training Plan

## Phase 1: English DistilBERT

### Setup

```bash
pip install torch transformers datasets accelerate optimum[onnxruntime] scikit-learn
```

### Dataset

- **Primary**: `eusip/silicone` config `dyda_da` (DailyDialog dialogue acts)
- **Augmentation**: Custom ASR edge cases (see `tests/test_cases.json`)
- **Labels** (native, no remapping): {0: commissive, 1: directive, 2: inform, 3: question}

### Hyperparameters

```yaml
base_model: distilbert-base-uncased
max_seq_length: 128
batch_size: 32
learning_rate: 2e-5
num_epochs: 4
warmup_ratio: 0.1
weight_decay: 0.01
seed: 42
```

### Evaluation

- Classification report (precision, recall, F1 per class)
- Custom edge-case test suite (`tests/test_cases.json`)
- Confusion matrix
- Comparison against TigreGotico baseline

### Export

1. `pytorch_model.bin` + `tokenizer.json` + `config.json` → for candle loading
2. Optional: ONNX export via `optimum-cli export onnx`

### Success criteria

- F1 > 0.90 on DailyDialog test set
- > 0.85 accuracy on custom ASR edge-case test suite
- "Can you hear me?" → question
- "Do something for me." → directive
- "Hey man, how's it going?" → question

## Phase 2: Multilingual

### Dataset

- **MIAM benchmark**: `Bingsu/MIAM` — EN, FR, DE, ES, IT dialogue acts
- **Base model**: `distilbert-base-multilingual-cased`
- **Russian**: Needs additional data (not in MIAM)

### Approach

1. Per-language fine-tuning (separate models)
2. OR: unified multilingual model trained on all languages
3. TBD based on Phase 1 results and MIAM dataset structure

### Language-specific classifiers

Each language gets its own model directory:
```
models/
├── en/  (distilbert-base-uncased fine-tuned on DailyDialog)
├── de/  (distilbert-base-multilingual-cased fine-tuned on MIAM-de)
├── es/  (...)
├── fr/  (...)
└── it/  (...)
```

phonolitui's cascade selects the right classifier based on `TranscriptChunk.language`.
