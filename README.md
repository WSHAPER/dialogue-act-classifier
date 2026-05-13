# phonolitui-sentence-classifier

Fine-tuned DistilBERT model for **4-class dialogue act classification** (commissive / directive / inform / question), optimized for **conversational ASR transcripts** — the classification backend for [phonolitui](https://github.com/wilmergaz88/phonolitui).

## Why this exists

phonolitui is a privacy-first, terminal-native AI meeting assistant. It transcribes meetings locally via Whisper and classifies each utterance into one of **4 dialogue act classes** to power features like:
- Auto-highlighting questions in the transcript
- Routing questions to LLM Q&A
- Extracting action items (directives) from the meeting
- Tracking commitments/promises (commissives)
- TUI footer labels showing sentence type per chunk

The 4-class output is mapped to phonolitui's 3-class `SentenceType` at inference time:
- `commissive` → Statement
- `directive` → Instruction
- `inform` → Statement
- `question` → Question

### The problem with off-the-shelf models

We evaluated 5+ models before deciding to fine-tune:

| Model | Approach | Issue |
|-------|----------|-------|
| **TigreGotico/sentence-types** (current) | TF-IDF + linear classifier | Trained on synthetic/written text. Fails on conversational edge cases: "Can you hear me?" → `request` (should be question), "Do something for me." → `polar_question` (should be instruction). 99.2% on written data, ~70% on ASR. |
| mrsinghania/asr-question-detection | Distilled BERT, 2-class | Binary only (question/statement). Cannot detect instructions at all. |
| facebook/bart-large-mnli | Zero-shot NLI | 1.6GB, slow inference, English-only, weaker than fine-tuned. |
| ESM-based SILICONE models | Protein sequence models | Not text models, unsuitable. |
| Various dialogue act papers | Research code only | No published weights on HuggingFace. |

**No pre-trained model exists** that does 4-class dialogue act classification well on conversational/ASR data with publicly available weights.

## Approach

Fine-tune **DistilBERT** (`distilbert-base-uncased`, 67M params) on DailyDialog dialogue acts using the dataset's native 4-class labels:

| Class | Label | Index | Description |
|-------|-------|-------|-------------|
| Commissive | `commissive` | 0 | Promises, commitments ("I'll handle it.") |
| Directive | `directive` | 1 | Commands, requests ("Send the report.") |
| Inform | `inform` | 2 | Statements, facts ("The deadline is Friday.") |
| Question | `question` | 3 | Questions, inquiries ("What is the timeline?") |

### Why DistilBERT

- **Small**: 67M params, ~250MB — fits in memory alongside Whisper
- **Fast**: ~5-15ms CPU inference via candle (pure Rust, no ONNX runtime needed)
- **Compatible**: Loads directly with `candle-transformers` using the exact same pattern phonolitui already has for BERT classification
- **No CUDA conflict**: Pure CPU inference via candle, no `ort`/ONNX Runtime competing with Whisper's GPU context

### Why DailyDialog

- **Conversational**: 13,118 multi-turn dialogues about daily life — much closer to meeting speech than TigreGotico's synthetic data
- **100K+ utterances** with train/val/test splits
- **4 dialogue act classes** used directly — no remapping, preserving the full label granularity
- **HuggingFace dataset**: `eusip/silicone` config `dyda_da`

## Training Plan

### Phase 1: English model

1. Load `distilbert-base-uncased` + `eusip/silicone` (dyda_da)
2. Use native 4-class labels (commissive=0, directive=1, inform=2, question=3)
3. Fine-tune with `transformers.Trainer` (15-30 min on single GPU)
4. Augment training data with ASR edge cases:
   - "Can you hear me?" → question
   - "Do something for me." → directive
   - "Hey man, how's it going?" → question
   - "Please send the report." → directive
   - "Close the door." → directive
   - "I think this is the right approach." → inform
   - "I'll handle the deployment." → commissive
   - ASR artifacts: "uh what is the timeline" → question
5. Evaluate on held-out test set + custom ASR edge-case test suite
6. Export: `pytorch_model.bin` + `tokenizer.json` + `config.json`
7. Optional ONNX export via `optimum-cli export onnx`

### Phase 2: Multilingual model

1. Base model: `distilbert-base-multilingual-cased` (104 languages)
2. Dataset: MIAM benchmark (`Bingsu/MIAM`) — EN, FR, DE, ES, IT
3. Same label mapping, language-specific training
4. Language-specific OR unified multilingual model (TBD based on accuracy)

## Rust Integration

The fine-tuned model loads in phonolitui via the **existing** `NeuralClassifier` pattern:

```rust
// crates/core/src/classify/neural_classifier.rs — already works, just swap model files
let config = BertConfig { /* distilbert config */ };
let bert = BertModel::load(vb.pp("distilbert"), &config)?;
let classifier = candle_nn::linear(config.hidden_size, 4, vb.pp("classifier"))?;
// 4-class softmax: [commissive_prob, directive_prob, inform_prob, question_prob]
```

No new dependencies needed. The `candle-core`, `candle-nn`, `candle-transformers`, and `tokenizers` crates are already in `Cargo.toml`.

## Repository Structure

```
phonolitui-sentence-classifier/
├── README.md                 # This file
├── RESEARCH.md               # Full model comparison research
├── DATASETS.md               # Dataset documentation
├── TRAINING_PLAN.md          # Detailed training methodology
├── train.py                  # Fine-tuning script
├── evaluate.py               # Evaluation + edge-case test suite
├── export.py                 # Export to pytorch_model.bin + ONNX
├── augment.py                # ASR edge-case data augmentation
├── config.yaml               # Training hyperparameters
├── requirements.txt          # Python dependencies
├── tests/
│   ├── test_cases.json       # Edge-case test suite (ASR + conversational)
│   └── expected_outputs.json # Ground truth for test cases
└── models/                   # Output directory for trained models
    ├── en/                   # English model
    │   ├── pytorch_model.bin
    │   ├── tokenizer.json
    │   └── config.json
    └── multilingual/         # Multilingual model (Phase 2)
```

## Datasets

| Dataset | HF Path | Languages | Size | Classes | License |
|---------|---------|-----------|------|---------|---------|
| DailyDialog Acts | `eusip/silicone` config `dyda_da` | EN | ~100K utt | 4 (inform, question, directive, commissive) | CC BY-NC-SA 4.0 |
| SwDA | `eusip/silicone` config `swda` | EN | ~200K utt | 41 (collapsible to 5) | LDC |
| MRDA | `eusip/silicone` config `mrda` | EN | ~108K utt | 5 | NIST |
| MIAM | `Bingsu/MIAM` | EN/FR/DE/ES/IT | varies | varies | varies |

## Candle Compatibility

| Architecture | candle-transformers 0.10.2 | Notes |
|--------------|---------------------------|-------|
| BERT | Supported | Already used in phonolitui |
| DistilBERT | Supported | Target architecture |
| RoBERTa | Supported | Alternative base |
| XLM-RoBERTa | Supported | For multilingual |
| BART | Supported | Overkill for this task |

## License

TBD (depends on training data license — DailyDialog is CC BY-NC-SA 4.0)
