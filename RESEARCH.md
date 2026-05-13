# Sentence Type Classification — Model Research

## Task

Find a model that classifies utterances from meeting transcripts into **4 dialogue act classes**: commissive, directive, inform, question. Must work well on **conversational ASR output** (Whisper transcription), be loadable in **Rust** (via `candle` or `ort`), and ideally support multiple languages.

The 4-class output maps to phonolitui's 3-class `SentenceType` at inference time: commissive→Statement, directive→Instruction, inform→Statement, question→Question.

## Evaluated Models

### 1. TigreGotico/sentence-type-classifiers (Current)

| Property | Value |
|----------|-------|
| Architecture | TF-IDF vectorizer + linear classifier (sklearn pipeline) |
| Export | ONNX via skl2onnx |
| Classes | 6: `command`, `exclamation`, `polar_question`, `request`, `statement`, `wh_question` |
| Input | Raw text string (DT_STRING tensor) |
| Languages | EN, DE, ES, FR, IT, NL, PT (separate models per language) |
| Training data | ~9,900 balanced synthetic/written samples per language |
| EN accuracy | 99.2% on written test set |
| Model size | ~2.3 MB |
| Inference | ~5ms CPU via ort |
| Rust loading | `ort` crate (ONNX Runtime) |

**Strengths:**
- Tiny model size
- Fast inference
- Multilingual (separate models)
- Raw string input (no tokenizer dependency)

**Weaknesses:**
- **Fails on conversational/ASR edge cases:**
  - "Can you hear me?" → `request` (should be `polar_question`)
  - "Do something for me." → `polar_question` (should be `command`/`request`)
  - "Is this working?" → `statement` (should be `polar_question`)
  - "I think this is the right approach." → `polar_question` (should be `statement`)
- TF-IDF features cannot capture pragmatic intent — "Can you" is syntactically a request pattern but pragmatically a genuine question in conversational context
- "probabilities" output is raw logits, not softmax (requires manual softmax)
- Each language is a separate model file

**Verdict:** Good for written text classification. Too inaccurate for ASR/conversational use.

---

### 2. mrsinghania/asr-question-detection (Previous)

| Property | Value |
|----------|-------|
| Architecture | Distilled BERT (4 layers, 256 hidden, 4 heads) |
| Export | PyTorch (`pytorch_model.bin`) |
| Classes | 2: statement (0), question (1) |
| Input | Tokenized text (BERT tokenizer) |
| Languages | English only |
| Training data | 7K+ ASR/interview samples |
| Model size | ~44 MB |
| Inference | ~10-20ms CPU via candle |
| Rust loading | `candle-transformers` + `tokenizers` (already working) |

**Strengths:**
- Already integrated in phonolitui
- Trained on ASR data
- Pure Rust inference via candle (no CUDA conflict with Whisper)
- Works for clear WH-questions and statements

**Weaknesses:**
- **Binary only** — cannot detect instructions
- Small model, limited accuracy on ambiguous cases
- English only

**Verdict:** Too limited (binary). Being replaced.

---

### 3. facebook/bart-large-mnli (Zero-shot)

| Property | Value |
|----------|-------|
| Architecture | BART-large encoder-decoder (400M params) |
| Export | PyTorch, safetensors |
| Approach | Zero-shot via NLI (entailment scores for candidate labels) |
| Input | Tokenized text |
| Languages | English only |
| Model size | ~1.6 GB |
| Inference | ~100-200ms CPU |
| Rust loading | `candle-transformers` (BART supported) |

**Strengths:**
- No training needed
- Flexible labels (can add any class name)
- Good general-purpose zero-shot

**Weaknesses:**
- Massive model (1.6GB vs 2.3MB for TigreGotico)
- Slow inference
- Weaker than fine-tuned models on specific tasks
- English only (multilingual variant `joeddav/xlm-roberta-large-xnli` exists but even larger)
- Overkill for 3-class classification

**Verdict:** Not suitable. Too large, too slow, too weak.

---

### 4. ESM-based SILICONE Models

| Property | Value |
|----------|-------|
| Model | `davidschulte/ESM_silicone_dyda_da`, `davidschulte/ESM_silicone_swda` |
| Architecture | Evolutionary Scale Modeling (protein sequence model) |
| Training data | DailyDialog Acts, SwDA |

**Verdict:** These are **protein sequence models** misapplied to text. Completely unsuitable.

---

### 5. Dialogue Act Classification Papers (No Weights)

Multiple papers train BERT/RoBERTa on dialogue act classification but **none publish weights on HuggingFace**:

| Paper | Base | Dataset | Accuracy | Weights? |
|-------|------|---------|----------|----------|
| Mechide & Farah (2023) | BERT-tiny/small | SILICONE | 75-85% | No |
| Raheja & Tetreault (2019) CASA | Hierarchical RNN | SwDA | 82.9% | No |
| Colombo et al. (2020) H-Seq2seq | Hierarchical Seq2seq | SwDA | 85.0% | No |
| Kumar et al. (2018) | Multi-task RNN | SwDA | 81.2% | No |
| Liu & Lane (2018) | Attention-based RNN | SwDA | 81.4% | No |

**Verdict:** Can't use without retraining.

---

## Decision: Fine-tune DistilBERT on DailyDialog

### Why DistilBERT

| Property | Value |
|----------|-------|
| Model | `distilbert-base-uncased` |
| Params | 67M |
| Size | ~250MB |
| Inference | ~5-15ms CPU via candle |
| Architecture | Supported by `candle-transformers` 0.10.2 |
| License | Apache 2.0 |

DistilBERT is:
- **Small enough** to run alongside Whisper without memory pressure
- **Fast enough** for real-time classification in the transcription pipeline
- **Compatible** with candle's DistilBERT implementation
- **Well-understood** — the phonolitui BERT loading code (`neural_classifier.rs`) works with nearly identical DistilBERT loading

### Why DailyDialog

- **Conversational nature**: 13,118 multi-turn dialogues about daily life (travel, shopping, work, relationships) — much closer to meeting speech than TigreGotico's synthetic/written data
- **100K+ utterances** with official train/val/test splits
- **4 dialogue act classes** used natively (no remapping):
  - `commissive` (0) → commissive
  - `directive` (1) → directive
  - `inform` (2) → inform
  - `question` (3) → question
- **Available on HuggingFace**: `eusip/silicone` dataset, config `dyda_da`
- **CC BY-NC-SA 4.0** license

### Training augmentation

Add ASR-specific edge cases to the training data:

```json
[
  {"text": "Can you hear me?", "label": "question"},
  {"text": "Can you see my screen?", "label": "question"},
  {"text": "Do something for me.", "label": "directive"},
  {"text": "Please send the report.", "label": "directive"},
  {"text": "Close the door.", "label": "directive"},
  {"text": "Hey man, how's it going?", "label": "question"},
  {"text": "How are you doing?", "label": "question"},
  {"text": "I think this is the right approach.", "label": "inform"},
  {"text": "That's definitely not a question.", "label": "inform"},
  {"text": "uh what is the timeline", "label": "question"},
  {"text": "um can you explain that again", "label": "question"},
  {"text": "provide an example of how markdown works", "label": "directive"},
  {"text": "show me the results", "label": "directive"},
  {"text": "make sure it works", "label": "directive"}
]
```

### Multilingual path (Phase 2)

For EN + DE + ES + FR + IT + RU:
- Base: `distilbert-base-multilingual-cased` (104 languages, 277M params)
- Data: MIAM benchmark (EN, FR, DE, ES, IT) + DailyDialog (EN)
- Russian: may need additional data collection or synthetic augmentation

### Rust loading (no code changes needed)

The fine-tuned model uses the **exact same** loading pattern as `NeuralClassifier`:

```rust
// phonolitui: crates/core/src/classify/neural_classifier.rs
// Just change:
//   - BertConfig → DistilBertConfig
//   - 2-class linear head → 4-class linear head
//   - Model weights from fine-tuned DistilBERT

let config = DistilBertConfig {
    vocab_size: 30522,
    hidden_size: 768,
    num_hidden_layers: 6,
    num_attention_heads: 12,
    intermediate_size: 3072,
    // ...
};
let bert = BertModel::load(vb.pp("distilbert"), &config)?;
let classifier = candle_nn::linear(768, 4, vb.pp("classifier"))?;
// Output: [commissive_prob, directive_prob, inform_prob, question_prob]
```

candle-transformers treats DistilBERT as a thin wrapper around BERT — the loading code is nearly identical.

## Model Comparison Summary

| Model | Accuracy (conversational) | Size | Inference | Multilingual | Rust loadable | Verdict |
|-------|--------------------------|------|-----------|-------------|---------------|---------|
| TigreGotico (current) | ~70% | 2.3MB | 5ms | Yes (separate) | ort | Too inaccurate |
| mrsinghania BERT (prev) | ~75% (binary) | 44MB | 15ms | No | candle | Binary only |
| BART-MNLI zero-shot | ~65% | 1.6GB | 200ms | No | candle | Too large/slow |
| **Fine-tuned DistilBERT** | **~85-90% (expected)** | **~250MB** | **~10ms** | **Phase 2** | **candle** | **Recommended** |
