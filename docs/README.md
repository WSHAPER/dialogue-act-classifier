# Documentation — dialogue-act-classifier

Fine-tuned DistilBERT models for **4-class dialogue act classification**
(commissive / directive / inform / question), optimized for
**conversational ASR transcripts** at **sub-millisecond GPU latency**.

> **Why this project exists:** No pre-trained model with publicly available
> weights does 4-class dialogue act classification well on
> conversational/ASR data. Five alternatives were evaluated and rejected
> before fine-tuning DistilBERT on DailyDialog. See
> [Models →](models.md) for the full comparison.

---

## Knowledge Base

| Document | What it covers |
|----------|---------------|
| [Architecture](architecture.md) | Component diagram, data flow, how the pieces fit together |
| [Training](training.md) | English & multilingual pipelines, augmentation, hyperparameters, label maps |
| [Optimization](optimization.md) | Inference speed journey (1.5 ms → 0.529 ms), dead ends, final recipe |
| [Models](models.md) | Model zoo (EN / multilingual, FP32 / FP16), deployment formats, Rust integration |
| [Datasets](datasets.md) | DailyDialog, XDailyDialog, Russian corpus, label mappings, known issues |

---

## Quick Orientation

```
dialogue-act-classifier/
├── train.py                  # English fine-tuning
├── train_multilingual.py     # Multilingual (EN+DE+RU) fine-tuning
├── evaluate.py               # DailyDialog test + edge-case suite + baseline comparison
├── evaluate_multilingual.py  # Per-language evaluation
├── export.py                 # PyTorch (candle) + ONNX export + quantization trigger
├── quantize.py               # FP16 / INT8 quantization + TensorRT + IOBinding benchmarks
├── augment.py                # ASR edge-case data augmentation
├── config.yaml               # English hyperparameters
├── config_multilingual.yaml  # Multilingual hyperparameters
├── tests/                    # Edge-case test suites
├── eval_results/             # Accuracy / F1 / latency results
├── hf_cards/                 # HuggingFace model card sources
└── docs/                     # ← you are here
```

### Two models, one interface

| Model | Base | Languages | Size (FP16 ONNX) | Hub |
|-------|------|-----------|-------------------|-----|
| English | `distilbert-base-uncased` | EN | 134 MB | `WSHAPER/dialogue-act-classifier-fp16` |
| Multilingual | `distilbert-base-multilingual-cased` | EN + DE + RU | 271 MB | `WSHAPER/dialogue-act-classifier-fp16-multilingual` |

Both output the same 4-class softmax:
`[commissive_prob, directive_prob, inform_prob, question_prob]`.

### Best latency achieved

| Stage | p50 | Delta |
|-------|-----|-------|
| Baseline CUDA EP | 1.005 ms | — |
| [Session tuning + IOBinding](optimization.md#phase-1-session-tuning--iobinding) | 0.883 ms | −12 % |
| [TensorRT EP](optimization.md#tensorrt-ep) | 0.552 ms | −37 % |
| [TensorRT + IOBinding](optimization.md#tensorrt--iobinding-final) | **0.529 ms** | −4 % |

Zero F1 degradation across all stages (0.7617 → 0.7619).
Full breakdown in [Optimization →](optimization.md).

---

## Getting Started

See [PREREQUISITES.md](../PREREQUISITES.md) for hardware/software setup,
then:

```bash
python train.py                 # English model
python evaluate.py              # DailyDialog + edge cases
python export.py --onnx         # ONNX export
python quantize.py --mode fp16  # FP16 quantization + benchmark
```

For multilingual, swap to `train_multilingual.py` and
`config_multilingual.yaml` — see [Training →](training.md#multilingual-pipeline).
