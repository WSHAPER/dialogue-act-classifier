# Architecture

How the components fit together — from raw dataset to sub-millisecond
GPU inference.

Related: [Training](training.md) · [Optimization](optimization.md) ·
[Models](models.md) · [Datasets](datasets.md)

---

## Overview

```
                         ┌──────────────────────────────────────┐
  Raw Data               │  Training Pipeline                   │
  ─────────              │                                      │
  DailyDialog (HF) ──────┤  augment.py ──┐                      │
  XDailyDialog (Git) ────┤              ├─► train.py            │
  xdailydialog-ru (HF) ──┤  tests/*.json─┘    │                 │
                         │                   models/en/final    │
                         │                   models/multi/final │
                         └──────────┬───────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────────────────────┐
                         │  Export & Quantize                   │
                         │                                      │
                         │  export.py ───► pytorch_model.bin    │  ← candle (Rust)
                         │             ──► model.onnx (FP32)    │
                         │                                      │
                         │  quantize.py ─► model.onnx (FP16)   │  ← ONNX Runtime
                         └──────────┬───────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────────────────────┐
                         │  Serving / Inference                  │
                         │                                      │
                         │  CUDA EP ────────── 0.883 ms p50    │
                         │  TensorRT EP ─────── 0.552 ms p50    │
                         │  TensorRT + IOBind ─ 0.529 ms p50   │
                         │                                      │
                         │  candle-transformers (Rust, CPU)     │
                         │  onnxruntime (Python, GPU)           │
                         └──────────────────────────────────────┘
```

---

## Component Reference

### Training Layer

| File | Role | Config | Output |
|------|------|--------|--------|
| [`augment.py`](../augment.py) | Generates ASR-like variants from edge-case test cases | — | HuggingFace `Dataset` |
| [`train.py`](../train.py) | English fine-tuning on DailyDialog + augmentation | [`config.yaml`](../config.yaml) | `models/en/final/` |
| [`train_multilingual.py`](../train_multilingual.py) | Multilingual fine-tuning on SILICONE + XDailyDialog + Russian | [`config_multilingual.yaml`](../config_multilingual.yaml) | `models/multilingual/final/` |

`train.py` calls [`build_augmented_dataset()`](../augment.py) from
`augment.py` to expand the edge-case suite into thousands of ASR-variant
training samples (fillers, casing, punctuation stripping). This is
controlled by `--augment-factor` (default 5×).

The two training scripts share the same `Trainer`-based loop and
[`compute_metrics()`](../train.py) function (accuracy, F1 macro/weighted)
but differ in dataset loading:

- **English**: [`load_dailydialog()`](../train.py) → SILICONE `dyda_da`
  split, columns `Utterance` → `text`, `Label` → `label`.
- **Multilingual**: [`load_silicone_dyda()`](../train_multilingual.py) +
  [`load_xdailydialog()`](../train_multilingual.py) + optional
  [`translate_to_russian()`](../train_multilingual.py) merged with
  `concatenate_datasets()`.

See [Training](training.md) for hyperparameters and dataset details.

### Evaluation Layer

| File | What it measures | See |
|------|-----------------|-----|
| [`evaluate.py`](../evaluate.py) | DailyDialog test set, edge-case suite, TigreGotico baseline | [Models → Performance](models.md#english-model-performance) |
| [`evaluate_multilingual.py`](../evaluate_multilingual.py) | Per-language accuracy on EN/DE/RU | [Models → Multilingual](models.md#multilingual-model-performance) |

Both produce JSON results in [`eval_results/`](../eval_results/).

`evaluate.py` also has a
[`benchmark_quantized_model()`](../evaluate.py) mode (`--quantized-model`)
that measures ONNX latency percentiles (p50/p95/p99) on the FP16 model.

### Export & Quantization Layer

| File | Formats | Details |
|------|---------|---------|
| [`export.py`](../export.py) | `pytorch_model.bin` + `config.json` + `tokenizer.json` | candle-compatible DistilBertConfig |
| [`export.py --onnx`](../export.py) | `model.onnx` (FP32, dynamic batch axis) | via `optimum` or manual `torch.onnx.export` |
| [`quantize.py`](../quantize.py) | FP16 ONNX, INT8 static, INT8 dynamic | FP16 is the production path |
| [`quantize.py --tensorrt --iobinding`](../quantize.py) | TensorRT EP + IOBinding benchmarks | Full optimization stack |

`export.py` generates a candle-specific `config.json` (see
[`CANDLE_CONFIG`](../export.py)) that maps DistilBERT parameters to the
format expected by `candle-transformers`. The quantization pipeline in
`quantize.py` chains through `export_fp32_onnx()` →
`optimize_graph()` (FP16 conversion) → benchmark stages.

See [Optimization](optimization.md) for the full speed journey and
[Models](models.md) for deployment formats.

---

## Data Flow: Inference

### Python (ONNX Runtime, GPU)

```
Input text
    │
    ▼
AutoTokenizer.from_pretrained(model_path)
    │  padding="max_length", max_length=48, truncation=True
    ▼
{input_ids: int64[1,48], attention_mask: int64[1,48]}
    │
    ▼  (IOBinding: OrtValues resident on GPU)
OrtValue.ortvalue_from_numpy(..., "cuda", 0)
    │
    ▼
session.run_with_iobinding(io_binding)
    │  TensorRT EP, FP16 engine cached on disk
    ▼
logits[1, 4]  →  softmax  →  argmax  →  label
```

Key parameters:
- `max_length=48` (not 128) — covers p95 of DailyDialog utterances,
  decoupled from training `max_seq_length=128` via
  [`config.yaml: inference_max_length`](../config.yaml).
- `ORT_ENABLE_ALL` + `cudnn_conv_algo_search=EXHAUSTIVE` — session-level
  graph optimization, see [Optimization](optimization.md#session-options).
- IOBinding keeps `OrtValues` on GPU, uses `update_inplace()` for
  zero-copy input writes — see
  [Optimization → IOBinding](optimization.md#iobinding).

### Rust (candle-transformers, CPU)

```
Input text
    │
    ▼
tokenizers::Tokenizer::from_file("tokenizer.json")
    │
    ▼
candle_transformers::models::bert::BertModel::load()
    │  + candle_nn::linear(768, 4, "classifier")
    ▼
[commissive_prob, directive_prob, inform_prob, question_prob]
```

No ONNX Runtime dependency — pure Rust, no CUDA context conflict with
Whisper. See [Models → Rust Integration](models.md#rust-integration-candle-transformers).

---

## Configuration Files

| File | Used by | Key fields |
|------|---------|------------|
| [`config.yaml`](../config.yaml) | `train.py`, `quantize.py` | `base_model`, `max_seq_length`, `inference_max_length`, `label_map`, dataset path |
| [`config_multilingual.yaml`](../config_multilingual.yaml) | `train_multilingual.py` | `xdailydialog` repo/languages, `russian` translation model, `act_label_map` |
| [`tests/test_cases.json`](../tests/test_cases.json) | `evaluate.py`, `augment.py` | 52 edge-case samples with `text`, `expected`, `category` |
| [`tests/test_cases_multilingual.json`](../tests/test_cases_multilingual.json) | `evaluate_multilingual.py` | Multilingual edge cases |

The `act_label_map` in the multilingual config maps XDailyDialog's
1-based labels to the project's 0-based indices — see
[Datasets → Label Mappings](datasets.md#label-mappings).
