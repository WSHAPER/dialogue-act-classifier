---
language:
  - en
  - de
  - ru
license: apache-2.0
base_model: distilbert-base-multilingual-cased
tags:
  - onnx
  - fp16
  - text-classification
  - dialogue-act-classification
  - conversational-ai
  - quantized
  - tensorrt
  - multilingual
pipeline_tag: text-classification
---

# Dialogue Act Classifier — Multilingual FP16 ONNX

FP16 ONNX export of a multilingual DistilBERT model fine-tuned for **4-class dialogue act classification** in English, German, and Russian.

For an **English-only variant** using `distilbert-base-uncased` (smaller, 134MB), see [WSHAPER/dialogue-act-classifier-fp16](https://huggingface.co/WSHAPER/dialogue-act-classifier-fp16).

The full PyTorch checkpoint is at [WSHAPER/distilbert-multilingual-dialogue-act-classifier](https://huggingface.co/WSHAPER/distilbert-multilingual-dialogue-act-classifier). The optimization pipeline is on [GitHub](https://github.com/WSHAPER/dialogue-act-classifier).

## Labels

| ID | Label | Description |
|----|-------|-------------|
| 0 | commissive | Promises, commitments ("I'll handle it.") |
| 1 | directive | Commands, requests ("Send the report.") |
| 2 | inform | Statements, facts ("The deadline is Friday.") |
| 3 | question | Questions, inquiries ("What is the timeline?") |

## Performance (NVIDIA RTX A3000, CUDA 12.8)

| Metric | FP32 ONNX | FP16 ONNX |
|--------|-----------|-----------|
| F1-macro | 0.7239 | 0.7247 |
| Accuracy | 80.66% | 80.70% |
| Model size | ~540 MB | 270.8 MB |

### Latency by Execution Provider

| Provider | p50 | p95 | p99 |
|----------|-----|-----|-----|
| CUDAExecutionProvider (ORT_ENABLE_ALL) | 0.907ms | 0.956ms | 1.000ms |
| CUDA EP + IOBinding | 0.899ms | 0.928ms | 0.942ms |
| TensorRT EP (FP16) | 0.556ms | 0.588ms | 0.640ms |
| **TensorRT EP + IOBinding** | **0.528ms** | **0.554ms** | **0.562ms** |

### Batch Throughput (TensorRT EP, dynamic padding)

| Batch Size | Throughput | Per-sample |
|------------|-----------|------------|
| 1 | 178 samples/s | 5.63ms |
| 4 | 495 samples/s | 2.02ms |
| 8 | 513 samples/s | 1.95ms |
| **16** | **524 samples/s** | **1.91ms** |
| 32 | 411 samples/s | 2.43ms |

### Per-Language Accuracy (PyTorch FP32)

| Language | Test Set | Accuracy | F1 Macro |
|----------|----------|----------|----------|
| English | SILICONE dyda_da | 80.8% | 0.725 |
| English | XDailyDialog | 82.5% | 0.750 |
| German | XDailyDialog | 81.8% | 0.738 |
| Russian | xdailydialog-ru | 81.7% | 0.734 |

## Usage

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("WSHAPER/dialogue-act-classifier-fp16-multilingual")
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

text = "Können Sie mir den Bericht schicken?"  # or any EN/DE/RU text
inputs = tokenizer(text, padding="max_length", truncation=True, max_length=48, return_tensors="np")
outputs = session.run(None, {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
})

label_names = ["commissive", "directive", "inform", "question"]
probs = np.exp(outputs[0][0]) / np.exp(outputs[0][0]).sum()
pred = label_names[np.argmax(probs)]
print(f"Prediction: {pred} (confidence: {probs.max():.2f})")
```

## Training Details

- **Base model**: distilbert-base-multilingual-cased
- **Datasets**: XDailyDialog (EN+DE+IT), [WSHAPER/xdailydialog-ru](https://huggingface.co/datasets/WSHAPER/xdailydialog-ru) (Russian, machine-translated)
- **Epochs**: 5, **Batch size**: 32, **LR**: 2e-5
- **Languages**: en, de, ru
- **Quantization**: FP16 via onnxruntime.transformers.float16
- **Inference optimizations**: ORT_ENABLE_ALL, cudnn exhaustive search, TensorRT EP, IOBinding
