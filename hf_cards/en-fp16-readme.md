---
language: en
license: apache-2.0
base_model: distilbert-base-uncased
tags:
  - onnx
  - fp16
  - text-classification
  - dialogue-act-classification
  - conversational-ai
  - quantized
  - tensorrt
pipeline_tag: text-classification
---

# Dialogue Act Classifier — English FP16 ONNX

FP16 ONNX export of a DistilBERT model fine-tuned for **4-class dialogue act classification** on DailyDialog (English).

For a **multilingual variant** (English, German, Russian) using `distilbert-base-multilingual-cased`, see [WSHAPER/dialogue-act-classifier-fp16-multilingual](https://huggingface.co/WSHAPER/dialogue-act-classifier-fp16-multilingual).

The full optimization pipeline (session tuning, TensorRT EP, IOBinding benchmarks) is available on [GitHub](https://github.com/WSHAPER/dialogue-act-classifier).

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
| F1-macro | 0.7617 | 0.7619 |
| Accuracy | 83.36% | 83.39% |
| Model size | 803.9 MB | 134.1 MB |

### Latency by Execution Provider

| Provider | p50 | p95 | p99 |
|----------|-----|-----|-----|
| CUDAExecutionProvider (ORT_ENABLE_ALL) | 0.905ms | 0.989ms | 2.754ms |
| CUDA EP + IOBinding | 0.883ms | 0.920ms | 0.968ms |
| TensorRT EP (FP16) | 0.552ms | 0.582ms | 0.592ms |
| **TensorRT EP + IOBinding** | **0.529ms** | **0.556ms** | **0.561ms** |

### Batch Throughput (TensorRT EP, dynamic padding)

| Batch Size | Throughput | Per-sample |
|------------|-----------|------------|
| 1 | 190 samples/s | 5.27ms |
| 4 | 515 samples/s | 1.94ms |
| **8** | **548 samples/s** | **1.83ms** |
| 16 | 536 samples/s | 1.87ms |

## Usage

### Basic (CUDAExecutionProvider)

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("WSHAPER/dialogue-act-classifier-fp16")
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

text = "Can you send me the report?"
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

### TensorRT + IOBinding (fastest)

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("WSHAPER/dialogue-act-classifier-fp16")

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "model.onnx", sess_options=sess_opts,
    providers=[("TensorrtExecutionProvider", {
        "device_id": 0, "trt_fp16_enable": True,
        "trt_engine_cache_enable": True, "trt_engine_cache_path": "./trt_cache",
    })]
)

text = "Can you send me the report?"
inputs = tokenizer(text, padding="max_length", truncation=True, max_length=48, return_tensors="np")

input_ids_gpu = ort.OrtValue.ortvalue_from_numpy(inputs["input_ids"].astype(np.int64), "cuda", 0)
mask_gpu = ort.OrtValue.ortvalue_from_numpy(inputs["attention_mask"].astype(np.int64), "cuda", 0)

io_binding = session.io_binding()
io_binding.bind_ortvalue_input("input_ids", input_ids_gpu)
io_binding.bind_ortvalue_input("attention_mask", mask_gpu)
io_binding.bind_output("logits", "cuda", 0)

session.run_with_iobinding(io_binding)
logits = io_binding.get_outputs()[0].numpy()

label_names = ["commissive", "directive", "inform", "question"]
pred = label_names[np.argmax(logits[0])]
print(f"Prediction: {pred}")
```

## Training Details

- **Base model**: distilbert-base-uncased
- **Dataset**: DailyDialog ([eusip/silicone](https://huggingface.co/datasets/eusip/silicone), dyda_da config)
- **Epochs**: 4, **Batch size**: 32, **LR**: 2e-5
- **Augmentation**: ASR edge-case augmentation
- **Quantization**: FP16 via onnxruntime.transformers.float16
- **Inference optimizations**: ORT_ENABLE_ALL, cudnn exhaustive search, TensorRT EP, IOBinding
