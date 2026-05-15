# Models

Model zoo, performance metrics, deployment formats, and integration guides.

Related: [Architecture](architecture.md) · [Optimization](optimization.md) ·
[Training](training.md) · [Datasets](datasets.md)

---

## Which Model Should I Use?

| Need | Model | Hub | Size (FP16 ONNX) |
|------|-------|-----|-------------------|
| English only, smallest, fastest | English FP16 | [`WSHAPER/dialogue-act-classifier-fp16`](https://huggingface.co/WSHAPER/dialogue-act-classifier-fp16) | 134 MB |
| English + German + Russian | Multilingual FP16 | [`WSHAPER/dialogue-act-classifier-fp16-multilingual`](https://huggingface.co/WSHAPER/dialogue-act-classifier-fp16-multilingual) | 271 MB |

Both output identical 4-class softmax:
`[commissive_prob, directive_prob, inform_prob, question_prob]`.

The multilingual PyTorch checkpoint (for re-training or fine-tuning) is
at [`WSHAPER/distilbert-multilingual-dialogue-act-classifier`](https://huggingface.co/WSHAPER/distilbert-multilingual-dialogue-act-classifier).

---

## Label Schema

| ID | Label | Description | Examples |
|----|-------|-------------|----------|
| 0 | commissive | Promises, commitments | "I'll handle it.", "We will deliver tomorrow." |
| 1 | directive | Commands, requests | "Send the report.", "Please close the door." |
| 2 | inform | Statements, facts | "The deadline is Friday.", "I think this works." |
| 3 | question | Questions, inquiries | "What is the timeline?", "Can you hear me?" |

---

## English Model Performance

**Base:** `distilbert-base-uncased` (67M params) ·
**Dataset:** DailyDialog via SILICONE `dyda_da`

### Accuracy

| Metric | Value |
|--------|-------|
| Accuracy | 83.4 % |
| F1 macro | 0.7617 |
| F1 weighted | 0.8412 |
| Edge-case suite | 45/45 (100 %) |

### Confusion Matrix (DailyDialog test set, 7,740 samples)

```
              pred:comm  pred:dir  pred:inf  pred:ques
actual:comm       338        76       283        21
actual:dir         56       930       210        82
actual:inf        184       202     3,099        49
actual:ques         7        83        34     2,086
```

`inform` (3,534 samples) dominates the dataset, which pulls weighted
metrics up. `commissive` (718 samples) is the hardest class — often
confused with `inform` because both describe future or factual states.

### Latency (NVIDIA RTX A3000, FP16 ONNX)

| Provider | p50 | p95 | p99 |
|----------|-----|-----|-----|
| CUDA EP (ORT_ENABLE_ALL) | 0.905 ms | 0.989 ms | 2.754 ms |
| CUDA EP + IOBinding | 0.883 ms | 0.920 ms | 0.968 ms |
| TensorRT EP | 0.552 ms | 0.582 ms | 0.592 ms |
| **TensorRT + IOBinding** | **0.529 ms** | **0.556 ms** | **0.561 ms** |

Full optimization journey in [Optimization →](optimization.md).

---

## Multilingual Model Performance

**Base:** `distilbert-base-multilingual-cased` (277M params) ·
**Datasets:** SILICONE + XDailyDialog + xdailydialog-ru

### Per-Language Accuracy (PyTorch FP32)

| Language | Test Set | N | Accuracy | F1 Macro |
|----------|----------|---|----------|----------|
| English | SILICONE dyda_da | 7,740 | 80.8 % | 0.725 |
| English | XDailyDialog | 7,716 | 82.5 % | 0.750 |
| German | XDailyDialog | 7,716 | 81.8 % | 0.738 |
| Russian | xdailydialog-ru | 7,623 | 81.7 % | 0.734 |

### Latency (FP16 ONNX)

| Provider | p50 | p95 | p99 |
|----------|-----|-----|-----|
| CUDA EP + IOBinding | 0.899 ms | 0.928 ms | 0.942 ms |
| **TensorRT + IOBinding** | **0.528 ms** | **0.554 ms** | **0.562 ms** |

The multilingual model is nearly as fast as the English-only model
despite being 2× larger (271 MB vs 134 MB), because the bottleneck is
kernel launch overhead, not compute.

---

## Quantization Comparison

| Format | Size | F1 macro | GPU Supported | Verdict |
|--------|------|----------|---------------|---------|
| FP32 ONNX | 804 MB | 0.7617 | ✅ | Baseline |
| **FP16 ONNX** | **134 MB** | **0.7619** | **✅ Full CUDA** | **Production** |
| INT8 dynamic | ~100 MB | — | ❌ 96 Memcpy fallback | Broken |
| INT8 static | ~100 MB | Collapsed | ❌ Weight corruption | Broken |

FP16 is the only viable quantization path for CUDA inference. INT8
ops are unsupported by CUDAExecutionProvider. See
[Optimization → Quantization](optimization.md#stage-1-model-quantization)
for the full analysis.

---

## Deployment Formats

### PyTorch (candle-transformers, Rust, CPU)

Exported by [`export.py`](../export.py):

```
candle-export/
├── pytorch_model.bin    # Model weights
├── config.json          # DistilBertConfig (candle-compatible)
├── tokenizer.json       # Fast tokenizer
├── label_map.json       # {0: "commissive", ...}
└── model_meta.json      # Metadata
```

The [`CANDLE_CONFIG`](../export.py) dict in `export.py` ensures the
`config.json` is compatible with `candle-transformers`' DistilBERT
loader. Key params: `dim=768`, `n_layers=6`, `n_heads=12`,
`hidden_dim=3072`, `vocab_size=30522`.

### ONNX (ONNX Runtime, Python/C++/Rust, GPU)

Exported by [`export.py --onnx`](../export.py):

```
onnx-export/
└── model.onnx           # FP32 ONNX with dynamic batch axis
```

Then quantized by [`quantize.py`](../quantize.py):

```
quantized/
├── fp32/model.onnx      # FP32 baseline
└── fp16/model.onnx      # FP16 production model (134 MB)
```

The ONNX model has dynamic batch axes: `{0: "batch_size"}` on all
input and output tensors.

---

## Rust Integration (candle-transformers)

The fine-tuned model loads via `candle-transformers` with a standard
DistilBERT + linear classifier pattern:

```rust
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

let config = BertConfig {
    vocab_size: 30522,
    hidden_size: 768,
    num_hidden_layers: 6,
    num_attention_heads: 12,
    intermediate_size: 3072,
    ..Default::default()
};

let bert = BertModel::load(vb.pp("distilbert"), &config)?;
let classifier = candle_nn::linear(config.hidden_size, 4, vb.pp("classifier"))?;

// Forward pass → 4-class softmax
// Output: [commissive_prob, directive_prob, inform_prob, question_prob]
```

**Why Rust + candle instead of ONNX Runtime:**
- No `ort` / ONNX Runtime dependency — no CUDA context conflict with
  Whisper's GPU context.
- Pure CPU inference — the model is small enough (~250 MB) that CPU
  inference at 5–15 ms is acceptable for non-real-time use.
- `candle-transformers` 0.10.2 supports DistilBERT natively — it's
  treated as a thin wrapper around BERT.

For GPU inference at sub-millisecond latency, use the ONNX Runtime path
with TensorRT EP — see [Optimization](optimization.md#stage-3-tensorrt-execution-provider).

---

## Usage Examples

### Python: CUDA EP (simplest)

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

labels = ["commissive", "directive", "inform", "question"]
probs = np.exp(outputs[0][0]) / np.exp(outputs[0][0]).sum()
print(f"{labels[np.argmax(probs)]} ({probs.max():.2f})")
# → question (0.97)
```

### Python: TensorRT + IOBinding (fastest)

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

# Pre-allocate GPU tensors
inputs = tokenizer("Can you send me the report?", padding="max_length",
                    truncation=True, max_length=48, return_tensors="np")
input_ids_gpu = ort.OrtValue.ortvalue_from_numpy(
    inputs["input_ids"].astype(np.int64), "cuda", 0)
mask_gpu = ort.OrtValue.ortvalue_from_numpy(
    inputs["attention_mask"].astype(np.int64), "cuda", 0)

io_binding = session.io_binding()
io_binding.bind_ortvalue_input("input_ids", input_ids_gpu)
io_binding.bind_ortvalue_input("attention_mask", mask_gpu)
io_binding.bind_output("logits", "cuda", 0)

session.run_with_iobinding(io_binding)
logits = io_binding.get_outputs()[0].numpy()

labels = ["commissive", "directive", "inform", "question"]
print(f"{labels[np.argmax(logits[0])]}")
# → question
```

### PyTorch (transformers)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "WSHAPER/dialogue-act-classifier-fp16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("I'll handle the deployment.", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

labels = ["commissive", "directive", "inform", "question"]
print(labels[logits.argmax(-1).item()])
# → commissive
```

---

## Baseline Comparison

The model was built to replace TigreGotico's TF-IDF classifier, which
failed on conversational/ASR edge cases:

| Edge Case | TigreGotico | This Model |
|-----------|-------------|------------|
| "Can you hear me?" | `request` ❌ | `question` ✅ |
| "Do something for me." | `polar_question` ❌ | `directive` ✅ |
| "Is this working?" | `statement` ❌ | `question` ✅ |
| "I think this is the right approach." | `polar_question` ❌ | `inform` ✅ |
| "uh what is the timeline" | `statement` ❌ | `question` ✅ |

TigreGotico's TF-IDF features cannot capture pragmatic intent — "Can you"
is syntactically a request pattern but pragmatically a genuine question
in conversational context. The fine-tuned DistilBERT handles this
correctly.

Full research comparison in [`RESEARCH.md`](../RESEARCH.md).
