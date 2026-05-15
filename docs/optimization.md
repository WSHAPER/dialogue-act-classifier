# Inference Optimization

The full journey from **1.5 ms → 0.529 ms** single-sample p50 latency
on an NVIDIA RTX A3000, with **zero F1 degradation** (0.7617 → 0.7619).

Related: [Architecture](architecture.md) · [Models](models.md) ·
[Training](training.md)

---

## Motivation

The dialogue act classifier runs alongside Whisper in a real-time
meeting transcription pipeline. Every millisecond of classifier latency
directly adds to end-to-end transcript processing time. The target was
sub-millisecond inference on GPU without sacrificing accuracy.

---

## Results Summary

| Stage | p50 | p95 | p99 | F1 macro | Speedup | Commit |
|-------|-----|-----|-----|----------|---------|--------|
| Baseline CUDA EP (FP32) | ~1.5 ms | — | — | 0.7617 | 1.0× | — |
| FP16 ONNX conversion | 1.005 ms | — | — | 0.7617 | 1.5× | `e3f7c18` |
| Phase 1: session + IOBinding + seq length | 0.883 ms | 0.920 ms | 0.968 ms | 0.7619 | 1.7× | `886e0a3` |
| TensorRT EP | 0.552 ms | 0.582 ms | 0.592 ms | — | 2.7× | `1a66afa` |
| **TensorRT + IOBinding** | **0.529 ms** | **0.556 ms** | **0.561 ms** | — | **2.8×** | `eb87d9e` |

Cumulative: **1.9× speedup** from the FP16 baseline, **3× from FP32**.

Batch throughput peaks at **548 samples/s** at batch size 8 — see
[Batch Inference](#batch-inference).

---

## Stage 1: Model Quantization

### What was tried

| Method | Size | Accuracy | Speed | GPU? | Verdict |
|--------|------|----------|-------|------|---------|
| **FP16** (`onnxruntime.float16`) | 804→134 MB (6×) | F1 0.7617→0.7617 | 1.53× faster | ✅ Full CUDA | **Winner** |
| INT8 dynamic (ORT) | ~100 MB | OK | 7× **slower** | ❌ 96 Memcpy fallback nodes | Dead end |
| INT8 static (ORT) | ~100 MB | Collapsed to majority class | — | ❌ LayerNorm weight corruption | Dead end |

**Why INT8 failed on GPU:** The CUDAExecutionProvider doesn't support
most INT8 operators — they fall back to CPU via Memcpy nodes. INT8
static quantization additionally corrupted LayerNorm weights, causing the
model to predict only the majority class.

**Why FP16 works:** All FP16 ops are natively supported by CUDA EP.
Zero accuracy loss (F1 unchanged at 4 decimal places). Real speedup
because FP16 matmuls use Tensor Cores on Ampere GPUs.

### Implementation

[`quantize.py`](../quantize.py) → [`optimize_graph()`](../quantize.py):
Uses `onnxruntime.transformers.float16.convert_float_to_float16()` with
`keep_io_types=True` (keeps input/output tensors as FP32 for
compatibility, converts internal weights to FP16).

Graph optimization is intentionally deferred to session level
(ORT_ENABLE_ALL) rather than applied via ONNX graph passes, because
the ONNX-level optimizer creates CPU-only fused nodes that block CUDA
Graphs partitioning — see [Dead Ends](#dead-ends).

---

## Stage 2: Phase 1 Optimizations

Implemented in commit `886e0a3` — four changes targeting different
overhead sources at sub-millisecond latency:

### Session Options

```python
sess_opts.graph_optimization_level = ORT_ENABLE_ALL
sess_opts.enable_mem_pattern = True
sess_opts.enable_mem_reuse = True
sess_opts.execution_mode = ORT_SEQUENTIAL
```

With `cudnn_conv_algo_search=EXHAUSTIVE` in the CUDA provider options,
cuDNN searches all algorithms including Tensor Core variants for FP16
matmuls. The default (`DEFAULT`) skips the expensive search and misses
faster kernel variants.

### Sequence Length Reduction (128 → 48)

DailyDialog utterances average ~12 tokens (p95 = 35). Padding to 128
wastes ~75% of compute per inference. `inference_max_length=48` is
decoupled from `max_seq_length=128` in [`config.yaml`](../config.yaml)
so training stability is unaffected.

This alone saved significant compute but is hard to isolate in
benchmarks because other changes were applied simultaneously.

### IOBinding

Standard `session.run()` copies numpy arrays to GPU on every call.
IOBinding pre-allocates `OrtValues` on the GPU and reuses them:

```python
input_ids_gpu = ort.OrtValue.ortvalue_from_numpy(input_ids, "cuda", 0)
mask_gpu = ort.OrtValue.ortvalue_from_numpy(attention_mask, "cuda", 0)

io_binding = session.io_binding()
io_binding.bind_ortvalue_input("input_ids", input_ids_gpu)
io_binding.bind_ortvalue_input("attention_mask", mask_gpu)
io_binding.bind_output("logits", "cuda", 0)

session.run_with_iobinding(io_binding)
```

For repeated inference (e.g., in a serving loop), `update_inplace()`
writes new tokenized inputs directly into the existing GPU memory —
zero-copy, no numpy→GPU memcpy per call.

**Impact:** p99 dropped from 3.092 ms → 0.968 ms (3.2× tighter tail
latency). The p99 improvement is the real win — it eliminates sporadic
slow inferences caused by memory allocation and copy overhead.

---

## Stage 3: TensorRT Execution Provider

Implemented in commits `1a66afa` and `fc844be`.

### Why TensorRT

Phase 1 plateaued at 0.883 ms p50. The ONNX graph has 6 inherent
**Memcpy nodes** (CPU↔GPU copies) from `Shape` and `Gather` operators
needed for dynamic axes. These Memcpy nodes block CUDA Graphs entirely
(which requires 100% CUDA partitioning).

TensorRT compiles the ONNX graph into a fully GPU-partitioned engine
that handles `Shape`/`Gather` ops natively, enabling:

- **Kernel fusion** — fuses attention, LayerNorm, and feed-forward
  subgraphs into single GPU kernels.
- **Kernel auto-tuning** — selects optimal algorithms for the specific
  GPU architecture (Ampere in this case).
- **FP16 Tensor Core utilization** — dedicated hardware acceleration.

### Setup

```bash
pip install tensorrt  # 10.16
```

TensorRT engines are compiled on first inference and cached to disk
(`trt_engine_cache_path`). Subsequent loads are near-instant.

### Impact

0.883 ms → 0.552 ms p50 (37% faster). Cumulative from FP16 baseline:
1.9× speedup.

---

## Stage 4: TensorRT + IOBinding (Final)

Commit `eb87d9e`.

Combines TensorRT EP's GPU-partitioned engine with IOBinding's
GPU-resident tensors. The combination keeps all data on GPU throughout
the entire inference lifecycle — tokenized inputs are written directly
into GPU memory via `update_inplace()`, TensorRT runs the fused engine,
and outputs stay on GPU.

**Result:** 0.552 ms → 0.529 ms p50 (4% additional gain). p99 = 0.561 ms.

This is the **current production configuration**. See
[Models → Usage](models.md#usage-examples) for the full code.

---

## Batch Inference

Commit `600e951` — [`benchmark_batch()`](../quantize.py).

Measures throughput with dynamic padding (pad to longest in batch, not
`max_length`) across batch sizes 1–64 on TensorRT EP:

| Batch Size | Throughput | Per-sample | Accuracy | F1 |
|------------|-----------|------------|----------|-----|
| 1 | 190 s/s | 5.27 ms | — | — |
| 4 | 515 s/s | 1.94 ms | — | — |
| **8** | **548 s/s** | **1.83 ms** | 0.833 | 0.762 |
| 16 | 536 s/s | 1.87 ms | — | — |
| 32 | ~460 s/s | ~2.2 ms | — | — |

**Batch size 8 is the sweet spot.** Beyond that, tokenization and
padding overhead grows faster than GPU parallelism for this small model
(134 MB, 67M params).

Note: Per-sample latency (1.83 ms) includes the full pipeline
(tokenize + CPU→GPU transfer + inference), vs 0.529 ms which isolates
pure GPU inference. The gap is the tokenization and data transfer
overhead.

---

## Dead Ends

### `ort.transformers.optimizer` (ONNX graph-level fusion)

Attempted attention and LayerNorm fusion via the ONNX graph optimizer.
**Result:** Created CPU-only `SkipLayerNormalization` nodes → 12 Memcpy
nodes (vs 6 baseline). Also blocks CUDA Graphs entirely. Reverted to
session-level optimization only (`ORT_ENABLE_ALL`).

### CUDA Graphs

Blocked by the 6 inherent Memcpy nodes from `Shape`/`Gather` ops for
dynamic axes. CUDA Graphs requires 100% CUDA partitioning. TensorRT EP
is the correct path to eliminate these nodes.

### INT8 Quantization

Both static and dynamic INT8 via ONNX Runtime fail on GPU:

- **Static:** Corrupts LayerNorm weights → model collapses to majority
  class prediction.
- **Dynamic:** Adds 96 Memcpy nodes → 7× slowdown. INT8 ops aren't
  supported by CUDAExecutionProvider and fall back to CPU.

FP16 is the correct quantization path for CUDA inference. See
[Models → Quantization Comparison](models.md#quantization-comparison).

---

## Reproducing the Benchmarks

```bash
# FP16 quantization + basic benchmark
python quantize.py --mode fp16

# Add IOBinding benchmark
python quantize.py --mode fp16 --iobinding

# Add TensorRT EP benchmark
python quantize.py --mode fp16 --tensorrt

# Full stack: TensorRT + IOBinding
python quantize.py --mode fp16 --trt-iobinding

# Batch throughput
python quantize.py --mode fp16 --batch

# All benchmarks at once
python quantize.py --mode fp16 --iobinding --tensorrt --trt-iobinding --batch
```

Results are saved as JSON in the model's `quantized/` directory:
- `benchmark_results.json` — FP32 vs FP16 latency
- `iobinding_results.json` — IOBinding p50/p95/p99
- `tensorrt_results.json` — TensorRT EP latency
- `trt_iobinding_results.json` — TensorRT + IOBinding (fastest)
- `batch_results.json` — Throughput by batch size
