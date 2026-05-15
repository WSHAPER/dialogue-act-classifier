"""
Post-training quantization for the DistilBERT dialogue-act classifier.

Supports three modes:
  - fp16:    FP16 conversion (best for GPU inference — 2x size reduction, real speedup)
  - dynamic: weight-only INT8 quantization (works on CPU and GPU)
  - static:  INT8 static quantization with calibration (CPU only, highest compression)

Usage:
    python quantize.py                                    # FP16 conversion (default)
    python quantize.py --mode dynamic                     # INT8 dynamic quantization
    python quantize.py --mode static                      # INT8 static quantization (CPU)
    python quantize.py --model-path models/en/final
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

LABEL_NAMES = ["commissive", "directive", "inform", "question"]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)





def export_fp32_onnx(model_path: str, output_dir: str):
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Exporting FP32 ONNX model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    onnx_file = out / "model.onnx"
    if onnx_file.exists():
        print(f"  Saved model.onnx ({onnx_file.stat().st_size / 1e6:.1f} MB)")
    else:
        onnx_files = list(out.glob("*.onnx"))
        for f in onnx_files:
            print(f"  Saved {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    return out


def optimize_graph(fp32_onnx_path: str, output_path: str, model_path: str = None) -> bool:
    """Convert FP32 ONNX to FP16. Graph optimization is handled at session
    level via ORT_ENABLE_ALL, which avoids creating CPU-only fused nodes that
    block CUDA Graphs partitioning."""
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import onnx

    print("Converting FP32 ONNX to FP16...")
    model = onnx.load(str(fp32_onnx_path))
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(output_path))
    print("  FP16 conversion complete (graph optimization deferred to ORT_ENABLE_ALL at session level)")
    return True


def _create_optimized_session(onnx_path, use_gpu=True):
    """Create ONNX Runtime session with full optimization: ORT_ENABLE_ALL,
    memory patterns, cudnn exhaustive search for Tensor Core FP16."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_mem_reuse = True
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.optimized_model_filepath = str(onnx_path).replace(".onnx", "-ort-cache.onnx")

    providers = []
    if use_gpu:
        providers.append((
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ))
    providers.append("CPUExecutionProvider")

    return ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)


def _load_inference_max_length() -> int:
    cfg = load_config()
    return cfg.get("inference_max_length", cfg.get("max_seq_length", 48))


def _calibrationDataReader(tokenizer, max_length, num_samples, seed):
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset("eusip/silicone", "dyda_da", split="train", trust_remote_code=True)
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    subset = ds.select(indices)
    texts = [ex["Utterance"] for ex in subset]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )

    class _Reader:
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.index = 0

        def get_next(self):
            if self.index >= self.input_ids.shape[0]:
                return None
            feed = {
                "input_ids": self.input_ids[self.index : self.index + 1],
                "attention_mask": self.attention_mask[self.index : self.index + 1],
            }
            self.index += 1
            return feed

        def rewind(self):
            self.index = 0

    return _Reader(encoded["input_ids"], encoded["attention_mask"])


def quantize_model(
    model_path: str,
    output_dir: str,
    mode: str,
    calibration_samples: int,
    seed: int,
):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from transformers import AutoTokenizer

    cfg = load_config()
    max_length = cfg["max_seq_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    fp32_dir = Path(output_dir) / "fp32"
    onnx_file = fp32_dir / "model.onnx"
    if not onnx_file.exists():
        candidates = list(fp32_dir.glob("*.onnx")) if fp32_dir.exists() else []
        if candidates:
            onnx_file = candidates[0]
        else:
            fp32_path = export_fp32_onnx(model_path, str(fp32_dir))
            onnx_file = fp32_path / "model.onnx"
            if not onnx_file.exists():
                candidates = list(fp32_path.glob("*.onnx"))
                onnx_file = candidates[0] if candidates else None
    else:
        fp32_path = fp32_dir
        print(f"Reusing existing FP32 ONNX: {onnx_file}")

    if not onnx_file or not onnx_file.exists():
        raise FileNotFoundError(f"No ONNX file found in {fp32_dir}")

    quant_dir_name = {"static": "int8_static", "dynamic": "int8_dynamic", "fp16": "fp16"}[mode]
    quant_dir = Path(output_dir) / quant_dir_name
    quant_dir.mkdir(parents=True, exist_ok=True)
    quantized_path = quant_dir / "model.onnx"

    if mode == "fp16":
        optimize_graph(str(onnx_file), str(quantized_path), model_path=model_path)
    elif mode == "static":
        from onnxruntime.quantization import quantize_static, QuantFormat
        from onnxruntime.quantization.shape_inference import quant_pre_process

        preprocessed = fp32_dir / "model_preprocessed.onnx"
        if not preprocessed.exists():
            print("Pre-processing ONNX model...")
            import onnx as _onnx
            model = _onnx.load(str(onnx_file))
            model = _onnx.shape_inference.infer_shapes(model)
            preprocessed_tmp = fp32_dir / "model_inferred.onnx"
            _onnx.save(model, str(preprocessed_tmp))
            try:
                quant_pre_process(str(preprocessed_tmp), str(preprocessed), auto_merge=True)
            except Exception:
                import shutil
                shutil.copy2(str(preprocessed_tmp), str(preprocessed))
            print(f"  Saved preprocessed model ({preprocessed.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"Reusing preprocessed model: {preprocessed}")

        print(f"\nCalibrating on {calibration_samples} samples...")
        calib_reader = _calibrationDataReader(tokenizer, max_length, calibration_samples, seed)

        print("Applying static INT8 quantization (CPU-targeted)...")
        quantize_static(
            str(preprocessed),
            str(quantized_path),
            calib_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    elif mode == "dynamic":
        from onnxruntime.quantization import quantize_dynamic, QuantType

        print("Applying dynamic INT8 quantization (weight-only)...")

        import onnx
        model = onnx.load(str(onnx_file))
        layernorm_nodes = set()
        for node in model.graph.node:
            if "LayerNorm" in node.op_type:
                layernorm_nodes.update(node.output)
        nodes_to_exclude = list(layernorm_nodes)
        print(f"  Excluding {len(nodes_to_exclude)} LayerNorm nodes")

        quantize_dynamic(
            str(onnx_file),
            str(quantized_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
            use_external_data_format=False,
            extra_options={"MatMatConstBOnly": True},
        )

    tokenizer.save_pretrained(str(quant_dir))

    label_map_path = Path(model_path) / "label_map.json"
    if label_map_path.exists():
        import shutil
        shutil.copy2(label_map_path, quant_dir / "label_map.json")

    quantized_onnx = list(quant_dir.glob("*.onnx"))
    for f in quantized_onnx:
        print(f"  Saved {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    print(f"\nQuantized model saved to {quant_dir}")
    return fp32_path, quant_dir


def benchmark_models(fp32_dir: Path, quant_dir: Path, use_gpu: bool = True):
    import onnxruntime as ort
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    tokenizer = AutoTokenizer.from_pretrained(str(fp32_dir))

    print("\nLoading DailyDialog test set for benchmarking...")
    ds = load_dataset("eusip/silicone", "dyda_da", split="test", trust_remote_code=True)
    texts = ds["Utterance"]
    labels = ds["Label"]

    results = {}
    LATENCY_SUBSAMPLE = 200
    inference_max_length = _load_inference_max_length()
    print(f"  Inference max_length: {inference_max_length}")

    quant_label = quant_dir.name.upper()
    for label, model_dir in [("FP32", fp32_dir), (quant_label, quant_dir)]:
        onnx_file = model_dir / "model.onnx"
        if not onnx_file.exists():
            candidates = list(model_dir.glob("*.onnx"))
            onnx_file = candidates[0] if candidates else None
        if not onnx_file:
            print(f"  Skipping {label}: no ONNX file found")
            continue

        print(f"\nBenchmarking {label}: {onnx_file.name}")
        session = _create_optimized_session(onnx_file, use_gpu=use_gpu)
        active_provider = session.get_providers()[0]
        print(f"  Execution provider: {active_provider}")

        all_preds = []
        batch_latencies = []

        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=inference_max_length,
                return_tensors="np",
            )
            input_feed = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }

            start = time.perf_counter()
            outputs = session.run(None, input_feed)
            elapsed = time.perf_counter() - start

            logits = outputs[0]
            preds = np.argmax(logits, axis=-1)
            all_preds.extend(preds.tolist())
            batch_latencies.append(elapsed)

        total_time = sum(batch_latencies)
        per_sample_ms = total_time / len(texts) * 1000

        per_item_latencies = []
        for i in range(min(LATENCY_SUBSAMPLE, len(texts))):
            inp = tokenizer(texts[i], padding="max_length", truncation=True, max_length=inference_max_length, return_tensors="np")
            feed = {
                "input_ids": inp["input_ids"].astype(np.int64),
                "attention_mask": inp["attention_mask"].astype(np.int64),
            }
            start = time.perf_counter()
            session.run(None, feed)
            elapsed = time.perf_counter() - start
            per_item_latencies.append(elapsed * 1000)

        sorted_lat = sorted(per_item_latencies)
        n = len(sorted_lat)
        p50 = sorted_lat[int(n * 0.50)]
        p95 = sorted_lat[int(n * 0.95)]
        p99 = sorted_lat[int(n * 0.99)]

        acc = accuracy_score(labels, all_preds)
        f1 = f1_score(labels, all_preds, average="macro")

        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-macro: {f1:.4f}")
        print(f"  Batch latency: {total_time:.2f}s total, {per_sample_ms:.3f}ms/sample")
        print(f"  Single-sample latency ({n} samples): p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
        print(f"\n  Classification Report:")
        print(classification_report(labels, all_preds, target_names=LABEL_NAMES, digits=4))

        results[label] = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "total_seconds": float(total_time),
            "per_sample_ms": float(per_sample_ms),
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "p99_ms": float(p99),
            "provider": active_provider,
        }

    if "FP32" in results and quant_label in results:
        fp32_f1 = results["FP32"]["f1_macro"]
        quant_f1 = results[quant_label]["f1_macro"]
        f1_delta = quant_f1 - fp32_f1
        speedup = results["FP32"]["p50_ms"] / results[quant_label]["p50_ms"] if results[quant_label]["p50_ms"] > 0 else 0

        print("\n" + "=" * 60)
        print("COMPARISON")
        print(f"  F1-macro: FP32={fp32_f1:.4f} → {quant_label}={quant_f1:.4f} (Δ={f1_delta:+.4f})")
        print(f"  Latency:  FP32={results['FP32']['p50_ms']:.3f}ms → {quant_label}={results[quant_label]['p50_ms']:.3f}ms ({speedup:.2f}x)")
        print(f"  Size:     FP32={_model_size_mb(fp32_dir):.1f}MB → {quant_label}={_model_size_mb(quant_dir):.1f}MB")
        print("=" * 60)

        if abs(f1_delta) > 0.02:
            print("\n⚠  F1 degradation exceeds 2% threshold.")

    return results


def _model_size_mb(directory: Path) -> float:
    total = sum(f.stat().st_size for f in directory.glob("*.onnx") if "-ort-cache" not in f.name)
    return total / 1e6


def benchmark_cuda_graphs(quant_dir: Path, use_gpu: bool = True):
    """Benchmark with IOBinding: keeps tensors on GPU, avoids CPU↔GPU copies.
    Also attempts CUDA Graphs capture if all nodes are CUDA-partitionable."""
    import onnxruntime as ort
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score

    if not use_gpu:
        print("IOBinding/CUDA Graphs require GPU — skipping")
        return None

    onnx_file = quant_dir / "model.onnx"
    if not onnx_file.exists():
        onnx_file = list(quant_dir.glob("*.onnx"))[0]

    max_length = _load_inference_max_length()

    print(f"\n{'=' * 60}")
    print(f"IOBinding Benchmark (max_length={max_length})")
    print(f"{'=' * 60}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_mem_reuse = True
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = [(
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": False,
        },
    )]

    session = ort.InferenceSession(str(onnx_file), sess_options=sess_opts, providers=providers)
    active_provider = session.get_providers()[0]
    print(f"  Provider: {active_provider}")

    tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))
    ds = load_dataset("eusip/silicone", "dyda_da", split="test", trust_remote_code=True)
    texts = ds["Utterance"]
    labels = ds["Label"]

    dummy = tokenizer("warmup", padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    input_ids_gpu = ort.OrtValue.ortvalue_from_numpy(dummy["input_ids"].astype(np.int64), "cuda", 0)
    mask_gpu = ort.OrtValue.ortvalue_from_numpy(dummy["attention_mask"].astype(np.int64), "cuda", 0)

    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input("input_ids", input_ids_gpu)
    io_binding.bind_ortvalue_input("attention_mask", mask_gpu)
    io_binding.bind_output("logits", "cuda", 0)

    for _ in range(10):
        session.run_with_iobinding(io_binding)

    n_samples = min(200, len(texts))
    all_preds = []
    latencies = []

    for i in range(n_samples):
        inp = tokenizer(texts[i], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        input_ids_gpu.update_inplace(inp["input_ids"].astype(np.int64))
        mask_gpu.update_inplace(inp["attention_mask"].astype(np.int64))

        t0 = time.perf_counter()
        session.run_with_iobinding(io_binding)
        latencies.append((time.perf_counter() - t0) * 1000)

        logits = io_binding.get_outputs()[0].numpy()
        all_preds.append(int(np.argmax(logits, axis=-1)[0]))

    for i in range(n_samples, len(texts)):
        inp = tokenizer(texts[i], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        input_ids_gpu.update_inplace(inp["input_ids"].astype(np.int64))
        mask_gpu.update_inplace(inp["attention_mask"].astype(np.int64))
        session.run_with_iobinding(io_binding)
        logits = io_binding.get_outputs()[0].numpy()
        all_preds.append(int(np.argmax(logits, axis=-1)[0]))

    s = sorted(latencies)
    n = len(s)
    p50 = s[n // 2]
    p95 = s[int(n * 0.95)]
    p99 = s[int(n * 0.99)]

    acc = accuracy_score(labels, all_preds)
    f1 = f1_score(labels, all_preds, average="macro")

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-macro: {f1:.4f}")
    print(f"  Single-sample latency ({n} samples): p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")

    result = {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "provider": active_provider,
        "iobinding": True,
    }

    print(f"{'=' * 60}")
    return result


def benchmark_tensorrt(quant_dir: Path, use_gpu: bool = True):
    """Benchmark with TensorRT execution provider: compiles ONNX graph into
    a GPU-optimized engine with operator fusion, kernel auto-tuning, and
    full CUDA partitioning (no Memcpy nodes)."""
    import onnxruntime as ort
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score

    if not use_gpu:
        print("TensorRT EP requires GPU — skipping")
        return None

    onnx_file = quant_dir / "model.onnx"
    if not onnx_file.exists():
        onnx_file = list(quant_dir.glob("*.onnx"))[0]

    max_length = _load_inference_max_length()

    print(f"\n{'=' * 60}")
    print(f"TensorRT EP Benchmark (max_length={max_length})")
    print(f"{'=' * 60}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    trt_cache = str(quant_dir / "trt_cache")
    providers = [(
        "TensorrtExecutionProvider",
        {
            "device_id": 0,
            "trt_max_workspace_size": 2 << 30,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache,
        },
    )]

    t_start = time.perf_counter()
    try:
        session = ort.InferenceSession(str(onnx_file), sess_options=sess_opts, providers=providers)
    except Exception as e:
        print(f"  Failed to create TensorRT session: {e}")
        return None
    compile_time = time.perf_counter() - t_start
    print(f"  Engine compilation: {compile_time:.1f}s")

    active_provider = session.get_providers()[0]
    print(f"  Active provider: {active_provider}")

    if "Tensorrt" not in active_provider:
        print(f"  WARNING: TensorRT EP not active, falling back to {active_provider}")

    tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))
    ds = load_dataset("eusip/silicone", "dyda_da", split="test", trust_remote_code=True)
    texts = ds["Utterance"]
    labels = ds["Label"]

    for _ in range(10):
        inp = tokenizer(texts[0], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        session.run(None, {"input_ids": inp["input_ids"].astype(np.int64), "attention_mask": inp["attention_mask"].astype(np.int64)})

    n_samples = min(200, len(texts))
    all_preds = []
    latencies = []

    for i in range(n_samples):
        inp = tokenizer(texts[i], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        feed = {"input_ids": inp["input_ids"].astype(np.int64), "attention_mask": inp["attention_mask"].astype(np.int64)}
        t0 = time.perf_counter()
        outputs = session.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_preds.append(int(np.argmax(outputs[0], axis=-1)[0]))

    for i in range(n_samples, len(texts)):
        inp = tokenizer(texts[i], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        feed = {"input_ids": inp["input_ids"].astype(np.int64), "attention_mask": inp["attention_mask"].astype(np.int64)}
        outputs = session.run(None, feed)
        all_preds.append(int(np.argmax(outputs[0], axis=-1)[0]))

    s = sorted(latencies)
    n = len(s)
    p50 = s[n // 2]
    p95 = s[int(n * 0.95)]
    p99 = s[int(n * 0.99)]

    acc = accuracy_score(labels, all_preds)
    f1 = f1_score(labels, all_preds, average="macro")

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-macro: {f1:.4f}")
    print(f"  Single-sample latency ({n} samples): p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")

    result = {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "provider": active_provider,
        "engine_compile_seconds": round(compile_time, 1),
        "tensorrt": True,
    }

    print(f"{'=' * 60}")
    return result


def benchmark_tensorrt_iobinding(quant_dir: Path, use_gpu: bool = True):
    """TensorRT EP + IOBinding: GPU-compiled engine with zero-copy inference.
    Also attempts CUDA Graphs capture since TensorRT partitions all nodes
    to GPU (no Memcpy nodes that block CUDA graphs under CUDAExecutionProvider)."""
    import onnxruntime as ort
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score

    if not use_gpu:
        print("TensorRT+IOBinding requires GPU — skipping")
        return None

    onnx_file = quant_dir / "model.onnx"
    if not onnx_file.exists():
        onnx_file = list(quant_dir.glob("*.onnx"))[0]

    max_length = _load_inference_max_length()

    print(f"\n{'=' * 60}")
    print(f"TensorRT + IOBinding Benchmark (max_length={max_length})")
    print(f"{'=' * 60}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    trt_cache = str(quant_dir / "trt_cache")
    providers = [(
        "TensorrtExecutionProvider",
        {
            "device_id": 0,
            "trt_max_workspace_size": 2 << 30,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache,
        },
    )]

    t_start = time.perf_counter()
    try:
        session = ort.InferenceSession(str(onnx_file), sess_options=sess_opts, providers=providers)
    except Exception as e:
        print(f"  Failed to create TensorRT session: {e}")
        return None
    compile_time = time.perf_counter() - t_start
    active_provider = session.get_providers()[0]
    print(f"  Engine compilation: {compile_time:.1f}s")
    print(f"  Active provider: {active_provider}")

    tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))
    ds = load_dataset("eusip/silicone", "dyda_da", split="test", trust_remote_code=True)
    texts = ds["Utterance"]
    labels = ds["Label"]

    dummy = tokenizer("warmup", padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    input_ids_gpu = ort.OrtValue.ortvalue_from_numpy(dummy["input_ids"].astype(np.int64), "cuda", 0)
    mask_gpu = ort.OrtValue.ortvalue_from_numpy(dummy["attention_mask"].astype(np.int64), "cuda", 0)

    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input("input_ids", input_ids_gpu)
    io_binding.bind_ortvalue_input("attention_mask", mask_gpu)
    io_binding.bind_output("logits", "cuda", 0)

    for _ in range(10):
        session.run_with_iobinding(io_binding)

    n_samples = min(200, len(texts))
    all_preds = []
    latencies = []

    for i in range(n_samples):
        inp = tokenizer(texts[i], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        input_ids_gpu.update_inplace(inp["input_ids"].astype(np.int64))
        mask_gpu.update_inplace(inp["attention_mask"].astype(np.int64))

        t0 = time.perf_counter()
        session.run_with_iobinding(io_binding)
        latencies.append((time.perf_counter() - t0) * 1000)

        logits = io_binding.get_outputs()[0].numpy()
        all_preds.append(int(np.argmax(logits, axis=-1)[0]))

    for i in range(n_samples, len(texts)):
        inp = tokenizer(texts[i], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        input_ids_gpu.update_inplace(inp["input_ids"].astype(np.int64))
        mask_gpu.update_inplace(inp["attention_mask"].astype(np.int64))
        session.run_with_iobinding(io_binding)
        logits = io_binding.get_outputs()[0].numpy()
        all_preds.append(int(np.argmax(logits, axis=-1)[0]))

    s = sorted(latencies)
    n = len(s)
    p50 = s[n // 2]
    p95 = s[int(n * 0.95)]
    p99 = s[int(n * 0.99)]

    acc = accuracy_score(labels, all_preds)
    f1 = f1_score(labels, all_preds, average="macro")

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-macro: {f1:.4f}")
    print(f"  Single-sample latency ({n} samples): p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")

    result = {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "provider": active_provider,
        "engine_compile_seconds": round(compile_time, 1),
        "tensorrt": True,
        "iobinding": True,
    }

    print(f"{'=' * 60}")
    return result


def benchmark_batch(quant_dir: Path, use_gpu: bool = True):
    """Benchmark batch inference with dynamic padding across multiple batch sizes.
    Measures throughput (samples/sec) and per-sample latency for serving workloads."""
    import onnxruntime as ort
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score

    if not use_gpu:
        print("Batch benchmark requires GPU — skipping")
        return None

    onnx_file = quant_dir / "model.onnx"
    if not onnx_file.exists():
        onnx_file = list(quant_dir.glob("*.onnx"))[0]

    max_length = _load_inference_max_length()

    print(f"\n{'=' * 60}")
    print(f"Batch Inference Benchmark (dynamic padding, max_length={max_length})")
    print(f"{'=' * 60}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_mem_reuse = True
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    trt_cache = str(quant_dir / "trt_cache")
    providers = [(
        "TensorrtExecutionProvider",
        {
            "device_id": 0,
            "trt_max_workspace_size": 2 << 30,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache,
        },
    )]

    try:
        session = ort.InferenceSession(str(onnx_file), sess_options=sess_opts, providers=providers)
    except Exception as e:
        print(f"  Failed to create session: {e}")
        return None

    active_provider = session.get_providers()[0]
    print(f"  Provider: {active_provider}")

    tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))
    ds = load_dataset("eusip/silicone", "dyda_da", split="test", trust_remote_code=True)
    texts = ds["Utterance"]
    labels = ds["Label"]

    batch_sizes = [1, 4, 8, 16, 32, 64]
    all_preds_accum = {}
    results = {}

    for bs in batch_sizes:
        all_preds = []
        batch_latencies = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            feed = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }

            t0 = time.perf_counter()
            outputs = session.run(None, feed)
            elapsed = time.perf_counter() - t0

            preds = np.argmax(outputs[0], axis=-1)
            all_preds.extend(preds.tolist())
            batch_latencies.append((elapsed, len(batch)))

        total_time = sum(t for t, _ in batch_latencies)
        total_samples = sum(n for _, n in batch_latencies)
        throughput = total_samples / total_time
        per_sample_ms = total_time / total_samples * 1000

        acc = accuracy_score(labels, all_preds)
        f1 = f1_score(labels, all_preds, average="macro")
        all_preds_accum[bs] = all_preds

        results[bs] = {
            "throughput_sps": round(throughput, 1),
            "per_sample_ms": round(per_sample_ms, 3),
            "total_seconds": round(total_time, 3),
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1), 4),
        }
        print(f"  bs={bs:>3d}: {throughput:>8.1f} samples/s, {per_sample_ms:.3f}ms/sample, "
              f"total={total_time:.2f}s, acc={acc:.4f}, f1={f1:.4f}")

    best_bs = max(results, key=lambda k: results[k]["throughput_sps"])
    print(f"\n  Best throughput: bs={best_bs} ({results[best_bs]['throughput_sps']:.1f} samples/s)")
    print(f"  Best latency:    bs={min(results, key=lambda k: results[k]['per_sample_ms'])} "
          f"({min(r['per_sample_ms'] for r in results.values()):.3f}ms/sample)")
    print(f"{'=' * 60}")

    return {"provider": active_provider, "batch_results": results, "best_batch_size": best_bs}


def main():
    parser = argparse.ArgumentParser(description="Quantization for dialogue-act classifier")
    parser.add_argument("--model-path", default="models/en/final")
    parser.add_argument("--output-dir", default=None, help="Defaults to model-path/quantized")
    parser.add_argument("--mode", choices=["fp16", "dynamic", "static"], default="fp16", help="Quantization mode (default: fp16)")
    parser.add_argument("--calibration-samples", type=int, default=500, help="Samples for static calibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="Benchmark on CPU only")
    parser.add_argument("--iobinding", action="store_true", help="Also benchmark with IOBinding (GPU-resident tensors)")
    parser.add_argument("--tensorrt", action="store_true", help="Also benchmark with TensorRT EP")
    parser.add_argument("--trt-iobinding", action="store_true", help="Benchmark TensorRT + IOBinding")
    parser.add_argument("--batch", action="store_true", help="Benchmark batch inference with dynamic padding")
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.model_path) / "quantized")

    fp32_dir, quant_dir = quantize_model(
        model_path=args.model_path,
        output_dir=output_dir,
        mode=args.mode,
        calibration_samples=args.calibration_samples,
        seed=args.seed,
    )

    if not args.skip_benchmark:
        results = benchmark_models(fp32_dir, quant_dir, use_gpu=not args.cpu_only)
        results_path = Path(output_dir) / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {results_path}")

    if args.iobinding:
        cg_results = benchmark_cuda_graphs(quant_dir, use_gpu=not args.cpu_only)
        if cg_results:
            cg_path = Path(output_dir) / "iobinding_results.json"
            with open(cg_path, "w") as f:
                json.dump(cg_results, f, indent=2)
            print(f"IOBinding results saved to {cg_path}")

    if args.tensorrt:
        trt_results = benchmark_tensorrt(quant_dir, use_gpu=not args.cpu_only)
        if trt_results:
            trt_path = Path(output_dir) / "tensorrt_results.json"
            with open(trt_path, "w") as f:
                json.dump(trt_results, f, indent=2)
            print(f"TensorRT results saved to {trt_path}")

    if args.trt_iobinding:
        trtio_results = benchmark_tensorrt_iobinding(quant_dir, use_gpu=not args.cpu_only)
        if trtio_results:
            trtio_path = Path(output_dir) / "trt_iobinding_results.json"
            with open(trtio_path, "w") as f:
                json.dump(trtio_results, f, indent=2)
            print(f"TRT+IOBinding results saved to {trtio_path}")

    if args.batch:
        batch_results = benchmark_batch(quant_dir, use_gpu=not args.cpu_only)
        if batch_results:
            batch_path = Path(output_dir) / "batch_results.json"
            with open(batch_path, "w") as f:
                json.dump(batch_results, f, indent=2)
            print(f"Batch results saved to {batch_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
