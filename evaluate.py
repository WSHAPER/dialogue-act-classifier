"""
Evaluate a fine-tuned sentence classifier.

Runs three evaluation modes:
  1. DailyDialog test set — classification report + confusion matrix
  2. Custom edge-case test suite (tests/test_cases.json)
  3. TigreGotico baseline comparison (if ONNX model available)

Usage:
    python evaluate.py                              # evaluate models/en/final
    python evaluate.py --model-path models/en/run_20260513_120000
    python evaluate.py --skip-baseline              # skip TigreGotico comparison
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABEL_NAMES = ["commissive", "directive", "inform", "question"]

INFERENCE_MAX_LENGTH = 48


def _create_optimized_session(onnx_path, use_gpu=True):
    """Create ONNX Runtime session with ORT_ENABLE_ALL, memory patterns,
    cudnn exhaustive search for Tensor Core FP16 acceleration."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_mem_reuse = True
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

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


def load_model_and_tokenizer(model_path: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def predict_batch(texts: list[str], model, tokenizer, device, batch_size: int = 64):
    import torch

    all_preds = []
    all_probs = []
    all_times = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.perf_counter() - start

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_times.append(elapsed)

    total_time = sum(all_times)
    avg_time = total_time / len(texts) * 1000
    return all_preds, all_probs, total_time, avg_time


def evaluate_dailydialog(model, tokenizer, device):
    from datasets import load_dataset

    print("=" * 60)
    print("Evaluating on DailyDialog test set")
    print("=" * 60)

    ds = load_dataset("eusip/silicone", "dyda_da", split="test", trust_remote_code=True)

    texts = ds["Utterance"]
    labels = ds["Label"]

    preds, probs, total_time, avg_time = predict_batch(texts, model, tokenizer, device)

    print(f"\nInference: {total_time:.2f}s total, {avg_time:.2f}ms/sample")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=4))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1, "avg_ms": avg_time, "confusion_matrix": cm.tolist()}


def evaluate_edge_cases(model, tokenizer, device, test_cases_path: str = "tests/test_cases.json"):
    print("\n" + "=" * 60)
    print("Evaluating on ASR edge-case test suite")
    print("=" * 60)

    with open(test_cases_path) as f:
        test_cases = json.load(f)

    label_map = {"commissive": 0, "directive": 1, "inform": 2, "question": 3}
    texts = [tc["text"] for tc in test_cases]
    expected = [label_map[tc["expected"]] for tc in test_cases]
    categories = [tc["category"] for tc in test_cases]

    preds, probs, total_time, avg_time = predict_batch(texts, model, tokenizer, device)

    print(f"\nInference: {avg_time:.2f}ms/sample")
    print(f"\nClassification Report:")
    print(classification_report(expected, preds, target_names=LABEL_NAMES, digits=4))

    correct = sum(1 for e, p in zip(expected, preds) if e == p)
    total = len(expected)
    print(f"\nEdge-case accuracy: {correct}/{total} ({correct/total:.1%})")

    print("\nPer-sample results:")
    for tc, pred, prob in zip(test_cases, preds, probs):
        status = "OK" if label_map[tc["expected"]] == pred else "FAIL"
        pred_name = LABEL_NAMES[pred]
        exp_name = tc["expected"]
        confidence = max(prob)
        print(f"  [{status}] \"{tc['text'][:50]}\" → {pred_name} (expected {exp_name}, conf {confidence:.2f})")

    failures = []
    for tc, pred, prob in zip(test_cases, preds, probs):
        if label_map[tc["expected"]] != pred:
            failures.append({
                "text": tc["text"],
                "expected": tc["expected"],
                "predicted": LABEL_NAMES[pred],
                "confidence": round(float(max(prob)), 4),
                "category": tc["category"],
            })

    return {
        "accuracy": correct / total,
        "failures": failures,
        "avg_ms": avg_time,
    }


def evaluate_tigreGotico_baseline(test_cases_path: str = "tests/test_cases.json"):
    print("\n" + "=" * 60)
    print("TigreGotico Baseline Comparison")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping baseline comparison")
        return None

    import glob

    model_candidates = list(Path.home().glob(
        ".local/share/dialogue-act-classifier/models/classify/sentence_type_EN*.onnx"
    ))
    if not model_candidates:
        print("No TigreGotico ONNX model found, skipping baseline")
        return None

    model_path = sorted(model_candidates)[-1]
    print(f"Loading: {model_path}")

    session = ort.InferenceSession(str(model_path))

    tigre_map = {
        "command": "directive",
        "request": "directive",
        "statement": "inform",
        "exclamation": "inform",
        "polar_question": "question",
        "wh_question": "question",
    }

    with open(test_cases_path) as f:
        test_cases = json.load(f)

    label_map = {"commissive": 0, "directive": 1, "inform": 2, "question": 3}
    texts = [tc["text"] for tc in test_cases]
    expected = [label_map[tc["expected"]] for tc in test_cases]

    preds = []
    for text in texts:
        input_array = np.array([text], dtype=object)
        outputs = session.run(None, {"input": input_array})
        logits = outputs[1][0]
        probs = np.exp(logits) / np.exp(logits).sum()

        class_names = session.get_outputs()[1].name
        try:
            meta = session.get_modelmeta()
            if "labels" in meta.custom_metadata_map:
                labels_str = meta.custom_metadata_map["labels"]
                class_labels = json.loads(labels_str)
            else:
                raise ValueError
        except Exception:
            class_labels = ["command", "exclamation", "polar_question", "request", "statement", "wh_question"]

        best_idx = int(np.argmax(probs))
        tigre_class = class_labels[best_idx]
        mapped = tigre_map.get(tigre_class, "inform")
        preds.append(label_map[mapped])

    correct = sum(1 for e, p in zip(expected, preds) if e == p)
    total = len(expected)
    acc = correct / total
    f1 = f1_score(expected, preds, average="macro")

    print(f"\nTigreGotico accuracy on edge cases: {correct}/{total} ({acc:.1%})")
    print(f"TigreGotico F1 (macro): {f1:.4f}")
    print(classification_report(expected, preds, target_names=LABEL_NAMES, digits=4))

    print("\nPer-sample results:")
    for tc, pred in zip(test_cases, preds):
        status = "OK" if label_map[tc["expected"]] == pred else "FAIL"
        pred_name = LABEL_NAMES[pred]
        exp_name = tc["expected"]
        print(f"  [{status}] \"{tc['text'][:50]}\" → {pred_name} (expected {exp_name})")

    return {"accuracy": acc, "f1_macro": f1}


def plot_confusion_matrix(cm, labels, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — DailyDialog Test Set")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


def benchmark_quantized_model(quantized_path: str, test_cases_path: str = "tests/test_cases.json"):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping quantized model benchmark")
        return None

    print("\n" + "=" * 60)
    print("Benchmarking Quantized ONNX Model")
    print("=" * 60)

    quant_dir = Path(quantized_path)
    onnx_file = quant_dir / "model.onnx"
    if not onnx_file.exists():
        candidates = list(quant_dir.glob("*.onnx"))
        if candidates:
            onnx_file = candidates[0]
        else:
            print(f"No ONNX file found in {quantized_path}")
            return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = _create_optimized_session(onnx_file, use_gpu=True)
    active_provider = session.get_providers()[0]
    print(f"Model: {onnx_file.name} ({onnx_file.stat().st_size / 1e6:.1f} MB)")
    print(f"Execution provider: {active_provider}")
    print(f"Inference max_length: {INFERENCE_MAX_LENGTH}")

    from transformers import AutoTokenizer
    from datasets import load_dataset
    from sklearn.metrics import classification_report

    tokenizer_path = quant_dir.parent / "fp32"
    if not (tokenizer_path / "tokenizer.json").exists():
        tokenizer_path = quant_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    ds = load_dataset("eusip/silicone", "dyda_da", split="test", trust_remote_code=True)
    texts = ds["Utterance"]
    labels = ds["Label"]

    all_preds = []
    per_item_latencies = []

    for i, text in enumerate(texts):
        inp = tokenizer(text, padding="max_length", truncation=True, max_length=INFERENCE_MAX_LENGTH, return_tensors="np")
        feed = {
            "input_ids": inp["input_ids"].astype(np.int64),
            "attention_mask": inp["attention_mask"].astype(np.int64),
        }
        start = time.perf_counter()
        outputs = session.run(None, feed)
        elapsed = time.perf_counter() - start
        per_item_latencies.append(elapsed * 1000)

        pred = int(np.argmax(outputs[0], axis=-1)[0])
        all_preds.append(pred)

    sorted_lat = sorted(per_item_latencies)
    n = len(sorted_lat)
    p50 = sorted_lat[int(n * 0.50)]
    p95 = sorted_lat[int(n * 0.95)]
    p99 = sorted_lat[int(n * 0.99)]
    total_ms = sum(per_item_latencies)
    avg_ms = total_ms / len(texts)

    acc = accuracy_score(labels, all_preds)
    f1 = f1_score(labels, all_preds, average="macro")

    print(f"\nDailyDialog Test Set:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-macro: {f1:.4f}")
    print(f"  Latency: avg={avg_ms:.3f}ms, p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
    print(f"\nClassification Report:")
    print(classification_report(labels, all_preds, target_names=LABEL_NAMES, digits=4))

    with open(test_cases_path) as f:
        test_cases = json.load(f)

    label_map = {"commissive": 0, "directive": 1, "inform": 2, "question": 3}
    ec_texts = [tc["text"] for tc in test_cases]
    ec_expected = [label_map[tc["expected"]] for tc in test_cases]

    ec_preds = []
    for text in ec_texts:
        inp = tokenizer(text, padding="max_length", truncation=True, max_length=INFERENCE_MAX_LENGTH, return_tensors="np")
        feed = {
            "input_ids": inp["input_ids"].astype(np.int64),
            "attention_mask": inp["attention_mask"].astype(np.int64),
        }
        outputs = session.run(None, feed)
        ec_preds.append(int(np.argmax(outputs[0], axis=-1)[0]))

    ec_correct = sum(1 for e, p in zip(ec_expected, ec_preds) if e == p)
    ec_acc = ec_correct / len(ec_expected)
    print(f"\nEdge-case accuracy: {ec_correct}/{len(ec_expected)} ({ec_acc:.1%})")

    return {
        "dailydialog": {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "avg_ms": float(avg_ms),
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "p99_ms": float(p99),
            "provider": active_provider,
        },
        "edge_cases": {
            "accuracy": float(ec_acc),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentence classifier")
    parser.add_argument("--model-path", default="models/en/final")
    parser.add_argument("--test-cases", default="tests/test_cases.json")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-dailydialog", action="store_true")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--quantized-model", default=None, help="Path to quantized ONNX model dir for benchmarking")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(args.model_path)
    print(f"Model: {args.model_path}, Device: {device}")

    results = {}

    if not args.skip_dailydialog:
        dd_results = evaluate_dailydialog(model, tokenizer, device)
        results["dailydialog"] = dd_results
        if "confusion_matrix" in dd_results:
            cm = np.array(dd_results["confusion_matrix"])
            plot_confusion_matrix(cm, LABEL_NAMES, str(output_dir / "confusion_matrix.png"))

    ec_results = evaluate_edge_cases(model, tokenizer, device, args.test_cases)
    results["edge_cases"] = ec_results

    if not args.skip_baseline:
        baseline = evaluate_tigreGotico_baseline(args.test_cases)
        if baseline:
            results["tigreGotico_baseline"] = baseline

            if "edge_cases" in results:
                our_acc = results["edge_cases"]["accuracy"]
                base_acc = baseline["accuracy"]
                print("\n" + "=" * 60)
                print("COMPARISON (edge-case test suite)")
                print(f"  Fine-tuned DistilBERT: {our_acc:.1%}")
                print(f"  TigreGotico baseline:  {base_acc:.1%}")
                improvement = our_acc - base_acc
                print(f"  Improvement:           {improvement:+.1%}")
                print("=" * 60)

    if args.quantized_model:
        quant_results = benchmark_quantized_model(args.quantized_model, args.test_cases)
        if quant_results:
            results["quantized_onnx"] = quant_results

            if "dailydialog" in results and "dailydialog" in quant_results:
                fp32_f1 = results["dailydialog"]["f1_macro"]
                int8_f1 = quant_results["dailydialog"]["f1_macro"]
                f1_delta = int8_f1 - fp32_f1
                fp32_ms = results["dailydialog"]["avg_ms"]
                int8_ms = quant_results["dailydialog"]["avg_ms"]
                speedup = fp32_ms / int8_ms if int8_ms > 0 else 0

                print("\n" + "=" * 60)
                print("FP32 vs INT8 COMPARISON (DailyDialog test set)")
                print(f"  F1-macro:  FP32={fp32_f1:.4f} → INT8={int8_f1:.4f} (Δ={f1_delta:+.4f})")
                print(f"  Latency:   FP32={fp32_ms:.3f}ms → INT8={int8_ms:.3f}ms ({speedup:.2f}x)")
                print("=" * 60)

    with open(output_dir / "evaluation_results.json", "w") as f:
        serializable = {}
        for k, v in results.items():
            sv = {}
            for vk, vv in v.items():
                if isinstance(vv, np.floating):
                    sv[vk] = float(vv)
                elif isinstance(vv, np.integer):
                    sv[vk] = int(vv)
                else:
                    sv[vk] = vv
            serializable[k] = sv
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
