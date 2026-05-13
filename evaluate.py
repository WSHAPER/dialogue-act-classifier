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

LABEL_NAMES = ["statement", "question", "instruction"]


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

    ds = load_dataset("eusip/silicone", "dyda_da", split="test")

    dd_map = {1: 0, 2: 1, 3: 2, 4: 0}
    texts = ds["Utterance"]
    labels = [dd_map[l] for l in ds["Label"]]

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

    label_map = {"statement": 0, "question": 1, "instruction": 2}
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
        ".local/share/phonolitui/models/classify/sentence_type_EN*.onnx"
    ))
    if not model_candidates:
        print("No TigreGotico ONNX model found, skipping baseline")
        return None

    model_path = sorted(model_candidates)[-1]
    print(f"Loading: {model_path}")

    session = ort.InferenceSession(str(model_path))

    tigre_map = {
        "command": "instruction",
        "request": "instruction",
        "statement": "statement",
        "exclamation": "statement",
        "polar_question": "question",
        "wh_question": "question",
    }

    with open(test_cases_path) as f:
        test_cases = json.load(f)

    label_map = {"statement": 0, "question": 1, "instruction": 2}
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
        mapped = tigre_map.get(tigre_class, "statement")
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentence classifier")
    parser.add_argument("--model-path", default="models/en/final")
    parser.add_argument("--test-cases", default="tests/test_cases.json")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-dailydialog", action="store_true")
    parser.add_argument("--output-dir", default="eval_results")
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
