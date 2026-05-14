"""
Per-language evaluation of the multilingual dialogue act classifier.

Tests on EN (SILICONE dyda_da + XDailyDialog), DE (XDailyDialog), RU (xdailydialog-ru).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

LABEL_NAMES = ["commissive", "directive", "inform", "question"]


def load_model_and_tokenizer(model_path: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def predict_batch(texts, model, tokenizer, device, batch_size=64):
    import torch

    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        preds = np.argmax(outputs.logits.cpu().numpy(), axis=-1)
        all_preds.extend(preds.tolist())
    return all_preds


def parse_xdailydialog(path: str) -> list[dict]:
    records = []
    act_map = {1: 2, 2: 3, 3: 1, 4: 0}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            utterances = [u.strip() for u in parts[0].split("__eou__") if u.strip()]
            acts = [int(a) for a in parts[2].split()]
            for utt, act in zip(utterances, acts):
                records.append({"text": utt, "label": act_map.get(act, 2)})
    return records


def load_ru_from_hf() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("WSHAPER/xdailydialog-ru", split="test", trust_remote_code=True)
    records = []
    for dialog in ds:
        for utt, act in zip(dialog["utterances"], dialog["acts"]):
            records.append({"text": utt, "label": act})
    return records


def eval_split(name: str, records: list[dict], model, tokenizer, device):
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]

    start = time.perf_counter()
    preds = predict_batch(texts, model, tokenizer, device)
    elapsed = time.perf_counter() - start

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    avg_ms = elapsed / len(texts) * 1000

    print(f"\n{'=' * 60}")
    print(f"  {name}: {len(records)} utterances, {avg_ms:.2f}ms/sample")
    print(f"  Accuracy: {acc:.4f}, F1 macro: {f1:.4f}")
    print(f"{'=' * 60}")
    print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=4, zero_division=0))

    return {"name": name, "n": len(records), "accuracy": float(acc), "f1_macro": float(f1), "avg_ms": float(avg_ms)}


def main():
    parser = argparse.ArgumentParser(description="Per-language evaluation")
    parser.add_argument("--model-path", default="models/multilingual/final")
    parser.add_argument("--xdailydialog-path", default=".reference/XDailyDialog/data")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_path)
    print(f"Model: {args.model_path}, Device: {device}\n")

    results = []

    print("Loading SILICONE dyda_da test (EN)...")
    from datasets import load_dataset
    silicone_test = load_dataset("eusip/silicone", "dyda_da", split="test", trust_remote_code=True)
    en_silicone = [{"text": r["Utterance"], "label": r["Label"]} for r in silicone_test]
    results.append(eval_split("EN (SILICONE dyda_da)", en_silicone, model, tokenizer, device))

    xdd_path = Path(args.xdailydialog_path)
    if (xdd_path / "en_test_human.txt").exists():
        en_xdd = parse_xdailydialog(str(xdd_path / "en_test_human.txt"))
        results.append(eval_split("EN (XDailyDialog)", en_xdd, model, tokenizer, device))

    if (xdd_path / "de_test_human.txt").exists():
        de_xdd = parse_xdailydialog(str(xdd_path / "de_test_human.txt"))
        results.append(eval_split("DE (XDailyDialog)", de_xdd, model, tokenizer, device))

    print("Loading RU (WSHAPER/xdailydialog-ru test)...")
    ru_records = load_ru_from_hf()
    results.append(eval_split("RU (xdailydialog-ru)", ru_records, model, tokenizer, device))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Language':<25} {'N':>6} {'Accuracy':>10} {'F1 macro':>10} {'ms/sample':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<25} {r['n']:>6} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r['avg_ms']:>10.2f}")

    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "multilingual_per_language.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to eval_results/multilingual_per_language.json")


if __name__ == "__main__":
    main()
