"""
Fine-tune multilingual DistilBERT on XDailyDialog + translated Russian data
for 4-class dialogue act classification: commissive/directive/inform/question.

Usage:
    python train_multilingual.py                    # all languages
    python train_multilingual.py --languages en de  # specific languages
    python train_multilingual.py --skip-ru          # skip Russian translation
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from datasets import concatenate_datasets, Dataset, DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_config(path: str = "config_multilingual.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_xdailydialog_line(line: str, act_map: dict) -> list[dict]:
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return []
    dialogue = parts[0]
    act_labels = parts[2].split()
    utterances = [u.strip() for u in dialogue.split("__eou__") if u.strip()]
    results = []
    for utt, act in zip(utterances, act_labels):
        label = act_map.get(int(act), 2)
        results.append({"text": utt, "label": label})
    return results


def load_xdailydialog(cfg: dict, languages: list[str] = None) -> DatasetDict:
    xdd_cfg = cfg["xdailydialog"]
    act_map = {int(k): v for k, v in xdd_cfg["act_label_map"].items()}
    data_dir = Path(xdd_cfg["local_path"]) / "data"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"XDailyDialog not found at {data_dir}. "
            f"Run: git clone {xdd_cfg['repo']} {xdd_cfg['local_path']}"
        )

    lang_configs = xdd_cfg["languages"]
    if languages:
        lang_configs = [lc for lc in lang_configs if lc["code"] in languages]

    splits = {"train": [], "validation": [], "test": []}
    split_file_map = {"train": "train", "validation": "dev", "test": "test"}

    for lang in lang_configs:
        code = lang["code"]
        print(f"  Loading {code}...")
        for split_name, file_key in split_file_map.items():
            filepath = data_dir / lang[file_key]
            if not filepath.exists():
                print(f"    Warning: {filepath} not found, skipping")
                continue
            records = []
            with open(filepath) as f:
                for line in f:
                    records.extend(parse_xdailydialog_line(line, act_map))
            for r in records:
                r["language"] = code
            splits[split_name].extend(records)
            print(f"    {split_name}: {len(records)} utterances")

    result = {}
    for split_name, records in splits.items():
        if records:
            result[split_name] = Dataset.from_list(records)

    return DatasetDict(result)


def translate_to_russian(ds: Dataset, batch_size: int = 64) -> Dataset:
    print("Translating to Russian (this may take a while)...")
    from transformers import pipeline

    translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-ru",
        device=0 if _cuda_available() else -1,
        batch_size=batch_size,
    )

    texts = ds["text"]
    labels = ds["label"]

    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = translator(batch, max_length=256)
        translated_texts.extend([r["translation_text"] for r in results])
        if (i + batch_size) % 5000 < batch_size:
            print(f"  Translated {min(i + batch_size, len(texts))}/{len(texts)}")

    records = [
        {"text": t, "label": l, "language": "ru"}
        for t, l in zip(translated_texts, labels)
    ]
    return Dataset.from_list(records)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_silicone_dyda() -> DatasetDict:
    ds = load_dataset("eusip/silicone", "dyda_da", trust_remote_code=True)

    def prepare(example):
        return {"text": example["Utterance"], "label": example["Label"], "language": "en"}

    result = {}
    for split in ("train", "validation", "test"):
        if split in ds:
            mapped = ds[split].map(prepare, remove_columns=ds[split].column_names)
            result[split] = mapped
    return DatasetDict(result)


def tokenize_dataset(ds, tokenizer, max_length: int):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train multilingual sentence classifier")
    parser.add_argument("--config", default="config_multilingual.yaml")
    parser.add_argument("--languages", nargs="+", default=None, help="Subset of languages (en de it ru)")
    parser.add_argument("--skip-ru", action="store_true", help="Skip Russian translation step")
    parser.add_argument("--skip-silicone", action="store_true", help="Skip SILICONE dyda_da (use XDailyDialog EN only)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-to-hub", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    import torch

    seed = cfg["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()

    base_model = cfg["base_model"]
    num_labels = len(cfg["label_map"])
    max_length = cfg["max_seq_length"]
    epochs = args.epochs or cfg["num_epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    lr = args.lr or cfg["learning_rate"]
    output_dir = args.output_dir or cfg["output_dir"]

    print(f"Base model: {base_model}")
    print(f"Labels: {cfg['label_map']}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model_config = AutoConfig.from_pretrained(
        base_model,
        num_labels=num_labels,
        id2label={str(i): name for i, name in cfg["label_map"].items()},
        label2id={name: i for i, name in cfg["label_map"].items()},
    )
    model = AutoModelForSequenceClassification.from_pretrained(base_model, config=model_config)

    all_splits = {"train": [], "validation": [], "test": []}

    print("\nLoading SILICONE dyda_da (EN)...")
    if not args.skip_silicone:
        try:
            silicone = load_silicone_dyda()
            for split in all_splits:
                if split in silicone:
                    all_splits[split].append(silicone[split])
                    print(f"  {split}: {len(silicone[split])}")
        except Exception as e:
            print(f"  Failed: {e}")

    print("\nLoading XDailyDialog...")
    try:
        xdd = load_xdailydialog(cfg, args.languages)
        for split in all_splits:
            if split in xdd:
                all_splits[split].append(xdd[split])
    except Exception as e:
        print(f"  Failed: {e}")

    if not args.skip_ru and (args.languages is None or "ru" in args.languages):
        print("\nGenerating Russian data via translation...")
        try:
            ru_source = all_splits["train"][0] if all_splits["train"] else None
            if ru_source:
                en_records = [r for r in ru_source if r.get("language") == "en"]
                if en_records:
                    en_ds = Dataset.from_list(en_records)
                    ru_ds = translate_to_russian(en_ds)
                    all_splits["train"].append(ru_ds)
                    print(f"  Russian train: {len(ru_ds)}")

                    for split_name in ("validation", "test"):
                        if all_splits[split_name]:
                            en_val = [r for r in all_splits[split_name][0] if r.get("language") == "en"]
                            if en_val:
                                ru_val = translate_to_russian(Dataset.from_list(en_val), batch_size=32)
                                all_splits[split_name].append(ru_val)
                                print(f"  Russian {split_name}: {len(ru_val)}")
        except Exception as e:
            print(f"  Russian translation failed: {e}")
            print("  Continuing without Russian...")

    merged = {}
    for split_name, datasets in all_splits.items():
        if datasets:
            merged[split_name] = concatenate_datasets(datasets)
            lang_dist = {}
            for r in merged[split_name]:
                lang = r.get("language", "unknown")
                lang_dist[lang] = lang_dist.get(lang, 0) + 1
            print(f"\n{split_name}: {len(merged[split_name])} total")
            for lang, count in sorted(lang_dist.items()):
                print(f"  {lang}: {count}")

    train_ds = merged.get("train")
    val_ds = merged.get("validation", merged.get("test"))
    if not train_ds:
        print("No training data found!")
        return

    train_ds = tokenize_dataset(train_ds, tokenizer, max_length)
    val_ds = tokenize_dataset(val_ds, tokenizer, max_length)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(run_dir / "logs"),
        logging_steps=100,
        seed=seed,
        fp16=use_cuda,
        dataloader_num_workers=0,
        report_to="none",
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with open(final_dir / "label_map.json", "w") as f:
        json.dump(cfg["label_map"], f, indent=2)
    with open(final_dir / "training_config.json", "w") as f:
        json.dump(
            {
                "base_model": base_model,
                "num_labels": num_labels,
                "max_seq_length": max_length,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "seed": seed,
                "trained_at": timestamp,
                "languages": list(set(
                    r.get("language", "unknown")
                    for r in merged.get("train", [])
                )),
            },
            f,
            indent=2,
        )

    print(f"\nModel saved to {final_dir}")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
