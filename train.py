"""
Fine-tune DistilBERT on DailyDialog dialogue acts for 3-class sentence
type classification: statement / question / instruction.

Usage:
    python train.py                          # defaults from config.yaml
    python train.py --epochs 6 --lr 3e-5    # override hyperparams
    python train.py --no-augment             # skip ASR augmentation
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from augment import build_augmented_dataset


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def remap_dailydialog(example: dict, label_map: dict) -> dict:
    example["label"] = label_map[example["Label"]]
    example["text"] = example["Utterance"]
    return example


def load_dailydialog(cfg: dict) -> dict:
    ds = load_dataset(cfg["dataset"]["name"], cfg["dataset"]["config"])
    dd_map = {int(k): v for k, v in cfg["dataset"]["dailydialog_map"].items()}

    splits = {}
    for split_name in ("train", "validation", "test"):
        if split_name in ds:
            mapped = ds[split_name].map(
                remap_dailydialog,
                fn_kwargs={"label_map": dd_map},
                remove_columns=ds[split_name].column_names,
            )
            mapped = mapped.remove_columns(
                [c for c in mapped.column_names if c not in ("text", "label")]
            )
            splits[split_name] = mapped
    return splits


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
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentence classification")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--push-to-hub", default=None, help="HuggingFace repo ID to push to")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = args.seed or cfg["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model_config = AutoConfig.from_pretrained(
        base_model,
        num_labels=num_labels,
        id2label={i: name for i, name in cfg["label_map"].items()},
        label2id=cfg["label_map"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, config=model_config
    )

    print("Loading DailyDialog...")
    splits = load_dailydialog(cfg)

    train_ds = splits["train"]
    val_ds = splits.get("validation", splits.get("test"))

    if not args.no_augment:
        print("Building ASR augmentation...")
        aug_ds = build_augmented_dataset(
            cfg.get("augmentation", "tests/test_cases.json"),
            augment_factor=args.augment_factor,
            seed=seed,
        )
        aug_ds = aug_ds.cast(train_ds.features)
        train_ds = concatenate_datasets([train_ds, aug_ds])
        print(f"Training set after augmentation: {len(train_ds)}")

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(run_dir / "logs"),
        logging_steps=50,
        seed=seed,
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
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
                "augmented": not args.no_augment,
                "augment_factor": args.augment_factor if not args.no_augment else 0,
            },
            f,
            indent=2,
        )

    print(f"\nModel saved to {final_dir}")
    print("Evaluating on validation set...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub}")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
