"""
Export a fine-tuned sentence classifier for Rust inference.

Outputs:
  1. PyTorch weights + tokenizer + config → for candle-transformers loading
  2. ONNX model (optional) → for ort crate loading

The candle export produces the exact files phonolitui's NeuralClassifier expects:
  - pytorch_model.bin (or model.safetensors)
  - tokenizer.json
  - config.json (DistilBertConfig compatible with candle)

Usage:
    python export.py                              # export models/en/final
    python export.py --model-path models/en/run_20260513_120000
    python export.py --onnx                       # also export ONNX
    python export.py --safetensors                # use safetensors instead of pytorch
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CANDLE_CONFIG = {
    "model_type": "distilbert",
    "activation": "gelu",
    "attention_dropout": 0.1,
    "dim": 768,
    "dropout": 0.1,
    "hidden_dim": 3072,
    "initializer_range": 0.02,
    "max_position_embeddings": 512,
    "n_heads": 12,
    "n_layers": 6,
    "pad_token_id": 0,
    "qa_dropout": 0.1,
    "seq_classif_dropout": 0.2,
    "sinusoidal_pos_embds": False,
    "vocab_size": 30522,
}


def export_pytorch(model_path: str, output_dir: str, use_safetensors: bool = False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Exporting to {out}...")

    if use_safetensors:
        model.save_pretrained(str(out), safe_serialization=True)
    else:
        state_dict = model.state_dict()

        candle_state = {}
        for key, value in state_dict.items():
            candle_state[key] = value

        torch.save(candle_state, out / "pytorch_model.bin")
        print(f"  Saved pytorch_model.bin ({(out / 'pytorch_model.bin').stat().st_size / 1e6:.1f} MB)")

    tokenizer.save_pretrained(str(out))
    print("  Saved tokenizer.json")

    original_config = json.loads(Path(model_path).joinpath("config.json").read_text())

    candle_cfg = {**CANDLE_CONFIG}
    if "max_position_embeddings" in original_config:
        candle_cfg["max_position_embeddings"] = original_config["max_position_embeddings"]
    if "vocab_size" in original_config:
        candle_cfg["vocab_size"] = original_config["vocab_size"]
    if "dim" in original_config:
        candle_cfg["dim"] = original_config["dim"]
    if "n_layers" in original_config:
        candle_cfg["n_layers"] = original_config["n_layers"]
    if "n_heads" in original_config:
        candle_cfg["n_heads"] = original_config["n_heads"]
    if "hidden_dim" in original_config:
        candle_cfg["hidden_dim"] = original_config["hidden_dim"]

    if "num_labels" not in original_config:
        candle_cfg["num_labels"] = 4

    id2label = original_config.get("id2label", {"0": "commissive", "1": "directive", "2": "inform", "3": "question"})
    label2id = original_config.get("label2id", {"commissive": 0, "directive": 1, "inform": 2, "question": 3})

    candle_cfg["id2label"] = id2label
    candle_cfg["label2id"] = label2id
    candle_cfg["num_labels"] = len(id2label)

    with open(out / "config.json", "w") as f:
        json.dump(candle_cfg, f, indent=2)
    print("  Saved config.json (candle-compatible)")

    label_map_path = Path(model_path) / "label_map.json"
    if label_map_path.exists():
        shutil.copy2(label_map_path, out / "label_map.json")
    else:
        with open(out / "label_map.json", "w") as f:
            json.dump(label2id, f, indent=2)
    print("  Saved label_map.json")

    meta = {
        "model_type": "distilbert-sentence-classifier",
        "classes": list(id2label.values()),
        "num_labels": len(id2label),
        "framework": "candle-transformers",
        "input": "tokenized text (BERT tokenizer, max_length=128)",
        "output": "softmax probabilities [commissive, directive, inform, question]",
    }
    with open(out / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved model_meta.json")

    files = list(out.iterdir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    print(f"\nExport complete: {len(files)} files, {total_size / 1e6:.1f} MB total")
    for f in sorted(files):
        if f.is_file():
            print(f"  {f.name} ({f.stat().st_size / 1e3:.1f} KB)")


def export_onnx(model_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Exporting ONNX model...")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except ImportError:
        print("optimum[onnxruntime] not installed. Trying manual export...")
        _export_onnx_manual(model_path, out)
        return

    model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    onnx_path = out / "model.onnx"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    if onnx_path.exists():
        print(f"  Saved model.onnx ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    else:
        onnx_files = list(out.glob("*.onnx"))
        for f in onnx_files:
            print(f"  Saved {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    print("ONNX export complete")


def _export_onnx_manual(model_path: str, out: Path):
    import numpy as np

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    dummy = tokenizer("This is a test sentence.", return_tensors="pt", max_length=128, padding="max_length", truncation=True)

    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"  Saved model.onnx ({onnx_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export sentence classifier for Rust inference")
    parser.add_argument("--model-path", default="models/en/final")
    parser.add_argument("--output-dir", default=None, help="Defaults to model-path/candle-export")
    parser.add_argument("--onnx", action="store_true", help="Also export ONNX model")
    parser.add_argument("--onnx-only", action="store_true", help="Only export ONNX model")
    parser.add_argument("--safetensors", action="store_true", help="Use safetensors format")
    args = parser.parse_args()

    model_path = args.model_path

    if not args.onnx_only:
        pytorch_output = args.output_dir or str(Path(model_path) / "candle-export")
        export_pytorch(model_path, pytorch_output, args.safetensors)

    if args.onnx or args.onnx_only:
        onnx_output = args.output_dir or str(Path(model_path) / "onnx-export")
        export_onnx(model_path, onnx_output)


if __name__ == "__main__":
    main()
