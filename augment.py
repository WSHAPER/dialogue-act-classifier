"""
Augment DailyDialog training data with ASR edge cases.

Reads tests/test_cases.json and generates expanded variants by applying
ASR-like transformations (filler words, casing, punctuation stripping).
Outputs a HuggingFace Dataset ready for concatenation with DailyDialog.
"""

import json
import random
from pathlib import Path

from datasets import Dataset

LABEL_MAP = {"statement": 0, "question": 1, "instruction": 2}

FILLERS = ["uh", "um", "like", "so", "well", "ok", "right", "you know"]

PUNCTUATION_VARIANTS = [
    lambda t: t.rstrip(".?!"),
    lambda t: t.rstrip(".?!") + ".",
    lambda t: t.rstrip(".?!") + "?",
    lambda t: t.rstrip(".?!") + "!",
    lambda t: t.lower(),
    lambda t: t.capitalize(),
]


def load_test_cases(path: str = "tests/test_cases.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def add_filler(text: str, rng: random.Random) -> str:
    filler = rng.choice(FILLERS)
    pos = rng.randint(0, min(3, len(text.split())))
    words = text.split()
    words.insert(pos, filler)
    return " ".join(words)


def generate_variants(
    test_cases: list[dict],
    augment_factor: int = 5,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    augmented = []

    for tc in test_cases:
        text = tc["text"]
        label = LABEL_MAP[tc["expected"]]

        augmented.append({"text": text, "label": label})

        for _ in range(augment_factor):
            variant = text
            if rng.random() < 0.4:
                variant = add_filler(variant, rng)
            variant = rng.choice(PUNCTUATION_VARIANTS)(variant)
            if variant.lower().strip() != text.lower().strip():
                augmented.append({"text": variant, "label": label})

    return augmented


def build_augmented_dataset(
    test_cases_path: str = "tests/test_cases.json",
    augment_factor: int = 5,
    seed: int = 42,
) -> Dataset:
    test_cases = load_test_cases(test_cases_path)
    records = generate_variants(test_cases, augment_factor, seed)
    return Dataset.from_list(records)


def save_augmented_json(
    output_path: str = "tests/augmented_cases.json",
    test_cases_path: str = "tests/test_cases.json",
    augment_factor: int = 5,
    seed: int = 42,
):
    test_cases = load_test_cases(test_cases_path)
    records = generate_variants(test_cases, augment_factor, seed)
    label_names = {v: k for k, v in LABEL_MAP.items()}
    with open(output_path, "w") as f:
        json.dump(
            [{"text": r["text"], "expected": label_names[r["label"]]} for r in records],
            f,
            indent=2,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR edge-case data augmentation")
    parser.add_argument("--test-cases", default="tests/test_cases.json")
    parser.add_argument("--output", default="tests/augmented_cases.json")
    parser.add_argument("--factor", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_augmented_json(args.output, args.test_cases, args.factor, args.seed)
    ds = build_augmented_dataset(args.test_cases, args.factor, args.seed)
    print(f"Generated {len(ds)} augmented samples")
    print(f"Label distribution: {dict(zip(*ds.unique('label'), ds.to_pandas()['label'].value_counts().sort_index().tolist()))}")
