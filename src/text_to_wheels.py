"""CLI: read text commands from a file, classify each one and produce a
JSON file with the resulting wheel motor speeds.

Run:

    python -m src.text_to_wheels \
        --input data/examples/basic.txt \
        --output outputs/result.json \
        --classifier sklearn
"""
import argparse
import json
from pathlib import Path

from src.wheel_mapping import WHEEL_COMMANDS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate natural-language commands into wheel motor commands."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a text file with one command per line."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path where the JSON result will be written."
    )
    parser.add_argument(
        "--classifier", "-c",
        choices=["sklearn", "st"], default="sklearn",
        help="Which classifier to use (default: sklearn)."
    )
    parser.add_argument(
        "--dataset", "-d",
        default="data/training_commands.csv",
        help="Path to the training CSV (used only by the sklearn classifier)."
    )
    return parser.parse_args()


def build_classifier(name, dataset_path):
    """Instantiate and prepare the requested classifier."""
    if name == "sklearn":
        from src.classifier_sklearn import SklearnClassifier
        clf = SklearnClassifier()
        clf.train(dataset_path)
        return clf
    if name == "st":
        from src.classifier_st import SentenceTransformerClassifier
        return SentenceTransformerClassifier()
    raise ValueError(f"Unknown classifier: {name}")


def read_commands(path):
    """Return the non-empty stripped lines from the file at path."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    classifier = build_classifier(args.classifier, args.dataset)
    commands = read_commands(args.input)

    results = []
    for command in commands:
        action, confidence = classifier.predict(command)
        results.append({
            "input": command,
            "action": action,
            "wheels": WHEEL_COMMANDS[action],
            "confidence": round(confidence, 4),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(results)} commands with the '{args.classifier}' classifier.")
    print(f"Result written to {output_path}")


if __name__ == "__main__":
    main()
