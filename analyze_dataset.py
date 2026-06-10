from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_signs.data import summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the traffic sign dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Dataset"),
        help="Path to the extracted dataset folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dataset_report.json"),
        help="Where to write the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_dataset(args.dataset_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Dataset analysis complete")
    print(f"Dataset folder: {report['dataset_dir']}")
    print(f"Labels: {report['label_count']}")
    print(f"Raw train images found: {report['train_image_count_raw']}")
    print(f"Deduped train images used for training: {report['train_image_count_deduped']}")
    print(f"Test images found: {report['test_image_count']}")
    print(f"Train classes available: {report['train_class_count']}")
    print(f"Train.csv missing paths: {report['train_csv_missing_paths']}")
    print(f"Test.csv missing paths: {report['test_csv_missing_paths']}")
    print("Smallest classes:", report["smallest_train_classes"])
    print("Largest classes:", report["largest_train_classes"])
    if report["notes"]:
        print("Notes:")
        for note in report["notes"]:
            print(f"- {note}")
    print(f"Saved JSON report to: {args.output}")


if __name__ == "__main__":
    main()

