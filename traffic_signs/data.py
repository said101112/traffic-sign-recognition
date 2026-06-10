from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
COPY_SUFFIX_PATTERN = re.compile(r" - Copie(?=\.)")


@dataclass(frozen=True)
class Sample:
    path: Path
    class_id: int


def read_label_names(dataset_dir: Path) -> dict[int, str]:
    labels_path = dataset_dir / "labels.csv"
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {int(row["ClassId"]): row["Name"] for row in rows}


def read_train_csv_rows(dataset_dir: Path) -> list[dict[str, str]]:
    with (dataset_dir / "Train.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_test_csv_rows(dataset_dir: Path) -> list[dict[str, str]]:
    with (dataset_dir / "Test.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric_class_dirs(train_dir: Path) -> list[Path]:
    dirs = [path for path in train_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(dirs, key=lambda path: int(path.name))


def _dedupe_copy_variants(paths: list[Path]) -> list[Path]:
    by_canonical_name: dict[str, Path] = {}
    for path in paths:
        canonical_name = COPY_SUFFIX_PATTERN.sub("", path.name)
        current = by_canonical_name.get(canonical_name)
        if current is None:
            by_canonical_name[canonical_name] = path
            continue
        current_is_copy = COPY_SUFFIX_PATTERN.search(current.name) is not None
        candidate_is_copy = COPY_SUFFIX_PATTERN.search(path.name) is not None
        if current_is_copy and not candidate_is_copy:
            by_canonical_name[canonical_name] = path
    return sorted(by_canonical_name.values(), key=lambda path: path.name)


def scan_train_samples(dataset_dir: Path, drop_copy_duplicates: bool = True) -> list[Sample]:
    train_dir = dataset_dir / "Train"
    samples: list[Sample] = []

    for class_dir in _numeric_class_dirs(train_dir):
        class_id = int(class_dir.name)
        files = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ],
            key=lambda path: path.name,
        )
        if drop_copy_duplicates:
            files = _dedupe_copy_variants(files)
        samples.extend(Sample(path=path, class_id=class_id) for path in files)

    return samples


def scan_test_samples(dataset_dir: Path) -> list[Sample]:
    rows = read_test_csv_rows(dataset_dir)
    samples: list[Sample] = []
    for row in rows:
        path = dataset_dir / row["Path"]
        if path.exists():
            samples.append(Sample(path=path, class_id=int(row["ClassId"])))
    return samples


def available_class_ids(samples: list[Sample]) -> list[int]:
    return sorted({sample.class_id for sample in samples})


def split_train_val(
    samples: list[Sample],
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    indices = list(range(len(samples)))
    labels = [sample.class_id for sample in samples]
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train_samples = [samples[index] for index in train_indices]
    val_samples = [samples[index] for index in val_indices]
    return train_samples, val_samples


class TrafficSignDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        class_to_index: dict[int, int],
        transform=None,
    ) -> None:
        self.samples = samples
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_index[sample.class_id]


def summarize_dataset(dataset_dir: Path) -> dict[str, Any]:
    label_names = read_label_names(dataset_dir)
    train_csv_rows = read_train_csv_rows(dataset_dir)
    test_csv_rows = read_test_csv_rows(dataset_dir)
    raw_train_samples = scan_train_samples(dataset_dir, drop_copy_duplicates=False)
    train_samples = scan_train_samples(dataset_dir, drop_copy_duplicates=True)
    test_samples = scan_test_samples(dataset_dir)

    train_csv_missing = sum(
        1 for row in train_csv_rows if not (dataset_dir / row["Path"]).exists()
    )
    test_csv_missing = sum(
        1 for row in test_csv_rows if not (dataset_dir / row["Path"]).exists()
    )

    raw_counts = Counter(sample.class_id for sample in raw_train_samples)
    deduped_counts = Counter(sample.class_id for sample in train_samples)
    duplicate_extras = {
        class_id: raw_counts[class_id] - deduped_counts[class_id]
        for class_id in raw_counts
        if raw_counts[class_id] != deduped_counts[class_id]
    }
    smallest_classes = sorted(deduped_counts.items(), key=lambda item: (item[1], item[0]))[:10]
    largest_classes = sorted(
        deduped_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]

    image_formats = defaultdict(int)
    for sample in raw_train_samples:
        image_formats[sample.path.suffix.lower()] += 1

    test_formats = defaultdict(int)
    for sample in test_samples:
        test_formats[sample.path.suffix.lower()] += 1

    notes: list[str] = []
    if train_csv_missing == len(train_csv_rows):
        notes.append(
            "Train.csv paths do not match the extracted training images and should not be used "
            "as the canonical training source."
        )
    if duplicate_extras:
        notes.append(
            "Some training folders contain explicit '- Copie' duplicates. The training pipeline "
            "drops those duplicates automatically."
        )

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "label_count": len(label_names),
        "label_names": label_names,
        "train_csv_rows": len(train_csv_rows),
        "test_csv_rows": len(test_csv_rows),
        "train_csv_missing_paths": train_csv_missing,
        "test_csv_missing_paths": test_csv_missing,
        "train_image_count_raw": len(raw_train_samples),
        "train_image_count_deduped": len(train_samples),
        "test_image_count": len(test_samples),
        "train_class_count": len(deduped_counts),
        "train_class_distribution": dict(sorted(deduped_counts.items())),
        "duplicate_extras_by_class": dict(sorted(duplicate_extras.items())),
        "smallest_train_classes": smallest_classes,
        "largest_train_classes": largest_classes,
        "train_image_formats": dict(sorted(image_formats.items())),
        "test_image_formats": dict(sorted(test_formats.items())),
        "notes": notes,
    }

