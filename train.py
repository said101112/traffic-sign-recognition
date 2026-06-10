from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from traffic_signs.data import (
    TrafficSignDataset,
    available_class_ids,
    read_label_names,
    scan_test_samples,
    scan_train_samples,
    split_train_val,
    summarize_dataset,
)
from traffic_signs.model import TrafficSignNet

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a traffic sign classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for quick experiments.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap for quick experiments.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.10, 0.10),
                scale=(0.90, 1.10),
                shear=8,
            ),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.20,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, eval_transform


def build_loaders(args: argparse.Namespace):
    train_samples = scan_train_samples(args.dataset_dir, drop_copy_duplicates=True)
    test_samples = scan_test_samples(args.dataset_dir)

    if args.max_train_samples is not None:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_test_samples is not None:
        test_samples = test_samples[: args.max_test_samples]

    train_samples, val_samples = split_train_val(
        train_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    class_ids = available_class_ids(train_samples + val_samples + test_samples)
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}

    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = TrafficSignDataset(train_samples, class_to_index, transform=train_transform)
    val_dataset = TrafficSignDataset(val_samples, class_to_index, transform=eval_transform)
    test_dataset = TrafficSignDataset(test_samples, class_to_index, transform=eval_transform)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader, class_ids, train_samples, val_samples, test_samples


def make_class_weights(samples, class_to_index: dict[int, int], device: torch.device) -> torch.Tensor:
    counts = Counter(sample.class_id for sample in samples)
    weights = torch.ones(len(class_to_index), dtype=torch.float32)
    max_count = max(counts.values())
    for class_id, index in class_to_index.items():
        weights[index] = max_count / counts[class_id]
    return weights.to(device)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_examples += images.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / len(all_targets),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_f1": f1_score(all_targets, all_predictions, average="macro"),
        "targets": all_targets,
        "predictions": all_predictions,
    }


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    class_ids: list[int],
    class_names: dict[int, str],
    image_size: int,
    history: list[dict[str, float]],
    metrics: dict[str, float],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "class_ids": class_ids,
        "class_names": {str(key): value for key, value in class_names.items()},
        "image_size": image_size,
        "normalization": {"mean": MEAN, "std": STD},
        "history": history,
        "metrics": metrics,
    }
    torch.save(payload, output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_report = summarize_dataset(args.dataset_dir)
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(dataset_report, indent=2),
        encoding="utf-8",
    )

    train_loader, val_loader, test_loader, class_ids, train_samples, val_samples, test_samples = build_loaders(args)
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    label_names = read_label_names(args.dataset_dir)

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Classes: {len(class_ids)}")

    model = TrafficSignNet(num_classes=len(class_ids)).to(device)
    class_weights = make_class_weights(train_samples, class_to_index, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val_accuracy = 0.0
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train_loss={train_metrics['loss']:.4f} "
            f"- train_acc={train_metrics['accuracy']:.4f} "
            f"- val_loss={val_metrics['loss']:.4f} "
            f"- val_acc={val_metrics['accuracy']:.4f} "
            f"- val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state_dict)
    model.to(device)

    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    elapsed_seconds = time.time() - start_time

    target_names = [label_names[class_id] for class_id in class_ids]
    report = classification_report(
        test_metrics["targets"],
        test_metrics["predictions"],
        labels=list(range(len(class_ids))),
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    metrics_summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "final_val_accuracy": val_metrics["accuracy"],
        "final_val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "elapsed_seconds": elapsed_seconds,
        "num_classes": len(class_ids),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
    }

    save_checkpoint(
        args.output_dir / "best_model.pt",
        model,
        class_ids,
        label_names,
        args.image_size,
        history,
        metrics_summary,
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Training complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

