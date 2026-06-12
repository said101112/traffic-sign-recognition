from __future__ import annotations

import argparse
from pathlib import Path

from traffic_signs.inference import TrafficSignPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a traffic sign class for one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the image to classify.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/traffic_recognition_best_model.pt"),
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = TrafficSignPredictor(args.checkpoint)
    predictions = predictor.predict_file(args.image, top_k=args.top_k)
    for prediction in predictions:
        print(
            f"class_id={prediction.class_id} "
            f"score={prediction.score:.4f} "
            f"label={prediction.label}"
        )


if __name__ == "__main__":
    main()
