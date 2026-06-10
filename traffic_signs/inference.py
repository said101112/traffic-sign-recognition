from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from traffic_signs.model import TrafficSignNet
from traffic_signs.localization import DetectionBox, detect_traffic_sign


@dataclass
class Prediction:
    class_id: int
    label: str
    score: float


@dataclass
class PredictionResult:
    predictions: list[Prediction]
    detection_box: DetectionBox | None
    used_localized_crop: bool
    full_image_predictions: list[Prediction]
    localized_predictions: list[Prediction] | None


class TrafficSignPredictor:
    def __init__(self, checkpoint_path: Path | str) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        self.class_ids: list[int] = checkpoint["class_ids"]
        self.class_names = {int(key): value for key, value in checkpoint["class_names"].items()}
        self.image_size: int = checkpoint["image_size"]

        model = TrafficSignNet(num_classes=len(self.class_ids))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model

        mean = checkpoint["normalization"]["mean"]
        std = checkpoint["normalization"]["std"]
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def predict_pil(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            scores, indices = torch.topk(
                probabilities,
                k=min(top_k, len(self.class_ids)),
                dim=1,
            )

        predictions: list[Prediction] = []
        for score, index in zip(scores[0].tolist(), indices[0].tolist()):
            class_id = self.class_ids[index]
            predictions.append(
                Prediction(
                    class_id=class_id,
                    label=self.class_names[class_id],
                    score=score,
                )
            )
        return predictions

    @staticmethod
    def _should_use_localized_crop(
        full_image_predictions: list[Prediction],
        localized_predictions: list[Prediction],
        detection_box: DetectionBox,
    ) -> bool:
        full_top = full_image_predictions[0]
        localized_top = localized_predictions[0]

        if localized_top.label == full_top.label:
            if localized_top.score >= full_top.score - 0.05:
                return True
            return detection_box.area_ratio < 0.28 and localized_top.score >= full_top.score * 0.8

        if detection_box.area_ratio < 0.22 and localized_top.score >= full_top.score + 0.08:
            return True

        return localized_top.score >= full_top.score + 0.18

    def analyze_pil(self, image: Image.Image, top_k: int = 3) -> PredictionResult:
        full_image_predictions = self.predict_pil(image, top_k=top_k)
        detection_box = detect_traffic_sign(image)
        localized_predictions: list[Prediction] | None = None
        used_localized_crop = False
        predictions = full_image_predictions

        if detection_box is not None:
            localized_image = image.crop(detection_box.as_tuple())
            localized_predictions = self.predict_pil(localized_image, top_k=top_k)
            used_localized_crop = self._should_use_localized_crop(
                full_image_predictions,
                localized_predictions,
                detection_box,
            )
            if used_localized_crop:
                predictions = localized_predictions

        return PredictionResult(
            predictions=predictions,
            detection_box=detection_box,
            used_localized_crop=used_localized_crop,
            full_image_predictions=full_image_predictions,
            localized_predictions=localized_predictions,
        )

    def predict_file(self, image_path: Path | str, top_k: int = 3) -> list[Prediction]:
        image = Image.open(image_path).convert("RGB")
        return self.predict_pil(image, top_k=top_k)
