from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from traffic_signs.model import TrafficSignNet


@dataclass
class Prediction:
    class_id: int
    label: str
    score: float


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

    def predict_file(self, image_path: Path | str, top_k: int = 3) -> list[Prediction]:
        image = Image.open(image_path).convert("RGB")
        return self.predict_pil(image, top_k=top_k)

