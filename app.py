from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from traffic_signs.inference import TrafficSignPredictor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
CHECKPOINT_PATH = Path("artifacts/best_model.pt")


@lru_cache(maxsize=1)
def get_predictor() -> TrafficSignPredictor:
    return TrafficSignPredictor(CHECKPOINT_PATH)


def image_to_data_url(image: Image.Image) -> str:
    preview = image.convert("RGB").copy()
    preview.thumbnail((420, 420))
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None
    preview_url = None
    predictions = None

    if request.method == "POST":
        if not CHECKPOINT_PATH.exists():
            error_message = (
                "Model checkpoint not found. Train the model first with "
                "`python train.py`."
            )
        else:
            uploaded_file = request.files.get("image")
            if uploaded_file is None or uploaded_file.filename == "":
                error_message = "Please choose an image before submitting."
            else:
                try:
                    image = Image.open(uploaded_file.stream).convert("RGB")
                    preview_url = image_to_data_url(image)
                    predictions = get_predictor().predict_pil(image, top_k=3)
                except UnidentifiedImageError:
                    error_message = "The uploaded file is not a supported image."
                except Exception as exc:  # pragma: no cover - defensive UI fallback
                    error_message = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        error_message=error_message,
        preview_url=preview_url,
        predictions=predictions,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

