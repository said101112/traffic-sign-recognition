from __future__ import annotations

import base64
import binascii
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from traffic_signs.inference import Prediction, PredictionResult, TrafficSignPredictor
from traffic_signs.localization import DetectionBox

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
CHECKPOINT_PATH = Path("models/traffic_recognition_best_model.pt")


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


def decode_camera_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def load_submitted_image() -> Image.Image:
    uploaded_file = request.files.get("image")
    if uploaded_file is not None and uploaded_file.filename:
        return Image.open(uploaded_file.stream).convert("RGB")

    raise ValueError("Please choose an image or capture one with the camera.")


def annotate_detection(
    image: Image.Image,
    detection_box: DetectionBox | None,
    label_text: str | None = None,
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    if detection_box is None:
        return annotated

    draw = ImageDraw.Draw(annotated)
    stroke_width = max(3, round(min(annotated.size) * 0.01))
    outline_color = "#ff7a1a"
    draw.rectangle(detection_box.as_tuple(), outline=outline_color, width=stroke_width)

    if label_text:
        font_size = max(16, round(min(annotated.size) * 0.04))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        # Using textbbox instead of deprecated getsize
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 8
        label_left = detection_box.left
        label_top = max(0, detection_box.top - text_height - (padding * 2) - 6)
        label_right = min(annotated.width, label_left + text_width + (padding * 2))
        label_bottom = label_top + text_height + (padding * 2)
        draw.rectangle((label_left, label_top, label_right, label_bottom), fill=outline_color)
        draw.text((label_left + padding, label_top + padding), label_text, fill="white", font=font)

    return annotated


def prediction_to_payload(prediction: Prediction) -> dict[str, Any]:
    return {
        "class_id": prediction.class_id,
        "label": prediction.label,
        "score": prediction.score,
        "percentage": round(prediction.score * 100, 2),
    }


def detection_box_to_payload(
    detection_box: DetectionBox | None,
    image: Image.Image,
) -> dict[str, Any] | None:
    if detection_box is None:
        return None

    return {
        "left": detection_box.left,
        "top": detection_box.top,
        "right": detection_box.right,
        "bottom": detection_box.bottom,
        "width": detection_box.width,
        "height": detection_box.height,
        "left_ratio": detection_box.left / image.width,
        "top_ratio": detection_box.top / image.height,
        "width_ratio": detection_box.width / image.width,
        "height_ratio": detection_box.height / image.height,
        "score": detection_box.score,
        "area_ratio": detection_box.area_ratio,
        "source": detection_box.source,
    }


def build_detection_message(analysis: PredictionResult) -> str:
    if analysis.detection_box is not None:
        if analysis.used_localized_crop:
            return "Traffic sign localized and the crop was used for classification."
        return (
            "Traffic sign localized, but the final prediction kept the full image because "
            "it was more reliable."
        )

    return "No clear sign contour found. Prediction used the full image."


def analyze_image(image: Image.Image, source_mode: str) -> dict[str, Any]:
    analysis = get_predictor().analyze_pil(image, top_k=3)
    predictions = analysis.predictions

    label_text = None
    if predictions:
        label_text = f"{predictions[0].label} {predictions[0].score * 100:.1f}%"

    preview_url = image_to_data_url(
        annotate_detection(
            image,
            analysis.detection_box,
            label_text=label_text,
        )
    )

    localized_preview_url = None
    if analysis.detection_box is not None:
        localized_preview_url = image_to_data_url(image.crop(analysis.detection_box.as_tuple()))

    return {
        "predictions": predictions,
        "predictions_payload": [prediction_to_payload(prediction) for prediction in predictions],
        "preview_url": preview_url,
        "localized_preview_url": localized_preview_url,
        "detection_message": build_detection_message(analysis),
        "source_mode": source_mode,
        "used_localized_crop": analysis.used_localized_crop,
        "detection_box": analysis.detection_box,
        "detection_box_payload": detection_box_to_payload(analysis.detection_box, image),
        "image_width": image.width,
        "image_height": image.height,
    }


@app.route("/")
def index():
    return render_template("landing.html")


@app.route("/app", methods=["GET", "POST"])
def app_page():
    error_message = None
    preview_url = None
    predictions = None
    localized_preview_url = None
    detection_message = None
    source_mode = None

    if request.method == "POST":
        if not CHECKPOINT_PATH.exists():
            error_message = (
                "Model checkpoint not found. Train the model first with "
                "`python train.py`."
            )
        else:
            try:
                image = load_submitted_image()
                result = analyze_image(image, source_mode="uploaded image")
                predictions = result["predictions"]
                preview_url = result["preview_url"]
                localized_preview_url = result["localized_preview_url"]
                detection_message = result["detection_message"]
                source_mode = result["source_mode"]
            except ValueError as exc:
                error_message = str(exc)
            except (UnidentifiedImageError, binascii.Error):
                error_message = "The submitted content is not a supported image."
            except Exception as exc:  # pragma: no cover - defensive UI fallback
                error_message = f"Prediction failed: {exc}"

    return render_template(
        "app_page.html",
        error_message=error_message,
        preview_url=preview_url,
        localized_preview_url=localized_preview_url,
        detection_message=detection_message,
        predictions=predictions,
        source_mode=source_mode,
    )


@app.route("/api/analyze-frame", methods=["POST"])
def analyze_frame():
    if not CHECKPOINT_PATH.exists():
        return jsonify(
            {
                "error": (
                    "Model checkpoint not found. Train the model first with `python train.py`."
                )
            }
        ), 503

    payload = request.get_json(silent=True) or {}
    image_data = str(payload.get("image", "")).strip()
    if not image_data:
        return jsonify({"error": "No webcam frame was provided."}), 400

    try:
        image = decode_camera_image(image_data)
        result = analyze_image(image, source_mode="webcam live")
        return jsonify(
            {
                "predictions": result["predictions_payload"],
                "preview_url": result["preview_url"],
                "localized_preview_url": result["localized_preview_url"],
                "detection_message": result["detection_message"],
                "source_mode": result["source_mode"],
                "used_localized_crop": result["used_localized_crop"],
                "detection_box": result["detection_box_payload"],
                "image_width": result["image_width"],
                "image_height": result["image_height"],
            }
        )
    except (UnidentifiedImageError, binascii.Error, ValueError):
        return jsonify({"error": "The webcam frame is not a supported image."}), 400
    except Exception as exc:  # pragma: no cover - defensive UI fallback
        return jsonify({"error": f"Live prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
