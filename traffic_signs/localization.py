from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectionBox:
    left: int
    top: int
    right: int
    bottom: int
    score: float
    area_ratio: float
    source: str

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.GaussianBlur(cleaned, (5, 5), 0)


def _score_candidate(
    *,
    contour: np.ndarray,
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    source: str,
) -> DetectionBox | None:
    image_area = float(image_width * image_height)
    area = cv2.contourArea(contour)
    if area <= image_area * 0.002:
        return None

    x, y, width, height = cv2.boundingRect(contour)
    if width < 20 or height < 20:
        return None

    area_ratio = (width * height) / image_area

    aspect_ratio = width / max(height, 1)
    if not 0.45 <= aspect_ratio <= 1.8:
        return None

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    if solidity < 0.45:
        return None

    bbox_area = float(width * height)
    fill_ratio = area / bbox_area if bbox_area > 0 else 0.0

    mask_crop = mask[y : y + height, x : x + width]
    mask_coverage = float(mask_crop.mean() / 255.0) if mask_crop.size else 0.0
    border_hits = sum(
        [
            x <= 2,
            y <= 2,
            (x + width) >= image_width - 2,
            (y + height) >= image_height - 2,
        ]
    )

    if area_ratio >= 0.985 and border_hits == 4 and mask_coverage < 0.34:
        return None

    center_x = x + (width / 2.0)
    center_y = y + (height / 2.0)
    dx = (center_x - (image_width / 2.0)) / max(image_width / 2.0, 1.0)
    dy = (center_y - (image_height / 2.0)) / max(image_height / 2.0, 1.0)
    center_penalty = (dx * dx + dy * dy) ** 0.5

    if fill_ratio < 0.08 and mask_coverage < 0.12:
        return None

    score = (
        area_ratio * 3.7
        + fill_ratio * 1.4
        + mask_coverage * 1.1
        + solidity * 0.6
        - max(0.0, area_ratio - 0.9) * 1.8
        - center_penalty * 0.45
    )
    return DetectionBox(
        left=x,
        top=y,
        right=x + width,
        bottom=y + height,
        score=score,
        area_ratio=area_ratio,
        source=source,
    )


def _score_mask_bbox(
    *,
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    source: str,
) -> DetectionBox | None:
    points = cv2.findNonZero(mask)
    if points is None:
        return None

    x, y, width, height = cv2.boundingRect(points)
    if width < 20 or height < 20:
        return None

    image_area = float(image_width * image_height)
    area_ratio = (width * height) / image_area
    aspect_ratio = width / max(height, 1)
    if not 0.45 <= aspect_ratio <= 1.8:
        return None

    mask_crop = mask[y : y + height, x : x + width]
    mask_coverage = float(mask_crop.mean() / 255.0) if mask_crop.size else 0.0
    if mask_coverage < 0.16:
        return None

    border_hits = sum(
        [
            x <= 2,
            y <= 2,
            (x + width) >= image_width - 2,
            (y + height) >= image_height - 2,
        ]
    )
    if area_ratio >= 0.99 and border_hits == 4 and mask_coverage < 0.4:
        return None

    center_x = x + (width / 2.0)
    center_y = y + (height / 2.0)
    dx = (center_x - (image_width / 2.0)) / max(image_width / 2.0, 1.0)
    dy = (center_y - (image_height / 2.0)) / max(image_height / 2.0, 1.0)
    center_penalty = (dx * dx + dy * dy) ** 0.5

    score = (
        area_ratio * 4.0
        + mask_coverage * 1.8
        - max(0.0, area_ratio - 0.94) * 1.4
        - center_penalty * 0.35
    )
    return DetectionBox(
        left=x,
        top=y,
        right=x + width,
        bottom=y + height,
        score=score,
        area_ratio=area_ratio,
        source=f"{source}-union",
    )


def _pad_box(
    box: DetectionBox,
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.12,
) -> DetectionBox:
    pad_x = max(8, int(box.width * padding_ratio))
    pad_y = max(8, int(box.height * padding_ratio))
    left = max(0, box.left - pad_x)
    top = max(0, box.top - pad_y)
    right = min(image_width, box.right + pad_x)
    bottom = min(image_height, box.bottom + pad_y)
    padded_area_ratio = ((right - left) * (bottom - top)) / float(image_width * image_height)
    return DetectionBox(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        score=box.score,
        area_ratio=padded_area_ratio,
        source=box.source,
    )


def _find_color_candidates(hsv_image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv_image, (0, 70, 60), (12, 255, 255)),
        cv2.inRange(hsv_image, (165, 70, 60), (180, 255, 255)),
    )
    blue_mask = cv2.inRange(hsv_image, (90, 80, 50), (138, 255, 255))
    yellow_mask = cv2.inRange(hsv_image, (14, 80, 80), (38, 255, 255))
    return [
        ("red", _clean_mask(red_mask)),
        ("blue", _clean_mask(blue_mask)),
        ("yellow", _clean_mask(yellow_mask)),
    ]


def _find_edge_fallback(rgb_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 70, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(edges, kernel, iterations=1)


def detect_traffic_sign(image: Image.Image) -> DetectionBox | None:
    rgb_image = np.asarray(image.convert("RGB"))
    image_height, image_width = rgb_image.shape[:2]
    if image_height == 0 or image_width == 0:
        return None

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    best_candidate: DetectionBox | None = None

    for source, mask in _find_color_candidates(hsv_image):
        source_best_candidate: DetectionBox | None = None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            candidate = _score_candidate(
                contour=contour,
                mask=mask,
                image_width=image_width,
                image_height=image_height,
                source=source,
            )
            if candidate is None:
                continue
            if source_best_candidate is None or candidate.score > source_best_candidate.score:
                source_best_candidate = candidate

        if source_best_candidate is not None:
            if best_candidate is None or source_best_candidate.score > best_candidate.score:
                best_candidate = source_best_candidate
            continue

        union_candidate = _score_mask_bbox(
            mask=mask,
            image_width=image_width,
            image_height=image_height,
            source=source,
        )
        if union_candidate is not None:
            if best_candidate is None or union_candidate.score > best_candidate.score:
                best_candidate = union_candidate

    if best_candidate is None:
        edge_mask = _find_edge_fallback(rgb_image)
        contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            candidate = _score_candidate(
                contour=contour,
                mask=edge_mask,
                image_width=image_width,
                image_height=image_height,
                source="edge",
            )
            if candidate is None:
                continue
            if best_candidate is None or candidate.score > best_candidate.score:
                best_candidate = candidate

    if best_candidate is None:
        return None

    return _pad_box(best_candidate, image_width=image_width, image_height=image_height)
