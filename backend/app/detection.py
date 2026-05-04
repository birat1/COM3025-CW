"""Module for object detection."""
from pathlib import Path
from typing import Any

import cv2

from app.models import yolo_model


def get_spatial_detection(box: Any, image_width: int, image_height: int) -> tuple[str, str, float]:  # noqa: ANN401
    """Estimate the spatial position and rough proximity of a detected object based on its bounding box."""
    # Extract bounding box coordinates
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # Calculate the center of the bounding box and its area relative to the image
    box_centre_x = (x1 + x2) / 2
    box_area = (x2 - x1) * (y2 - y1)
    image_area = image_width * image_height

    size_ratio = box_area / image_area if image_area > 0 else 0

    # Determine horizontal position
    if box_centre_x < image_width * 0.33:
        position = "left"
    elif box_centre_x > image_width * 0.66:
        position = "right"
    else:
        position = "centre"

    # Determine proximity based on size ratio
    if size_ratio > 0.40:
        proximity = "close"
    elif size_ratio > 0.10:
        proximity = "ahead"
    else:
        proximity = "far"

    return position, proximity, size_ratio

def run_detection(image_path: str, confidence: float = 0.25, output_path: str | None = None) -> list[dict]:
    """Run YOLO object detection and saves annotated image."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")  # noqa: EM102, TRY003

    results = yolo_model(image_path, conf=confidence, verbose=False)
    result = results[0]

    height, width = result.orig_shape

    # Process detections and extract spatial information
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_id = int(box.cls)
        class_name = result.names[class_id]
        conf = float(box.conf)

        position, proximity, size_ratio = get_spatial_detection(box, width, height)

        detections.append({
            "object": class_name,
            "class_id": class_id,
            "confidence": round(conf, 2),
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
            "position": position,
            "proximity": proximity,
            "size_ratio": round(size_ratio, 4),
            "is_central": position == "centre",
            "is_close": proximity == "close"
        })

    detections.sort(key=lambda x: (x["is_close"], x["is_central"], x["confidence"]), reverse=True)

    # Save annotated image if output path is provided
    if output_path:
        annotated_image = result.plot()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), annotated_image)

        if not success:
            raise RuntimeError(f"Failed to save annotated image to: {output_path}")  # noqa: EM102, TRY003

    return detections
