"""Module for object detection."""
from pathlib import Path
import cv2

from app.models import yolo_model

def get_spatial_detection(box: any, image_width: int, image_height: int) -> tuple[str, str]:
    """Determines the spatial position and distance of a detected object based on its bounding box."""
    # Extract bounding box coordinates
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # Calculate the center of the bounding box and its area relative to the image
    box_centre_x = (x1 + x2) / 2
    box_area = (x2 - x1) * (y2 - y1)
    image_area = image_width * image_height
    size_ratio = box_area / image_area

    # Determine horizontal position
    if box_centre_x < image_width * 0.33:
        position = "left"
    elif box_centre_x > image_width * 0.66:
        position = "right"
    else:
        position = "centre"

    # Determine distance based on size ratio
    if size_ratio > 0.40:
        distance = "close up"
    elif size_ratio > 0.10:
        distance = "mid-range"
    else:
        distance = "far away"

    return position, distance

def run_detection(image_path: str, confidence: float = 0.5, output_path: str | None = None) -> list:
    """Runs object detection and saves annotated image."""
    results = yolo_model(image_path, conf=confidence, verbose=False)
    result = results[0]

    height, width = result.orig_shape

    # Process detections and extract spatial information
    detections = []
    for box in result.boxes:
        class_name = result.names[int(box.cls)]
        conf = float(box.conf)

        position, distance = get_spatial_detection(box, width, height)

        detections.append({
            "object": class_name,
            "confidence": round(conf, 2),
            "position": position,
            "distance": distance
        })

    # Save annotated image if output path is provided
    if output_path:
        annotated_image = result.plot()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), annotated_image)

    return detections
