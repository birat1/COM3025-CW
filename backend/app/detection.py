"""Module for object detection."""
from app.models import yolo_model

def get_spatial_detection(box, image_width, image_height):
    """Determines the spatial position and distance of a detected object based on its bounding box."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    box_centre_x = (x1 + x2) / 2
    box_area = (x2 - x1) * (y2 - y1)
    image_area = image_width * image_height
    size_ratio = box_area / image_area

    if box_centre_x < image_width * 0.33:
        position = "left"
    elif box_centre_x > image_width * 0.66:
        position = "right"
    else:
        position = "centre"

    if size_ratio > 0.40:
        distance = "close up"
    elif size_ratio > 0.10:
        distance = "mid-range"
    else:
        distance = "far away"

    return position, distance

def run_detection(image_path, confidence=0.5):
    """Runs object detection on the given image and returns a list of detected objects with their spatial information."""
    results = yolo_model(image_path, conf=confidence, verbose=False)
    result = results[0]

    height, width = result.orig_shape

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

    return detections
