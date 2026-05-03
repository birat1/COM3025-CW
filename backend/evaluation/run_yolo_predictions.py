import logging
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO
from utils import Timer, list_images, save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IMAGE_DIR = Path("../tests/coco_indoor")
OUTPUT_DIR = Path("../outputs/predictions/yolo_predictions.json")

MODEL = "yolov8s.pt"
CONF_THRESHOLD = 0.25

def run_yolo() -> None:
    """Run YOLO object detection on images and save predictions."""
    model = YOLO(MODEL)
    images = list_images(IMAGE_DIR)

    all_preds = {}

    # Process each image and run detection
    for image_dir in tqdm(images, desc="Running YOLO"):
        with Timer() as timer:
            results = model(str(image_dir), conf=CONF_THRESHOLD, verbose=False)

        # Extract detections from the results
        result = results[0]
        detections = []

        # Check if there are any detections and process them
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])

                bbox = box.xyxy[0].tolist()

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "bbox": bbox,
                })

        # Save predictions and inference time for the current image
        all_preds[image_dir.name] = {
            "detections": detections,
            "inference_time_ms": timer.elapsed_ms,
        }

    save_json(all_preds, OUTPUT_DIR)
    logger.info(f"Saved predictions to {OUTPUT_DIR}")  # noqa: G004

if __name__ == "__main__":
    run_yolo()
