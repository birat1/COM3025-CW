import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor
from utils import Timer, list_images, save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IMAGE_DIR = Path("../tests/coco_indoor")
OUTPUT_DIR = Path("../outputs/predictions/detr_predictions.json")

MODEL = "facebook/detr-resnet-50"
CONF_THRESHOLD = 0.35

def run_detr() -> None:
    """Run DETR object detection on images and save predictions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the DETR model and processor
    processor = DetrImageProcessor.from_pretrained(MODEL)
    model = DetrForObjectDetection.from_pretrained(MODEL).to(device)
    model.eval()

    images = list_images(IMAGE_DIR)
    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")  # noqa: EM102, TRY003

    all_preds = {}

    # Process each image and run detection
    for image_dir in tqdm(images, desc="Running DETR"):
        image = Image.open(image_dir).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad(), Timer() as timer:
            outputs = model(**inputs)

        # Convert DETR normalised boxes to image size
        target_sizes = torch.tensor([image.size[::-1]], device=device)

        # Post-process the outputs to extract detections with confidence above threshold
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=CONF_THRESHOLD,
        )[0]

        detections = []

        # Extract labels, confidence scores, and bounding boxes from the results
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            label = model.config.id2label[int(label_id)]
            confidence = float(score)

            bbox = [float(value) for value in box.tolist()]

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
            })

        all_preds[image_dir.name] = {
            "detections": detections,
            "inference_time_ms": timer.elapsed_ms,
        }

    save_json(all_preds, OUTPUT_DIR)
    logger.info(f"Saved predictions to {OUTPUT_DIR}")  # noqa: G004

if __name__ == "__main__":
    run_detr()
