import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor
from utils import Timer, list_images, save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IMAGE_DIR = Path("../tests/coco_indoor")
OUTPUT_DIR = Path("../outputs/predictions/blip_captions.json")

MODEL = "Salesforce/blip-image-captioning-large"

def run_blip() -> None:
    """Run BLIP image captioning on images and save predictions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the BLIP model and processor
    processor = BlipProcessor.from_pretrained(MODEL)
    model = BlipForConditionalGeneration.from_pretrained(MODEL).to(device)
    model.eval()

    images = list_images(IMAGE_DIR)

    all_preds = {}

    # Process each image and run captioning
    for image_dir in tqdm(images, desc="Running BLIP"):
        image = Image.open(image_dir).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad(), Timer() as timer:
            outputs = model.generate(**inputs, max_new_tokens=30)

        caption = processor.decode(outputs[0], skip_special_tokens=True)

        # Save predictions and inference time for the current image
        all_preds[image_dir.name] = {
            "caption": caption,
            "inference_time_ms": timer.elapsed_ms,
        }

    save_json(all_preds, OUTPUT_DIR)
    logger.info(f"Saved BLIP captions to {OUTPUT_DIR}")  # noqa: G004

if __name__ == "__main__":
    run_blip()
