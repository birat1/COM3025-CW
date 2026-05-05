import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from utils import Timer, list_images, save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IMAGE_DIR = Path("../tests/coco_indoor")
OUTPUT_DIR = Path("../outputs/predictions/git_captions.json")

MODEL = "microsoft/git-base-coco"

def run_git() -> None:
    """Run GIT image captioning on images and save predictions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the GIT model and processor
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()

    images = list_images(IMAGE_DIR)

    all_preds = {}

    # Process each image and run captioning
    for image_dir in tqdm(images, desc="Running GIT"):
        image = Image.open(image_dir).convert("RGB")

        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad(), Timer() as timer:
            outputs = model.generate(
                pixel_values=pixel_values,
                max_length=30,
                num_beams=4,
            )

        caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        # Save predictions and inference time for the current image
        all_preds[image_dir.name] = {
            "caption": caption,
            "inference_time_ms": timer.elapsed_ms,
        }

    save_json(all_preds, OUTPUT_DIR)
    logger.info(f"Saved GIT captions to {OUTPUT_DIR}")  # noqa: G004

if __name__ == "__main__":
    run_git()
