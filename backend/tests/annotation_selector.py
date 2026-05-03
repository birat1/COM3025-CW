"""Script to select annotations and captions for images in coco_indoor directory."""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COCO_IMAGES_DIR = Path("coco_indoor")

INSTANCES_FILE = Path("coco_indoor/instances_val2017.json")
CAPTIONS_FILE = Path("coco_indoor/captions_val2017.json")

ANN_OUTPUT = COCO_IMAGES_DIR / "selected_instances.json"
CAP_OUTPUT = COCO_IMAGES_DIR / "selected_captions.json"

def load_json(directory: Path) -> dict:
    """Load JSON data from a file."""
    with Path.open(directory, "r") as f:
        return json.load(f)

def save_json(data: dict, output_path: Path) -> None:
    """Save data to a JSON file."""
    with Path.open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def get_image_files() -> set:
    """Get a set of image file names from the coco_indoor directory."""
    valid_extensions = {".jpg", ".jpeg", ".png"}

    return {
        file.name
        for file in COCO_IMAGES_DIR.iterdir()
        if file.suffix.lower() in valid_extensions
    }

def select_coco_annotations() -> None:
    """Select annotations and captions for images in coco_indoor directory."""
    image_files = get_image_files()
    if not image_files:
        logger.info("No images found in the directory.")
        return

    if not INSTANCES_FILE.exists():
        raise FileNotFoundError(f"Instances file not found: {INSTANCES_FILE}")  # noqa: EM102, TRY003

    if not CAPTIONS_FILE.exists():
        raise FileNotFoundError(f"Captions file not found: {CAPTIONS_FILE}")  # noqa: EM102, TRY003

    instances_data = load_json(INSTANCES_FILE)
    captions_data = load_json(CAPTIONS_FILE)

    # Create a mapping from category ID to category name for easy lookup
    category_mapping = {
        category["id"]: category["name"]
        for category in instances_data["categories"]
    }

    selected_instances = {}

    for img in instances_data["images"]:
        file_name = img["file_name"]

        if file_name not in image_files:
            continue

        # Initialise the selected instance data for this image
        selected_instances[file_name] = {
            "image_id": img["id"],
            "file_name": file_name,
            "width": img["width"],
            "height": img["height"],
            "objects": [],
        }

    image_id_to_file_name = {
        image_data["image_id"]: file_name
        for file_name, image_data in selected_instances.items()
    }

    # Process instance annotations and associate them with the corresponding images
    for annotation in instances_data["annotations"]:
        image_id = annotation["image_id"]

        if image_id not in image_id_to_file_name:
            continue

        file_name = image_id_to_file_name[image_id]
        category_id = annotation["category_id"]
        label = category_mapping.get(category_id, "unknown")

        # Convert COCO bbox format (x, y, width, height) to (x_min, y_min, x_max, y_max)
        x, y, w, h = annotation["bbox"]
        bbox_xyxy = [x, y, x + w, y + h]

        # Append the annotation details to the corresponding image's objects list
        selected_instances[file_name]["objects"].append({
            "annotation_id": annotation["id"],
            "category_id": category_id,
            "label": label,
            "bbox_coco": annotation["bbox"],
            "bbox_xyxy": bbox_xyxy,
            "area": annotation.get("area"),
            "iscrowd": annotation.get("iscrowd", 0),
        })

    selected_captions = {
        file_name: []
        for file_name in selected_instances
    }

    # Process caption annotations and associate them with the corresponding images
    for annotation in captions_data["annotations"]:
        image_id = annotation["image_id"]

        if image_id not in image_id_to_file_name:
            continue

        file_name = image_id_to_file_name[image_id]
        selected_captions[file_name].append(annotation["caption"])

    save_json(selected_instances, ANN_OUTPUT)
    save_json(selected_captions, CAP_OUTPUT)

    logger.info(f"Processed {len(selected_instances)} images")  # noqa: G004
    logger.info(f"Saved instance annotations to {ANN_OUTPUT}")  # noqa: G004
    logger.info(f"Saved captions to {CAP_OUTPUT}")  # noqa: G004

if __name__ == "__main__":
    select_coco_annotations()
