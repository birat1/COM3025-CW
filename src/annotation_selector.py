import os
import json

COCO_IMAGES_DIR = "data/coco_test"
INSTANCES_FILE = "data/annotations/instances_val2017.json"
CAPTIONS_FILE = "data/annotations/captions_val2017.json"

ANN_OUTPUT = "data/annotations/selected_annotations.json"
CAP_OUTPUT = "data/annotations/selected_captions.json"

def select_coco_annotations():
    # Get images from coco_test directory
    images_files = {f for f in os.listdir(COCO_IMAGES_DIR) if f.endswith(".jpg")}

    if not images_files:
        print("No images found in the directory.")
        return

    # Map image IDs to file names
    with open(INSTANCES_FILE, "r") as f:
        instances_data = json.load(f)

    id_to_filename = {
        img["id"]: img["file_name"]
        for img in instances_data["images"]
        if img["file_name"] in images_files
    }

    # Map category IDs to category names
    category_mapping = {cat["id"]: cat["name"] for cat in instances_data["categories"]}

    # Process annotations for images in coco_test
    selected_annotations = {}
    for ann in instances_data["annotations"]:
        image_id = ann["image_id"]
        if image_id in id_to_filename:
            file_name = id_to_filename[image_id]
            label = category_mapping[ann["category_id"]]

            selected_annotations.setdefault(file_name, [])
            if label not in selected_annotations[file_name]:
                selected_annotations[file_name].append(label)

    # Process captions for images in coco_test
    with open(CAPTIONS_FILE, "r") as f:
        captions_data = json.load(f)

    selected_captions = {}
    for ann in captions_data["annotations"]:
        image_id = ann["image_id"]
        if image_id in id_to_filename:
            file_name = id_to_filename[image_id]
            selected_captions.setdefault(file_name, [])
            selected_captions[file_name].append(ann["caption"])

    # Save selected annotations and captions to JSON files
    with open(ANN_OUTPUT, "w") as f:
        json.dump(selected_annotations, f, indent=2)

    with open(CAP_OUTPUT, "w") as f:
        json.dump(selected_captions, f, indent=2)

    print(f"Processed {len(images_files)} images")

if __name__ == "__main__":
    select_coco_annotations()