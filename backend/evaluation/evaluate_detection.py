import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH = "../tests/coco_indoor/selected_instances.json"
PREDICTIONS = "../outputs/predictions/yolo_predictions.json"
OUTPUT_DIR = "../outputs/metrics/yolo_metrics.json"
OUTPUT_CSV = "../outputs/metrics/yolo_detection_per_image.csv"

CONF_THRESHOLD = 0.25
SKIP_CROWD = True

def extract_true_labels(ground_truth: dict) -> list[str]:
    """Extract true labels from a ground truth entry."""
    objects = ground_truth.get("objects", [])
    return sorted({obj["label"] for obj in objects})

def extract_predicted_labels(predictions: dict) -> list[str]:
    """Extract predicted labels from a prediction entry."""
    detections = predictions.get("detections", [])
    return [detection["label"] for detection in detections]

def build_label_mapping(ground_truth: dict, predictions: dict) -> dict[str, int]:
    """Build a mapping for torchmetrics."""
    labels = set()

    # Collect all unique labels from both ground truth and predictions
    for entry in ground_truth.values():
        labels.update(extract_true_labels(entry))

    for entry in predictions.values():
        labels.update(extract_predicted_labels(entry))

    # Create a consistent mapping of labels to integer IDs
    return {label: idx for idx, label in enumerate(sorted(labels))}

def build_map_inputs(
        ground_truth: dict,
        predictions: dict,
        label_to_id: dict[str, int],
    ) -> tuple[list[dict], list[dict]]:
    """Convert selected_instances and yolo_predictions into torchmetrics format."""
    preds = []
    targets = []

    # Process each image in the ground truth and prepare corresponding predictions
    for image_name, entry in ground_truth.items():
        target_boxes = []
        target_labels = []

        # Extract bounding boxes and labels from the ground truth
        for obj in entry.get("objects", []):
            if SKIP_CROWD and obj.get("iscrowd", 0) == 1:
                continue

            label = obj["label"]
            bbox = obj.get("bbox_xyxy")

            if bbox is None:
                continue

            if label not in label_to_id:
                continue

            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                continue

            target_boxes.append([x1, y1, x2, y2])
            target_labels.append(label_to_id[label])

        # Extract bounding boxes, labels, and confidence scores from the predictions
        pred_entry = predictions.get(
            image_name,
            {
                "detections": [],
                "inference_time_ms": None,
            },
        )

        pred_boxes = []
        pred_scores = []
        pred_labels = []

        # Process each detection in the predictions
        for detection in pred_entry.get("detections", []):
            label = detection["label"]
            confidence = detection.get("confidence", 0)
            bbox = detection.get("bbox")

            if confidence < CONF_THRESHOLD:
                continue

            if label not in label_to_id:
                continue

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                continue

            pred_boxes.append([x1, y1, x2, y2])
            pred_scores.append(confidence)
            pred_labels.append(label_to_id[label])

        # Append the processed predictions and targets for the current image
        preds.append({
            "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
            "scores": torch.tensor(pred_scores, dtype=torch.float32),
            "labels": torch.tensor(pred_labels, dtype=torch.int64),
        })

        # Append the ground truth targets for the current image
        targets.append({
            "boxes": torch.tensor(target_boxes, dtype=torch.float32),
            "labels": torch.tensor(target_labels, dtype=torch.int64),
        })

    return preds, targets

def calculate_map_metrics(
    ground_truth: dict,
    predictions: dict,
    label_to_id: dict[str, int],
) -> dict[str, float]:
    """Calculate mAP metrics."""
    preds, targets = build_map_inputs(ground_truth, predictions, label_to_id)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
    )

    metric.update(preds, targets)
    result = metric.compute()

    return {
        "map": float(result["map"].item()),
        "map_50": float(result["map_50"].item()),
        "map_75": float(result["map_75"].item()),
        "mar_100": float(result["mar_100"].item()),
    }

def evaluate_detection() -> None:
    """Evaluate YOLO object detection predictions against ground truth."""
    ground_truth = load_json(GROUND_TRUTH)
    predictions = load_json(PREDICTIONS)

    # Build a mapping of labels to integer IDs for consistent evaluation
    label_to_id = build_label_mapping(ground_truth, predictions)
    all_labels = sorted(label_to_id.keys())

    # Prepare lists for true labels, predicted labels, and per-image metrics
    y_true = []
    y_pred = []
    jaccard_scores = []
    exact_matches = []
    per_img_rows = []
    inference_times = []

    # Evaluate each image's predictions against the ground truth
    for image, true_labels in ground_truth.items():
        true_set = set(extract_true_labels(true_labels))

        pred_entry = predictions.get(image, {
            "detections": [],
            "inference_time_ms": None,
        })

        # Extract predicted labels for the current image
        pred_set = set(extract_predicted_labels(pred_entry))

        # Collect inference time if available
        inference_time = pred_entry.get("inference_time_ms")
        if inference_time is not None:
            inference_times.append(inference_time)

        for label in all_labels:
            y_true.append(1 if label in true_set else 0)
            y_pred.append(1 if label in pred_set else 0)

        # Calculate Jaccard similarity and exact match for the current image
        intersection = true_set & pred_set
        union = true_set | pred_set

        jaccard = len(intersection) / len(union) if union else 0
        exact_match = true_set == pred_set

        jaccard_scores.append(jaccard)
        exact_matches.append(exact_match)

        # Save per-image metrics for detailed analysis
        per_img_rows.append({
            "image": image,
            "true_labels": sorted(true_set),
            "predicted_labels": sorted(pred_set),
            "correct_labels": sorted(intersection),
            "missed_labels": sorted(true_set - pred_set),
            "extra_labels": sorted(pred_set - true_set),
            "jaccard": jaccard,
            "exact_match": exact_match,
            "inference_time_ms": inference_time,
        })

    # Calculate overall precision, recall, and F1 score
    precision, recall, f1, _  = precision_recall_fscore_support(
        y_true, y_pred,
        average="micro",
        zero_division=0,
    )

    # Calculate average inference time and frames per second (FPS)
    avg_inference_time = float(np.mean(inference_times)) if inference_times else None
    fps = 1000 / avg_inference_time if avg_inference_time else None

    # Calculate mAP metrics using torchmetrics
    map_metrics = calculate_map_metrics(ground_truth, predictions, label_to_id)

    # Save overall metrics to a JSON file
    metrics = {
        "model": "YOLO",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_jaccard": float(np.mean(jaccard_scores)),
        "exact_match_accuracy": float(np.mean(exact_matches)),
        "map": map_metrics["map"],
        "map_50": map_metrics["map_50"],
        "map_75": map_metrics["map_75"],
        "mar_100": map_metrics["mar_100"],
        "average_inference_time_ms": avg_inference_time,
        "fps": fps,
        "num_images": len(ground_truth),
    }
    save_json(metrics, OUTPUT_DIR)

    # Save per-image metrics to a CSV file for detailed analysis
    per_image_df = pd.DataFrame(per_img_rows)
    per_image_df.to_csv(OUTPUT_CSV, index=False)

    logger.info(metrics)
    logger.info(f"Saved metrics to {OUTPUT_DIR}")  # noqa: G004

if __name__ == "__main__":
    evaluate_detection()
