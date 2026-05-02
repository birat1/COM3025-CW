import logging

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH = "../tests/coco_indoor/selected_instances.json"
PREDICTIONS = "../outputs/predictions/yolo_predictions.json"
OUTPUT_DIR = "../outputs/metrics/yolo_detection_metrics.json"

def extract_predicted_labels(predictions: dict) -> list[str]:
    """Extract predicted labels from a prediction entry."""
    detections = predictions.get("detections", [])
    return [detection["label"] for detection in detections]

def evaluate_detection() -> None:
    """Evaluate YOLO object detection predictions against ground truth."""
    ground_truth = load_json(GROUND_TRUTH)
    predictions = load_json(PREDICTIONS)

    all_labels = set()

    # Collect all unique labels from ground truth and predictions
    for labels in ground_truth.values():
        all_labels.update(labels)

    # Also include labels from predictions to ensure all are accounted for
    for pred in predictions.values():
        all_labels.update(extract_predicted_labels(pred))

    all_labels = sorted(all_labels)

    # Prepare lists for true labels, predicted labels, and per-image metrics
    y_true = []
    y_pred = []
    jaccard_scores = []
    exact_matches = []
    per_img_rows = []
    inference_times = []

    # Evaluate each image's predictions against the ground truth
    for image, true_labels in ground_truth.items():
        true_set = set(true_labels)

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

    # Save overall metrics to a JSON file
    metrics = {
        "model": "YOLO",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_jaccard": float(np.mean(jaccard_scores)),
        "exact_match_accuracy": float(np.mean(exact_matches)),
        "average_inference_time_ms": avg_inference_time,
        "fps": fps,
        "num_images": len(ground_truth),
    }
    save_json(metrics, OUTPUT_DIR)

    # Save per-image metrics to a CSV file for detailed analysis
    per_image_df = pd.DataFrame(per_img_rows)
    per_image_df.to_csv("../outputs/metrics/yolo_detection_per_image.csv", index=False)

    logger.info(metrics)
    logger.info(f"Saved metrics to {OUTPUT_DIR}")  # noqa: G004

if __name__ == "__main__":
    evaluate_detection()
