import logging
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH = "../tests/coco_indoor/selected_captions.json"

MODEL_PREDICTIONS = {
    "BLIP": "../outputs/predictions/blip_captions.json",
    "ViT-GPT2": "../outputs/predictions/vit_gpt2_captions.json",
    "GIT": "../outputs/predictions/git_captions.json",
}

OUTPUT_DIR = "../outputs/metrics/model_captions_metrics.json"
PER_IMAGE_DIR = "../outputs/metrics"

def get_prediction_caption(pred: dict | str) -> str:
    """Extract the predicted caption from a prediction entry."""
    if isinstance(pred, str):
        return pred

    return pred.get("caption", "")

def evaluate_model(
    model_name: str,
    ground_truth: dict,
    predictions: dict,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate model and return metrics."""
    # Initialise the smoothing function for BLEU and ROUGE scorer
    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Prepare lists for BLEU-1, BLEU-4, ROUGE-L scores, per-image metrics
    bleu1_scores = []
    bleu4_scores = []
    rouge_l_scores = []
    per_img_rows = []
    inference_times = []

    # Evaluate each image's predicted caption against the ground truth captions
    for image_name, true_captions in ground_truth.items():
        pred_entry = predictions.get(image_name)
        if pred_entry is None:
            continue

        # Extract the predicted caption for the current image
        pred_caption = get_prediction_caption(pred_entry)
        if not pred_caption:
            continue

        if isinstance(pred_entry, dict) and pred_entry.get("inference_time_ms") is not None:
            inference_times.append(pred_entry["inference_time_ms"])

        # Calculate BLEU-1, BLEU-4, and ROUGE-L scores for the current image
        tokenised_refs = [caption.lower().split() for caption in true_captions]
        tokenised_preds = pred_caption.lower().split()

        bleu1 = sentence_bleu(
            tokenised_refs,
            tokenised_preds,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie,
        )

        bleu4 = sentence_bleu(
            tokenised_refs,
            tokenised_preds,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )

        rouge_scores = [
            rouge.score(true_caption, pred_caption)["rougeL"].fmeasure
            for true_caption in true_captions
        ]

        # Take the best ROUGE-L score across all ground truth captions for this image
        best_rouge_l = max(rouge_scores)

        # Append the scores and per-image metrics for detailed analysis
        bleu1_scores.append(bleu1)
        bleu4_scores.append(bleu4)
        rouge_l_scores.append(best_rouge_l)

        # Save per-image metrics for detailed analysis
        per_img_rows.append({
            "model": model_name,
            "image": image_name,
            "predicted_caption": pred_caption,
            "true_captions": true_captions,
            "bleu1": bleu1,
            "bleu4": bleu4,
            "rougeL": best_rouge_l,
        })

    # Calculate average inference time and frames per second (FPS)
    avg_inference_time = float(np.mean(inference_times)) if inference_times else None
    fps = 1000 / avg_inference_time if avg_inference_time else None

    # Save overall metrics to a JSON file
    metrics = {
        "model": model_name,
        "bleu1": float(np.mean(bleu1_scores)),
        "bleu4": float(np.mean(bleu4_scores)),
        "rougeL": float(np.mean(rouge_l_scores)),
        "avg_inference_time_ms": avg_inference_time,
        "fps": fps,
        "num_images": len(per_img_rows),
    }

    return metrics, pd.DataFrame(per_img_rows)

def evaluate_captions() -> None:
    """Evaluate captions against ground truth captions."""
    # Download the NLTK punkt tokenizer for BLEU score calculation
    nltk.download("punkt", quiet=True)

    ground_truth = load_json(GROUND_TRUTH)

    combined_results = {
        "num_images": len(ground_truth),
        "models_evaluated": [],
        "results": {},
    }

    comparison_rows = []

    for model_name, pred_dir in MODEL_PREDICTIONS.items():
        path = Path(pred_dir)

        if not path.exists():
            logger.warning(f"Skipping {model_name} - predictions file not found at {pred_dir}")  # noqa: G004
            continue

        logger.info(f"Evaluating {model_name} captions...")  # noqa: G004

        predictions = load_json(pred_dir)

        metrics, per_img_df = evaluate_model(
            model_name=model_name,
            ground_truth=ground_truth,
            predictions=predictions,
        )

        combined_results["models_evaluated"].append(model_name)
        combined_results["results"][model_name] = metrics
        comparison_rows.append(metrics)

        per_img_output = Path(PER_IMAGE_DIR) / f"{model_name.lower()}_caption_per_image.csv"
        per_img_df.to_csv(per_img_output, index=False)

        logger.info(metrics)

    save_json(combined_results, OUTPUT_DIR)

    logger.info(f"Saved overall metrics to {OUTPUT_DIR}")  # noqa: G004

if __name__ == "__main__":
    evaluate_captions()
