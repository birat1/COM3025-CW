"""Module for models."""
import logging

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")  # noqa: G004

# YOLO
yolo_model = YOLO("yolov8s.pt")

# GIT
git_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco").to(device)
git_model.eval()
