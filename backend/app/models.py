"""Module for models."""
import logging

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")  # noqa: G004

# YOLO
yolo_model = YOLO("yolov8s.pt")
