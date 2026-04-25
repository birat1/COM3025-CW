"""Module for models."""
import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo_model = YOLO('yolov8s.pt')

blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)
