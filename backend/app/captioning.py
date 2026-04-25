"""Module for image captioning."""
import torch
from PIL import Image
from app.models import blip_processor, blip_model, device

def generate_caption(image_path: str) -> str:
    """Generates a caption for the given image using the BLIP model."""
    image = Image.open(image_path).convert("RGB")

    inputs = blip_processor(image, return_tensors="pt").to(device)

    # Generate caption with BLIP model
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=30)

    caption = blip_processor.decode(output[0], skip_special_tokens=True)

    return caption
