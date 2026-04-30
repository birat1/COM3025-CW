"""Module for image captioning."""
import torch
from PIL import Image, UnidentifiedImageError
from app.models import blip_processor, blip_model, device

def generate_caption(image_path: str, max_new_tokens: int = 30) -> str:
    """Generates a caption for the given image using the BLIP model."""
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"Could not open image: {image_path}")

    inputs = blip_processor(image, return_tensors="pt").to(device)

    # Generate caption with BLIP model
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)

    caption = blip_processor.decode(output[0], skip_special_tokens=True)

    return caption
