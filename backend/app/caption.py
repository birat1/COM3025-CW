"""Module for image captioning."""
import torch
from PIL import Image, UnidentifiedImageError

from app.models import device, git_model, git_processor


def generate_caption(image_path: str, max_new_tokens: int = 30) -> str:
    """Generate a caption for the given image using the GIT model."""
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Could not open image: {image_path}") from e  # noqa: EM102, TRY003

    pixel_values = git_processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate caption with GIT model
    with torch.no_grad():
        output = git_model.generate(
            pixel_values=pixel_values,
            max_length=max_new_tokens,
            num_beams=4,
        )

    caption = git_processor.batch_decode(output, skip_special_tokens=True)[0]

    return caption.strip()
