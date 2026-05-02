import json
import time
from pathlib import Path


def load_json(path: str) -> dict:
    """Load a JSON file from the given path."""
    path = Path(path)

    with Path.open(path, "r") as f:
        return json.load(f)

def save_json(data: dict, path: str) -> None:
    """Save data to a JSON file at the given path."""
    path = Path(path)

    with Path.open(path, "w") as f:
        json.dump(data, f, indent=2)

def list_images(image_dir: str) -> list[Path]:
    """List all image files in the given directory."""
    image_dir = Path(image_dir)

    valid_extensions = [".jpg", ".jpeg", ".png"]

    return sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in valid_extensions
    ])

class Timer:
    """Class to measure elapsed time."""

    def __enter__(self):
        """Start the timer when entering the context."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        """Calculate elapsed time when exiting the context."""
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        self.elapsed_ms = self.elapsed * 1000
