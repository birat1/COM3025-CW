import shutil
import time
import uuid
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.detection import run_detection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = BASE_DIR / "outputs" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

def _cleanup_old_files(directory: Path, max_age: int = 180) -> None:
    """Utility function to clean up files in a directory."""  # noqa: D401
    cutoff = time.time()

    for file in directory.iterdir():
        if file.is_file():
            file_age = cutoff - file.stat().st_mtime

            if file_age > max_age:
                file.unlink(missing_ok=True)

def is_camera_covered(image_path: str) -> bool:
    """Detect if camera is covered."""
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(gray.mean())
    std_brightness = float(gray.std())

    # very dark with low constrast
    if mean_brightness < 25 and std_brightness < 15:
        return True
    
    
    if std_brightness < 8:
        return True
    
    return False

def describe(obj: str, position: str, proximity: str) -> str:
    """Create a simple description of object."""
    if proximity == "close":
        if position == "centre":
            return f"{obj} close in front"
        return f"{obj} close on the {position}"
    
    if proximity == "ahead":
        if position == "centre":
            return f"{obj} ahead"
        return f"{obj} to the {position}"
    
    if position == "centre":
        return f"{obj} far ahead"
    return f"{obj} far on the {position}"


def create_assistive_message(detections: list[dict]) -> str:
    """Create an assistive message summarising the scene based on caption and detections."""
    if not detections:
        return "No objects recognised in view."

    phrases = [
        describe(d["object"], d["position"], d["proximity"])
        for d in detections[:3]
    ]

    phrases[0] = phrases[0][0].upper() + phrases[0][1:]

    return ". ".join(phrases) + "."  # noqa: RET504

@app.get("/")
def read_root() -> dict:
    """Health check endpoint."""
    return {"message": "Assistive vision backend is running!"}

@app.post("/analyse-frame")
async def analyse_frame(file: Annotated[UploadFile, File()]) -> dict:
    """Endpoint to analyse an uploaded image frame and return caption, detections, and audio."""
    start_time = time.time()

    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, JPG, and PNG are supported.")

    _cleanup_old_files(UPLOAD_DIR)
    _cleanup_old_files(IMAGES_DIR)

    # Save the uploaded image to a temporary location
    image_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{image_id}.jpg"

    try:
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        annotated_image_file_name = f"{image_id}_annotated.jpg"
        annotated_image_path = IMAGES_DIR / annotated_image_file_name

        if is_camera_covered(str(image_path)):
            return {
                "assistive_message": "Camera appears to be covered.",
                "detections": [],
                "annotated_image_url": None,
                "latency_seconds": round(time.time() - start_time, 2),
            }

        # Run detection and captioning
        detections = run_detection(str(image_path), output_path=str(annotated_image_path))
        assistive_message = create_assistive_message(detections)

        return {  # noqa: TRY300
            "assistive_message": assistive_message,
            "detections": detections,
            "annotated_image_url": f"/images/{annotated_image_file_name}",
            "latency_seconds": round(time.time() - start_time, 2),
        }
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to analyse frame: {error}.")  # noqa: B904
