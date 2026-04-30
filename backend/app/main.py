import shutil
import time
import uuid

from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.captioning import generate_caption
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
    """Utility function to clean up files in a directory."""
    cutoff = time.time()

    for file in directory.iterdir():
        if file.is_file():
            file_age = cutoff - file.stat().st_mtime

            if file_age > max_age:
                file.unlink(missing_ok=True)

def create_assistive_message(caption: str, detections: list[dict]) -> str:
    """Creates an assistive message summarizing the scene based on caption and detections."""
    if not detections:
        return f"{caption}"
    
    top_detections = detections[:3]

    object_summary = ". ".join([
        f"{d['object']} {d.get('proximity', d.get('distance', 'nearby'))} on the {d['position']}"
        for d in top_detections
    ])

    return f"{caption}"

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Assistive vision backend is running!"}

@app.post("/analyse-frame")
async def analyse_frame(file: UploadFile = File(...)):
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

        # Run detection and captioning
        detections = run_detection(str(image_path), output_path=str(annotated_image_path))
        caption = generate_caption(str(image_path), max_new_tokens=30)
        assistive_message = create_assistive_message(caption, detections)

        latency_seconds = round(time.time() - start_time, 2)

        return {
            "caption": caption,
            "assistive_message": assistive_message,
            "detections": detections,
            "annotated_image_url": f"/images/{annotated_image_file_name}",
            "latency_seconds": latency_seconds
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to analyse frame: {error}.")