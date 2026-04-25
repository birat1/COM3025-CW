import shutil
import uuid

from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.captioning import generate_caption
from app.detection import run_detection
from app.tts import text_to_speech

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
AUDIO_DIR = BASE_DIR / "outputs" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Assistive vision backend is running!"}

@app.post("/analyse-frame")
async def analyse_frame(file: UploadFile = File(...)):
    """Endpoint to analyse an uploaded image frame and return caption, detections, and audio."""
    # Save the uploaded image to a temporary location
    image_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{image_id}.jpg"

    with image_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    annotated_image_file_name = f"{image_id}_annotated.jpg"
    annotated_image_path = IMAGES_DIR / annotated_image_file_name

    # Run detection and captioning
    detections = run_detection(str(image_path), output_path=str(annotated_image_path))
    caption = generate_caption(str(image_path))

    # Create a summary of detected objects for TTS
    object_summary = ", ".join([
        f"{d['object']} on the {d['position']}, {d['distance']}"
        for d in detections
    ])

    # Run caption through TTS
    audio_file_name = text_to_speech(caption)

    return {
        "caption": caption,
        "detections": detections,
        "audio_url": f"/audio/{audio_file_name}",
        "annotated_image_url": f"/images/{annotated_image_file_name}"
    }