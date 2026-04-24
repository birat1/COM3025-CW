"""Module for text-to-speech conversion."""
from pathlib import Path
from gtts import gTTS
import uuid

AUDIO_DIR = Path("../outputs/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def text_to_speech(text):
    file_name = f"{uuid.uuid4()}.mp3"
    output_path = AUDIO_DIR / file_name

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(output_path)

    return file_name
