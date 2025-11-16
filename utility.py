from typing import Dict, Optional

from mutagen.id3 import ID3
from mutagen.mp3 import MP3


def get_mp3_data(path: str) -> Dict[str, Optional[str]]:
    """Extracts metadata (Title, Artist, Album) from an MP3 file."""
    try:
        audio = MP3(path, ID3=ID3)
        return {
            "title": audio.get("TIT2").text[0] if audio.get("TIT2") else None,
            "artist": audio.get("TPE1").text[0] if audio.get("TPE1") else None,
            "album": audio.get("TALB").text[0] if audio.get("TALB") else None,
        }
    except Exception:
        return {"title": None, "artist": None, "album": None}