import os
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from typing import Dict, Optional, List


def get_mp3_data(path: str) -> Dict[str, Optional[str]]:
    """Extracts metadata (Title, Artist, Album) from an MP3 file."""
    try:
        audio = MP3(path, ID3=ID3)
        # Ensure text is a list and get the first element
        title = audio.get("TIT2").text if audio.get("TIT2") else None
        artist = audio.get("TPE1").text if audio.get("TPE1") else None
        album = audio.get("TALB").text if audio.get("TALB") else None

        return {
            "title": title[0] if title else None,
            "artist": artist[0] if artist else None,
            "album": album[0] if album else None,
        }
    except Exception as e:
        # print(f"Error reading metadata from {path}: {e}")
        return {"title": None, "artist": None, "album": None}


def find_mp3_files(root_path: str) -> List[str]:
    """Recursively finds all mp3 files in a directory."""
    mp3_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))
    return mp3_files