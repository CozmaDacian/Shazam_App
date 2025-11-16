import os

import librosa
from tqdm import tqdm
from create_database import DATA_PATH
from create_database import DB_PATH
import sttft
import utility
from create_database import SongDatabase
from sttft import hash_peaks

# --- STFT Parameters ---
SAMPLE_RATE = 44100
WINDOW_SIZE = 2048
HOP_SIZE = 512
WINDOW_TYPE = 'hanning'

# --- Peak Finding Parameters ---
PEAK_NEIGHBORHOOD = 35
PEAK_THRESHOLD_DB = -10

# --- Hashing Parameters ---
TARGET_ZONE_DT_MIN = 10
TARGET_ZONE_DT_MAX = 100
TARGET_ZONE_DF_RANGE = 50
MAX_TARGETS_PER_ANCHOR = 5

def populate_song_database(db:SongDatabase,root_path:str):
    """
    The function populates the table songs with informations from the mp3Data
    Args:
    :param db: The database in which we store the population

    :param root_path: The main path where we can find all the songs
    :return:
    """

    print("Populating Song Database")

    mp3_files = []

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith("mp3"):
                mp3_files.append(os.path.join(root, file))

    print(f"Found {len(mp3_files)} mp3 files")

    songs_added = 0
    for full_path in tqdm(mp3_files,desc="Populating Song Database"):
        metadata = utility.get_mp3_data(full_path)
        song_id = db.insert_song(
            metadata["title"]
            , metadata["artist"]
            , metadata["album"],
            full_path
        )
        if song_id is None:
            songs_added += 1

    print(f"Added {songs_added} songs")


def populate_fingerprint_database(db:SongDatabase):
    songs_to_process = db.get_all_songs_to_fingerprint()
    if songs_to_process is None:
        print("No songs to process")
        return

    print(f"Found {len(songs_to_process)} songs to process")

    for song_id,file_path in tqdm(songs_to_process,desc="Populating Song Database"):
        try:
            signal,_ = librosa.load(file_path, sr=SAMPLE_RATE,mono=True)
            stft_matrix,_,_ = sttft.stft(signal,WINDOW_SIZE,HOP_SIZE,WINDOW_TYPE,SAMPLE_RATE)
            peaks = sttft.get_peaks(stft_matrix,PEAK_NEIGHBORHOOD,PEAK_THRESHOLD_DB)
            if not peaks:
                print(f"No peaks found for song {song_id}. Skipping.")
                continue

            fingerprint_data = hash_peaks(peaks,song_id)

            if not fingerprint_data:
                print(f"No fingerpint data possible for {song_id}")
                continue
            else:
                print(f"Found {len(fingerprint_data)} keys for song")
            db.insert_fingerprints(fingerprint_data)
        except Exception as e:
            print(f"Failed to process song id {song_id} {e}")



if __name__ == "__main__":

    db_manager = SongDatabase(DB_PATH)
    db_manager.create_tables()
    populate_song_database(db_manager, DATA_PATH)

    populate_fingerprint_database(db_manager)
    print("Done")






