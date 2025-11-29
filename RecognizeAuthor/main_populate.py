import os
from tqdm import tqdm

# Import our new classes and config
from config import DB_PATH, DATA_PATH
from database import SongDatabase
from audio_processor import AudioProcessor
from utils import get_mp3_data, find_mp3_files  # Import helpers


def populate_songs_table(db: SongDatabase, root_path: str):
    """
    Scans the data_path for all .mp3 files and adds their
    metadata to the songs table.
    """
    print("--- PHASE 1: POPULATING SONGS TABLE ---")
    print(f"Scanning {root_path} for MP3 files...")

    mp3_files = find_mp3_files(root_path)
    print(f"Found {len(mp3_files)} MP3 files.")

    songs_added = 0
    for full_path in tqdm(mp3_files, desc="Populating songs.db"):
        metadata = get_mp3_data(full_path)
        song_id = db.insert_song(
            metadata["title"],
            metadata["artist"],
            metadata["album"],
            full_path
        )
        if song_id is not None:  # Only count new songs
            songs_added += 1

    print(f"Phase 1 complete. Added {songs_added} new songs to '{DB_PATH}'.")


def populate_fingerprints_table(db: SongDatabase, processor: AudioProcessor):
    """
    Goes through every song in the songs table and creates
    fingerprints for it in the fingerprints table.
    """
    print("\n--- PHASE 2: FINGERPRINTING SONGS ---")

    songs_to_process = db.get_all_songs_to_fingerprint()
    if not songs_to_process:
        print("No songs found in database to fingerprint.")
        return

    print(f"Found {len(songs_to_process)} songs to fingerprint.")

    for song_id, file_path in tqdm(songs_to_process, desc="Fingerprinting songs"):
        try:
            # 1. Load audio
            signal = processor.load_audio(file_path)
            if signal is None:
                print(f"Warning: Could not load audio for song {song_id}. Skipping.")
                continue

            # 2. Run STFT
            stft_matrix = processor.stft(signal)

            # 3. Get Peaks (using DB settings)
            peaks = processor.get_peaks(stft_matrix, for_query=False)
            if not peaks:
                print(f"No peaks found for song {song_id}. Skipping.")
                continue

            # 4. Hash Peaks
            fingerprint_data = processor.hash_peaks_for_db(peaks, song_id)
            if not fingerprint_data:
                print(f"No hashes created for song {song_id}. Skipping.")
                continue

            # 5. Insert into DB
            db.insert_fingerprints(fingerprint_data)

        except Exception as e:
            print(f"Failed to fingerprint song {song_id} ({file_path}): {e}")

    print("Phase 2 complete. Fingerprinting finished.")


if __name__ == "__main__":
    # 1. Initialize the main classes
    db = SongDatabase(DB_PATH)
    processor = AudioProcessor()

    # 2. Run Phase 1 to find all songs
    populate_songs_table(db, DATA_PATH)

    # 3. Run Phase 2 to fingerprint every song
    #populate_fingerprints_table(db, processor)

    print("\nDatabase population complete.")
