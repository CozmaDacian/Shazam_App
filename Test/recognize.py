import librosa
import random
import sqlite3
from typing import Optional

# Import our new classes and config
from config import DB_PATH
from database import SongDatabase
from audio_processor import AudioProcessor

if __name__ == "__main__":
    db = SongDatabase(DB_PATH)
    processor = AudioProcessor()

    # --- MODIFICATION: Set the specific song ID to test ---
    song_id_to_test = 4

    print(f"\n--- 🎵 STARTING RECOGNITION TEST FOR SONG ID: {song_id_to_test} ---")

    # 1. Get the specific song from the DB
    # We use get_song_by_id_for_test (which you had in your DB class)
    song_data = db.get_song_by_id(song_id_to_test)

    if not song_data:
        print(f"No song found with ID {song_id_to_test}. Run population script.")
    else:
        original_song_id, title,album,file_path = song_data
        print(f"--- Testing Song: {original_song_id} ('{title}') ---")

        try:
            # 2. Load a 5-second audio clip from a random offset
            total_duration = librosa.get_duration(path=file_path, sr=processor.sample_rate)
            clip_duration = 5.0

            if total_duration < clip_duration:
                print(f"  [Skipping] Song is too short ({total_duration:.1f}s) to test.")
            else:
                max_offset = total_duration - clip_duration
                clip_offset = random.uniform(0, max_offset)

                print(f"Loading 5s clip from: {file_path} (at {clip_offset:.1f}s)")

                signal_clip = processor.load_audio(
                    file_path,
                    offset=clip_offset,
                    duration=clip_duration
                )

                if signal_clip is None or signal_clip.size == 0:
                    print("  [Error] Loaded an empty audio clip.")
                else:
                    # 3. Run recognition
                    print("  Running recognition...")
                    match = processor.recognize(db, signal_clip)

                    # 4. Check result
                    if match:
                        predicted_id, vote_count = match
                        print(f"\n  --- MATCH FOUND (Votes: {vote_count}) ---")

                        if predicted_id == song_id_to_test :
                            print(f"  ✅ SUCCESS: Matched song {original_song_id} correctly.")
                        else:
                            predicted_info = db.get_song_by_id(predicted_id)
                            predicted_title = predicted_info[0] if predicted_info else "Unknown Title"
                            print(
                                f"  ❌ FAILURE: Matched song {title} as {predicted_id} ('{predicted_title}')")
                    else:
                        print("\n  --- NO MATCH FOUND ---")

        except Exception as e:
            print(f"  🔥 ERROR processing song {original_song_id}: {e}")

    print("\n--- TEST COMPLETE ---")