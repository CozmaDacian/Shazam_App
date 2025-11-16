import librosa
import sqlite3  # You need this
import random  # You need this

import numpy as np

# --- Make sure these point to your other files ---
from create_database import SongDatabase, DB_PATH
from populate_database import SAMPLE_RATE
from recognition import recognize_song

# --- Import your audio processing functions ---
# You must import these so 'recognize_song' can use them
from sttft import stft, get_peaks, _create_hash_key, hash_query_clip_peaks


# ... (all your other functions and imports) ...

def run_recognition_test(db: SongDatabase, num_songs_to_test: int = 5):
    """
    Runs the full recognition test with robust audio loading and added noise.
    """
    print(f"\n--- 🎵 STARTING RECOGNITION TEST (First {num_songs_to_test} songs) ---")

    songs = db.get_all_songs_to_fingerprint()
    songs = songs[0:num_songs_to_test]

    if not songs:
        print("No songs in database to test. Run the population script first.")
        return

    successes = 0

    for original_song_id, file_path in songs:
        print(f"\n--- Testing Song: {original_song_id}  ---")

        try:
            total_duration = librosa.get_duration(path=file_path, sr=SAMPLE_RATE)
            clip_duration_seconds = 5.0

            if total_duration < clip_duration_seconds:
                print(f"  [Skipping] Song is too short ({total_duration:.1f}s) to test.")
                continue

            max_offset = total_duration - clip_duration_seconds
            clip_offset_seconds = random.uniform(0, max_offset)

            print(f"Loading 5s clip from: {file_path} (at {clip_offset_seconds:.1f}s)")

            signal_clip, _ = librosa.load(
                file_path,
                sr=SAMPLE_RATE,
                mono=True,
                offset=clip_offset_seconds,
                duration=clip_duration_seconds
            )

            if signal_clip.size == 0:
                print("  [Error] Loaded an empty audio clip.")
                continue

            # --- ✅ THIS IS THE NEW LINE ---
            # Add white noise (Gaussian noise)
            # We'll make the noise 10% of the signal's standard deviation
            noise_amplitude = 0.1 * np.std(signal_clip)
            noise = np.random.normal(0, noise_amplitude, signal_clip.shape)
            signal_clip = signal_clip + noise
            print("  [Test] Added white noise to the clip.")
            # --- END OF NEW LINE ---

            # 4. Run recognition
            print("Running recognition...")
            match = recognize_song(signal_clip, db)

            # 5. Check result
            if match:
                predicted_id, vote_count = match
                print("\n--- MATCH FOUND ---")
                print(f"  Votes:     {vote_count}")
                print(f"  Predicted: Song ID {predicted_id}")

                if predicted_id == original_song_id:
                    print("  ✅ SUCCESS: Matched song {original_song_id} correctly.")
                    successes += 1
                else:
                    print(f"  ❌ FAILURE: Matched song {original_song_id} as {predicted_id}.")
            else:
                print("\n--- NO MATCH FOUND ---")

        except Exception as e:
            print(f"  🔥 ERROR processing song {original_song_id}: {e}")

    print("\n--- TEST COMPLETE ---")
    print(f"  Result: {successes} / {len(songs)} songs matched correctly.")


# ... (rest of your script, including if __name__ == "__main__":) ...

if __name__ == "__main__":

    db = SongDatabase(DB_PATH)
    run_recognition_test(db)
    # 1. Get a test song from the DB
    print("Getting a test song from the database...")
    with db._connect() as conn:
        # Get the song with ID=5 (or any ID you know is fingerprinted)
        test_song = conn.execute("SELECT song_id, title, file_path FROM songs WHERE song_id = 5").fetchone()

    if not test_song:
        print("Could not find test song in database.")
    else:
        original_song_id, title, file_path = test_song
        print(f"--- Testing with Song ID: {original_song_id} ('{title}') ---")

        clip_duration_seconds = 5.0

        try:
            # --- ✅ ROBUST AUDIO LOADING ---

            # 1. First, get the song's total duration
            try:
                total_duration = librosa.get_duration(path=file_path, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"  [Error] librosa failed to get duration for {file_path}: {e}")
                # Use 'continue' if this were in a loop
                exit()

                # 2. Check if song is long enough
            if total_duration < clip_duration_seconds:
                print(f"  [Skipping] Song is too short ({total_duration:.1f}s) to test.")
                exit()

            # 3. Pick a valid, random offset
            max_offset = total_duration - clip_duration_seconds
            clip_offset_seconds = random.uniform(0, max_offset)

            print(f"Loading 5s clip from: {file_path} (at {clip_offset_seconds:.1f}s)")

            # 4. Load the 5-second clip from the *safe* offset
            signal_clip, _ = librosa.load(
                file_path,
                sr=SAMPLE_RATE,
                mono=True,
                offset=clip_offset_seconds,
                duration=clip_duration_seconds
            )
            # --- End of Robust Loading ---

            if signal_clip.size == 0:
                print("Loaded an empty clip!")
            else:
                # 3. Run recognition
                print("Running recognition...")
                match = recognize_song(signal_clip, db)

                # 4. Print results
                if match:
                    predicted_id, vote_count = match
                    print("\n--- MATCH FOUND ---")
                    print(f"  Votes:     {vote_count}")
                    print(f"  Predicted: Song ID {predicted_id}")

                    if predicted_id == original_song_id:
                        print("  ✅ SUCCESS: Correct song was matched.")
                    else:
                        print("  ❌ FAILURE: Incorrect song was matched.")
                else:
                    print("\n--- NO MATCH FOUND ---")

        except Exception as e:
            print(f"An error occurred during testing: {e}")