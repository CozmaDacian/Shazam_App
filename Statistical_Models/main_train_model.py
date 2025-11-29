# (REPLACE THIS FILE)
# FILE: main_train_model.py

import numpy as np
import librosa
import joblib
from tqdm import tqdm
from typing import List, Tuple
from collections import Counter

# Your project files
from database import SongDatabase
from audio_processor import AudioProcessor
from config import DB_PATH, PCA_N_COMPONENTS

# Scikit-learn imports
from sklearn.decomposition import IncrementalPCA  # <-- Use IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# How many songs to load into RAM at once for PCA fitting
PCA_BATCH_SIZE = 50


def extract_statistical_features(pca_frames: np.array) -> np.array:
    """
    Calculates the final feature vector from the PCA-transformed frames.
    (mean, std, delta-mean, delta-std)
    """
    deltas = librosa.feature.delta(pca_frames, axis=0)

    mean = np.mean(pca_frames, axis=0)
    std = np.std(pca_frames, axis=0)


    return np.concatenate([mean, std])


def fit_pca_in_batches(processor: AudioProcessor,
                       song_list: List[Tuple[str, str]]) -> IncrementalPCA:
    """
    Fits an IncrementalPCA model by loading and processing songs
    in small batches to avoid OOM errors.
    """
    print(f"\n--- Fitting PCA Model in batches of {PCA_BATCH_SIZE} ---")
    pca_model = IncrementalPCA(n_components=PCA_N_COMPONENTS)

    # Process in batches
    for i in tqdm(range(0, len(song_list), PCA_BATCH_SIZE), desc="Fitting PCA"):
        batch_song_list = song_list[i:i + PCA_BATCH_SIZE]
        batch_frames = []

        for file_path, _ in batch_song_list:
            signal = processor.load_audio(file_path)
            if signal is None or len(signal) < processor.window_size:
                continue

            stft_matrix = processor.stft(signal)
            S_abs_T = np.abs(stft_matrix).T
            batch_frames.append(S_abs_T)

        if not batch_frames:
            continue

        # Stack frames *only for this batch*
        batch_stacked_frames = np.vstack(batch_frames)

        # Fit on this batch
        pca_model.partial_fit(batch_stacked_frames)

        # Clear batch from memory
        del batch_stacked_frames
        del batch_frames

    print("PCA model fitting complete.")
    return pca_model


def extract_features_all_songs(processor: AudioProcessor,
                               pca_model: IncrementalPCA,
                               song_list: List[Tuple[str, str]]
                               ) -> Tuple[np.array, np.array]:
    """
    Loops through all songs a second time.
    Now that PCA is fitted, we transform each song one-by-one
    and extract its final, small feature vector.
    """
    print("\n--- Extracting Final Song Features ---")
    X_features = []
    y_labels = []

    for file_path, artist in tqdm(song_list, desc="Extracting features"):
        signal = processor.load_audio(file_path)
        if signal is None or len(signal) < processor.window_size:
            continue

        # 1. Get STFT
        stft_matrix = processor.stft(signal)
        S_abs_T = np.abs(stft_matrix).T

        # 2. Transform frames using the *fitted* PCA
        # Note: We must check for very short clips that have fewer
        # frames than n_components. We will pad them.
        if S_abs_T.shape[0] < pca_model.n_components:
            print(f"Warning: Skipping short song {file_path}")
            continue

        pca_frames = pca_model.transform(S_abs_T)

        # 3. Get statistical features (mean, std, etc.)
        song_feature_vector = extract_statistical_features(pca_frames)

        X_features.append(song_feature_vector)
        y_labels.append(artist)

    return np.array(X_features), np.array(y_labels)


def main():
    print("--- 🎵 Artist Classification Model Training ---")

    # --- 1. Setup ---
    db = SongDatabase(DB_PATH)
    processor = AudioProcessor()

    # --- 2. Load Data Paths ---
    song_list = db.get_songs_for_training()
    if not song_list:
        print("Error: No songs found. Run `main_populate.py` first.")
        return

    artist_counts = Counter([artist for _, artist in song_list])
    MIN_SONGS_PER_ARTIST = 30
    filtered_song_list = [
        (fp, art) for fp, art in song_list
        if artist_counts[art] >= MIN_SONGS_PER_ARTIST
    ]
    print(f"Original songs: {len(song_list)}. Filtered songs: {len(filtered_song_list)}")

    if not filtered_song_list:
        print("Error: No songs left after filtering.")
        return

    # --- 3. Fit PCA in Batches ---
    pca_model = fit_pca_in_batches(processor, filtered_song_list)

    # Save the fitted PCA model
    joblib.dump(pca_model, 'pca_model.joblib')
    print("PCA model saved to 'pca_model.joblib'")

    # --- 4. Extract Final Features (Loop 2) ---
    X, y_labels = extract_features_all_songs(processor, pca_model, filtered_song_list)

    if X.shape[0] == 0:
        print("Error: No features were extracted. Check audio files.")
        return

    # --- 5. Encode Labels ---
    print("\n--- Encoding Labels ---")
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    joblib.dump(le, 'label_encoder.joblib')
    print(f"Found {len(le.classes_)} artists. Label encoder saved to 'label_encoder.joblib'")

    # --- 6. Train Classifier ---
    print("\n--- Training Classification Model ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training on {len(y_train)} samples, testing on {len(y_test)} samples.")

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    joblib.dump(model, 'artist_classifier.joblib')
    print("Classifier model saved to 'artist_classifier.joblib'")

    # --- 7. Evaluate Model ---
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"**Overall Accuracy: {accuracy * 100:.2f}%**")

    print("\nClassification Report:")
    target_names = le.classes_
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    # --- Dependencies ---
    # Make sure scikit-learn, librosa, and joblib are installed
    # pip install scikit-learn librosa joblib

    main()