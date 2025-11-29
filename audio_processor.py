import numpy as np
import librosa
from typing import List, Tuple, Dict, Optional
from collections import Counter

# Import all constants from your config file
from config import *
# Import your database class for type hinting
from database import SongDatabase


class AudioProcessor:
    """
    A class to handle all audio processing:
    - Loading Audio
    - STFT
    - Peak Finding
    - Hashing (for DB and Query)
    - Recognition
    """

    def __init__(self):
        # Store constants from config
        self.sample_rate = SAMPLE_RATE
        self.window_size = WINDOW_SIZE
        self.hop_size = HOP_SIZE
        self.window_type = WINDOW_TYPE

        # Hashing params
        self.dt_min = TARGET_ZONE_DT_MIN
        self.dt_max = TARGET_ZONE_DT_MAX
        self.df_range = TARGET_ZONE_DF_RANGE
        self.max_targets = MAX_TARGETS_PER_ANCHOR

        # Peak finding params
        self.db_peak_thresh = DB_PEAK_THRESHOLD_DB
        self.db_peak_neighborhood = DB_PEAK_NEIGHBORHOOD
        self.query_peak_thresh = QUERY_PEAK_THRESHOLD_DB
        self.query_peak_neighborhood = QUERY_PEAK_NEIGHBORHOOD

    def load_audio(self, file_path: str, offset: float = 0.0, duration: Optional[float] = None) -> Optional[np.array]:
        """Loads and resamples audio using librosa."""
        try:
            signal, _ = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=True,
                offset=offset,
                duration=duration
            )
            return signal
        except Exception as e:
            print(f"Error loading audio {file_path}: {e}")
            return None

    def stft(self, signal: np.array) -> np.array:
        """Computes the Short-Time Fourier Transform (STFT) of a signal."""
        signal_length = len(signal)
        n_frames = 1 + (signal_length - self.window_size) // self.hop_size

        if (self.window_type == 'hamming'):
            window = np.hamming(self.window_size)
        elif (self.window_type == 'bartlett'):
            window = np.bartlett(self.window_size)
        elif (self.window_type == 'gaussian'):
            n = np.arange(0, self.window_size)
            center = (self.window_size - 1) // 2
            std_dev = (self.window_size - 1) // 5
            var = std_dev * std_dev
            window = np.exp(-(n - center) ** 2 / (2 * var))
        else:
            window = np.hanning(self.window_size)

        n_bins = self.window_size // 2 + 1
        stft_matrix = np.zeros((n_bins, n_frames), dtype=np.complex128)

        for i in range(n_frames):
            start_sample = i * self.hop_size
            end_sample = start_sample + self.window_size
            if end_sample > signal_length:
                break
            frame = signal[start_sample:end_sample]
            windowed_frame = window * frame
            fft_result = np.fft.rfft(windowed_frame, n=self.window_size)
            stft_matrix[:, i] = fft_result

        return stft_matrix

    def get_peaks(self, stft_matrix: np.array, for_query: bool = False) -> List[Tuple[int, int]]:
        """
        Finds the 2D local maxima (peaks) in a spectrogram
        using the manual loop method.
        """
        if for_query:
            neighborhood_size = self.query_peak_neighborhood
            amp_threshold_db = self.query_peak_thresh
        else:
            neighborhood_size = self.db_peak_neighborhood
            amp_threshold_db = self.db_peak_thresh

        S_db = 20 * np.log10(np.abs(stft_matrix) + 1e-6)
        n_bins, n_frames = S_db.shape
        peak_list = []
        N = neighborhood_size // 2

        for t in range(n_frames):
            for f in range(n_bins):
                pixel_value = S_db[f, t]
                if pixel_value < amp_threshold_db:
                    continue
                is_max = True
                for time_neigh in range(max(0, t - N), min(n_frames, t + N + 1)):
                    for freq_neigh in range(max(0, f - N), min(n_bins, f + N + 1)):
                        if time_neigh == t and freq_neigh == f:
                            continue
                        if S_db[freq_neigh, time_neigh] > pixel_value:
                            is_max = False
                            break
                    if not is_max:
                        break
                if is_max:
                    peak_list.append((t, f))  # (time_index, freq_index)

        return peak_list

    def _create_hash_key(self, f1: int, f2: int, dt: int) -> int:
        """Packs f1, f2, and dt into a single 32-bit integer hash key."""
        hash_key = ((int(f1) & MASK_F) << SHIFT_F1) | \
                   ((int(f2) & MASK_F) << SHIFT_F2) | \
                   (int(dt) & MASK_DT)
        return int(hash_key)

    def hash_peaks_for_db(self, peaks_list: list, song_id: int) -> List[Tuple[int, int, int]]:
        """Generates a list of (hash, song_id, time) tuples for DB insertion."""
        fingerprint_data = []
        peaks_list.sort(key=lambda x: x[0])  # Sort by time

        for i in range(len(peaks_list)):
            anchor_time, anchor_freq = peaks_list[i]
            if anchor_freq > MASK_F: continue
            targets_found = 0

            for j in range(i + 1, len(peaks_list)):
                target_time, target_freq = peaks_list[j]
                if target_freq > MASK_F: continue
                if targets_found >= self.max_targets:
                    break

                dt = target_time - anchor_time
                df = abs(target_freq - anchor_freq)

                if dt < self.dt_min: continue
                if dt > self.dt_max: break
                if df > self.df_range: continue

                hash_key = self._create_hash_key(anchor_freq, target_freq, dt)
                fingerprint_data.append((hash_key, song_id, anchor_time))
                targets_found += 1

        return fingerprint_data

    def hash_peaks_for_query(self, peaks_list: list) -> Dict[int, List[int]]:
        """Generates a {hash -> [times]} map for a query clip."""
        query_hashes = {}
        peaks_list.sort(key=lambda x: x[0])

        for i in range(len(peaks_list)):
            anchor_time, anchor_freq = peaks_list[i]
            if anchor_freq > MASK_F: continue
            targets_found = 0

            for j in range(i + 1, len(peaks_list)):
                target_time, target_freq = peaks_list[j]
                if target_freq > MASK_F: continue
                if targets_found >= self.max_targets:
                    break

                dt = target_time - anchor_time
                df = abs(target_freq - anchor_freq)

                if dt < self.dt_min: continue
                if dt > self.dt_max: break
                if df > self.df_range: continue

                hash_key = self._create_hash_key(anchor_freq, target_freq, dt)
                if hash_key not in query_hashes:
                    query_hashes[hash_key] = []
                query_hashes[hash_key].append(anchor_time)
                targets_found += 1

        return query_hashes

    def recognize(self, db: SongDatabase, signal_clip: np.array) -> Optional[Tuple[int, int]]:
        """
        Runs the full recognition pipeline on a signal clip.
        Returns (predicted_song_id, match_count) or None.
        (This logic is from your recognition.py)
        """
        stft_matrix = self.stft(signal_clip)
        peaks = self.get_peaks(stft_matrix, for_query=True)  # Use query settings
        if not peaks:
            print("  [Recognition] No peaks found in query clip.")
            return None

        query_hash_map = self.hash_peaks_for_query(peaks)
        if not query_hash_map:
            print("  [Recognition] No hashes created for query clip.")
            return None

        db_hits = db.get_matches_for_hashes(list(query_hash_map.keys()))
        if not db_hits:
            print("  [Recognition] No matches found in database.")
            return None

        offset_histogram = Counter()
        for hash_key, song_id, db_anchor_time in db_hits:
            query_anchor_times = query_hash_map.get(hash_key, [])
            for query_anchor_time in query_anchor_times:
                offset = db_anchor_time - query_anchor_time
                offset_histogram[(song_id, offset)] += 1

        if not offset_histogram:
            print("  [Recognition] No valid time offsets found.")
            return None

        best_match = offset_histogram.most_common(1)[0]
        (predicted_song_id, predicted_offset), match_count = best_match

        return (predicted_song_id, match_count)