from operator import truediv
from typing import List, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt


def _create_hash_key(f1: int, f2: int, dt: int) -> int:
    """
    Packs f1, f2, and dt into a single 32-bit integer hash key
    using bitwise shifting.
    """
    PEAK_HASH_BITS_DT = 12  # 12 bits = max delta_t of 4095 frames
    PEAK_HASH_BITS_F2 = 10  # 10 bits = max freq_bin of 1023
    PEAK_HASH_BITS_F1 = 10  # 10 bits = max freq_bin of 1023
    # Total = 32 bits

    # --- Define the "Shifts" ---
    # dt is at the "bottom" (no shift)
    SHIFT_F2 = PEAK_HASH_BITS_DT  # Shift f2 left by 12 bits
    SHIFT_F1 = PEAK_HASH_BITS_DT + PEAK_HASH_BITS_F2  # Shift f1 left by 22 bits

    # --- Define the "Masks" ---
    # (1 << N) - 1 creates a string of N ones (e.g., (1<<10)-1 = 1111111111 in binary)
    MASK_F = (1 << PEAK_HASH_BITS_F1) - 1
    MASK_DT = (1 << PEAK_HASH_BITS_DT) - 1

    # Use (y << SHIFT) to move it to its correct slot
    # Use (a | b | c) to combine them
    hash_key = ((int(f1) & MASK_F) << SHIFT_F1) | \
               ((int(f2) & MASK_F) << SHIFT_F2) | \
               (int(dt) & MASK_DT)

    return int(hash_key)


def stft(signal: np.array, window_size, hop_size, window_type: str, sample_rate=44100):
    signal_length = len(signal);

    n_frames = 1 + (signal_length - window_size) // hop_size

    window = np.zeros(window_size)
    if (window_type == 'hamming'):
        window = np.hamming(window_size)
    elif (window_type == 'bartlett'):
        window = np.bartlett(window_size)
    elif (window_type == 'gaussian'):
        n = np.arange(0, window_size)
        center = (window_size - 1) // 2
        std_dev = (window_size - 1) // 5
        var = std_dev * std_dev
        window = np.exp(-(n - center) ** 2 / (2 * var))
    elif (window_type == 'hanning'):
        window = np.hanning(window_size)
    else:
        # Default to a known window (e.g., hanning) if type is unknown
        window = np.hanning(window_size)

    n_bins = window_size // 2 + 1

    stft_matrix = np.zeros((n_bins, n_frames), dtype=np.complex128)

    for i in range(n_frames):
        start_sample = i * hop_size
        end_sample = start_sample + window_size

        if end_sample > signal_length:
            break

        frame = signal[start_sample:end_sample]

        windowed_frame = window * frame
        fft_result = np.fft.rfft(windowed_frame, n=window_size)

        stft_matrix[:, i] = fft_result

    frequencies = np.fft.rfftfreq(window_size, d=1 / sample_rate)
    times = (hop_size * np.arange(n_frames)) / sample_rate

    return stft_matrix, frequencies, times


def get_peaks(stft_matrix, neighborhood_size, amp_treshold_db):
    '''
    Find the 2D local maximum peaks in a spectogram

    Args:
          sttft_matrix : the matrix that contains the complex fourier in the spectogram
          neighborhood_size : the size of the neighborhood
          amp_treshold_db : the minimum amplitude in decibels for it to be considered sound not white noise

    Returns:
        a list of tuples time_index,freq_index coordinates for each peak
    '''
    # Convert complex numbers amplitude to decibels
    S_db = 20 * np.log10(np.abs(stft_matrix) + 1e-6)
    n_bins, n_frames = S_db.shape
    peak_list = []

    # Calculate the "radius" of the neighborhood
    # (e.g., a size of 21 means 10 on each side
    N = neighborhood_size // 2

    for t in range(n_frames):
        for f in range(n_bins):
            pixel_value = S_db[f, t]
            if pixel_value < amp_treshold_db:
                continue
            is_max = True
            # Iterate through neighborhood from t-N to t+N and f-N to f+N
            for time_neigh in range(max(0, t - N), min(n_frames, t + N + 1)):
                for freq_neigh in range(max(0, f - N), min(n_bins, f + N + 1)):

                    # If the same pixel don't check it
                    if time_neigh == t and freq_neigh == f:
                        continue

                    if S_db[freq_neigh, time_neigh] > pixel_value:
                        is_max = False
                        break
                if not is_max:
                    break

            if is_max:
                peak_list.append((t, f))

    return peak_list


from collections import Counter  # Add this import at the top
import librosa
import random


# ... (Keep your _create_hash_key, stft, and get_peaks functions) ...

def hash_query_clip_peaks(peaks_list: list) -> Dict[int, List[int]]:
    """
    Hashes peaks from a query clip.

    Returns:
        A dictionary mapping: { hash_key -> [list of anchor_times] }
    """
    peaks_list.sort(key=lambda x: x[0])
    query_hashes = {}

    PEAK_HASH_BITS_F1 = 10
    MASK_F = (1 << PEAK_HASH_BITS_F1) - 1

    TARGET_ZONE_DT_MIN = 10
    TARGET_ZONE_DT_MAX = 100
    TARGET_ZONE_DF_RANGE = 50
    MAX_TARGETS_PER_ANCHOR = 5

    for i in range(len(peaks_list)):
        anchor_time, anchor_freq = peaks_list[i]
        if anchor_freq > MASK_F: continue

        targets_found = 0

        for j in range(i + 1, len(peaks_list)):
            target_time, target_freq = peaks_list[j]
            if target_freq > MASK_F: continue

            if targets_found >= MAX_TARGETS_PER_ANCHOR:
                break

            dt = target_time - anchor_time
            df = abs(target_freq - anchor_freq)

            if dt < TARGET_ZONE_DT_MIN:
                continue
            if dt > TARGET_ZONE_DT_MAX:
                break
            if df > TARGET_ZONE_DF_RANGE:
                continue

            hash_key = _create_hash_key(anchor_freq, target_freq, dt)

            # Store the time this hash occurred in the query clip
            if hash_key not in query_hashes:
                query_hashes[hash_key] = []
            query_hashes[hash_key].append(anchor_time)

            targets_found += 1

    return query_hashes
def hash_peaks(peaks_list: list, song_id: int) -> List[Tuple[int, int, int]]:
    '''
    Generates the fingerprint for a song peak list
    '''

    TARGET_ZONE_DT_MIN = 10
    TARGET_ZONE_DT_MAX = 100
    TARGET_ZONE_DF_RANGE = 50

    # FIX: Change this from a dictionary to a list
    fingerprint_data = []

    peaks_list.sort(key=lambda x: x[0])

    PEAK_HASH_BITS_F1 = 10
    MASK_F = (1 << PEAK_HASH_BITS_F1) - 1

    for i in range(len(peaks_list)):
        anchor_time, anchor_freq = peaks_list[i]

        if anchor_freq > MASK_F:
            continue

        # Add a safety rail to limit hashes per anchor
        targets_found = 0
        MAX_TARGETS_PER_ANCHOR = 5

        for j in range(i + 1, len(peaks_list)):
            target_time, target_freq = peaks_list[j]

            if target_freq > MASK_F:
                continue

            if targets_found >= MAX_TARGETS_PER_ANCHOR:
                break

            dt = target_time - anchor_time
            df = abs(target_freq - anchor_freq)

            if dt < TARGET_ZONE_DT_MIN:
                continue

            if dt > TARGET_ZONE_DT_MAX:
                break

            if df > TARGET_ZONE_DF_RANGE:
                continue

            hash_key = _create_hash_key(anchor_freq, target_freq, dt)

            # FIX: Append the tuple (hash, song_id, time) to the list
            fingerprint_data.append((hash_key, song_id, anchor_time))
            targets_found += 1

    return fingerprint_data