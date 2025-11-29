# ==========================================================
# CONFIGURATION
# ==========================================================
import os

# --- Paths ---
# Use os.path.dirname to make paths relative to the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "E:/fma_medium/"
DB_PATH = os.path.join(BASE_DIR, "music_library.db")

# --- STFT Parameters ---
SAMPLE_RATE = 44100
WINDOW_SIZE = 2048
HOP_SIZE = 512
WINDOW_TYPE = 'hanning'

# --- Peak Finding Parameters ---
# Use stricter settings for the database (fewer, more robust peaks)
DB_PEAK_NEIGHBORHOOD = 35
DB_PEAK_THRESHOLD_DB = -10

# Use more lenient settings for the query clip (to find *something*)
QUERY_PEAK_NEIGHBORHOOD = 21
QUERY_PEAK_THRESHOLD_DB = -20

# --- Hashing Parameters ---
TARGET_ZONE_DT_MIN = 10
TARGET_ZONE_DT_MAX = 100
TARGET_ZONE_DF_RANGE = 50
MAX_TARGETS_PER_ANCHOR = 5

# --- Hashing Bitmask Constants ---
PEAK_HASH_BITS_DT = 12
PEAK_HASH_BITS_F2 = 10
PEAK_HASH_BITS_F1 = 10
SHIFT_F2 = PEAK_HASH_BITS_DT
SHIFT_F1 = PEAK_HASH_BITS_DT + PEAK_HASH_BITS_F2
MASK_F = (1 << PEAK_HASH_BITS_F1) - 1
MASK_DT = (1 << PEAK_HASH_BITS_DT) - 1



N_MFCC = 40
# --- PCA Feature Extraction Parameters ---
# How many "smart features" to flatten to
PCA_N_COMPONENTS = 120
# Our final feature vector will be (PCA_N_COMPONENTS * 2)
# because we take the mean and std_dev for each component.