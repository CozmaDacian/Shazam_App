from collections import Counter

import numpy as np

import sttft
from create_database import SongDatabase
from populate_database import  SAMPLE_RATE,HOP_SIZE,PEAK_NEIGHBORHOOD,PEAK_THRESHOLD_DB,MAX_TARGETS_PER_ANCHOR,WINDOW_SIZE,WINDOW_TYPE
from sttft import get_peaks, hash_query_clip_peaks


def recognize_song(signal_clip:np.array,db:SongDatabase):
    sttft_matrix,_,_ = sttft.stft(signal_clip,window_size=WINDOW_SIZE,hop_size=HOP_SIZE,window_type=WINDOW_TYPE,sample_rate=SAMPLE_RATE)
    peaks = get_peaks(sttft_matrix,PEAK_NEIGHBORHOOD,PEAK_THRESHOLD_DB)

    if not peaks:
        print('No peaks found in query clip')
        return None

    query_hash_map = hash_query_clip_peaks(peaks)
    if not query_hash_map:
         print("Recognition failed: No hashes created for query clip")
         return None

    list1 = query_hash_map.keys()
    db_hits = db.get_matches_for_hashes(list1)

    if not db_hits:
        print("Recognition failed: No matches found")
        return None

    offset_Histogram = Counter()

    for hash_key,song_id,db_anchor_time in db_hits:

        query_anchor_times = query_hash_map.get(hash_key,[])

        for query_anchor_time in query_anchor_times:
            offset = db_anchor_time - query_anchor_time
            offset_Histogram[(song_id,offset)] += 1

    if not offset_Histogram:
        print("Recognition failed: No valid time offsets found.")
        return None

        # Find the (song_id, offset) tuple with the highest vote count
    best_match = offset_Histogram.most_common(1)[0]
    (predicted_song_id, predicted_offset), match_count = best_match

    return (predicted_song_id, match_count)




