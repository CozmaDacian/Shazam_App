import numpy as np
import sqlite3
import os
import librosa
from typing import Dict,Optional,List,Tuple
from mutagen.id3 import ID3
from mutagen.mp3 import  MP3


DATA_PATH = "E:/fma_small/fma_small"
DB_PATH = "music_library.db"



class SongDatabase:
    def __init__(self, db_path:str):
        self.db_path = db_path

    def _connect(self)->sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs(
                song_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                artist TEXT,
                album TEXT,
                file_path TEXT UNIQUE
                );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints(
                hash_key INTEGER,
                song_id INTEGER,
                anchor_time INTEGER,
                FOREIGN KEY (song_id) REFERENCES songs(song_id));
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hash_key
            ON fingerprints (hash_key);
            """)

        print("Database tables created successfully")

    def insert_song(self,title:str,artist:str,album:str,file_path:str):

        """
        Insert a song into the database table songs

        """

        sql_command = "INSERT INTO songs(title,artist,album,file_path) VALUES(?,?,?,?)"
        data = (title,artist,album,file_path)

        with self._connect() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(sql_command,data)
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                print(f"Error inserting song {e}")
                return None
    def insert_fingerprints(self, fp_list: List[Tuple[int, int, int]]):
        """
        Inserts a batch of fingerprints into the 'fingerprints' table.
        """
        sql = "INSERT INTO fingerprints (hash_key, song_id, anchor_time) VALUES (?, ?, ?)"

        with self._connect() as conn:
            try:
                conn.executemany(sql, fp_list)
                conn.commit()
            except Exception as e:
                print(f"Error inserting fingerprints: {e}")

    def get_all_songs_to_fingerprint(self) -> List[Tuple[int, str]]:
        """
        Retrieves all (song_id, file_path) pairs from the 'songs' table.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT song_id, file_path FROM songs")
            return cursor.fetchall()

    def get_matches_for_hashes(self, hash_keys: List[int]) -> List[Tuple[int, int, int]]:
        """
        Queries the fingerprints database for a list of hash keys.
        """

        # 1. Guard clause (which you already have)
        #    This prevents the "IN ()" SQL error.
        if not hash_keys:
            return []

        placeholders = ','.join(['?'] * len(hash_keys))
        sql = f"SELECT hash_key, song_id, anchor_time FROM fingerprints WHERE hash_key IN ({placeholders})"

        with self._connect() as conn:
            try:
                cursor = conn.cursor()

                # --- ✅ THIS IS THE FIX ---
                # Explicitly cast the hash_keys list to a TUPLE.
                # The sqlite3 driver expects a tuple for parameter substitution.
                cursor.execute(sql, tuple(hash_keys))
                # --- END OF FIX ---

                return cursor.fetchall()
            except Exception as e:
                print(f"Error querying hashes: {e}")
                return []