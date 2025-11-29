import sqlite3
from typing import List, Tuple, Optional
from config import DB_PATH  # Import from config


class SongDatabase:
    """
    Manages all database operations for a single database file
    containing both 'songs' and 'fingerprints' tables.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        """Establishes a connection to the database."""
        conn = sqlite3.connect(self.db_path)
        return conn

    def _create_tables(self):
        """
        Creates the necessary tables ('songs', 'fingerprints')
        and indices in the database.
        """
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
                FOREIGN KEY (song_id) REFERENCES songs(song_id)
            );
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hash_key
            ON fingerprints (hash_key);
            """)

    def insert_song(self, title: str, artist: str, album: str, file_path: str) -> Optional[int]:
        """
        Inserts a single song into the 'songs' table.
        Returns the new song_id if successful, None if it's a duplicate.
        """
        sql = "INSERT INTO songs (title, artist, album, file_path) VALUES (?, ?, ?, ?)"
        data = (title, artist, album, file_path)

        with self._connect() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(sql, data)
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None  # Song already exists
            except Exception as e:
                print(f"Error inserting song: {e}")
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

    def get_songs_by_artist_like(self, song_artist: str) -> Optional[Tuple]:
        """Gets songs where the artist name contains the search text."""
        # The SQL query is changed from "artist = ?" to "artist LIKE ?"
        sql_query = "SELECT artist FROM songs WHERE artist LIKE ?"

        # We wrap the song_artist variable in '%' wildcards
        # This tells SQL to match any record that contains the song_artist string
        search_term = f"{song_artist}%"

        with self._connect() as conn:
            cursor = conn.cursor()
            # Pass the modified search_term as the parameter
            cursor.execute(sql_query, (search_term,))
            return cursor.fetchall()
    def get_songs_by_artist(self, song_artist: str) -> Optional[Tuple]:
        """Gets a single song's metadata by its ID."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title,song_id, album,file_path FROM songs WHERE artist = ?", (song_artist,))
            return cursor.fetchall()
    def get_song_by_id(self, song_id: int) -> Optional[Tuple]:
        """Gets a single song's metadata by its ID."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title, artist, album,file_path FROM songs WHERE song_id = ?", (song_id,))
            return cursor.fetchone()

    def get_matches_for_hashes(self, hash_keys: List[int]) -> List[Tuple[int, int, int]]:
        """
        Queries the fingerprints database for a list of hash keys.
        """
        if not hash_keys:
            return []

        placeholders = ','.join(['?'] * len(hash_keys))
        sql = f"SELECT hash_key, song_id, anchor_time FROM fingerprints WHERE hash_key IN ({placeholders})"

        with self._connect() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(sql, tuple(hash_keys))
                return cursor.fetchall()
            except Exception as e:
                print(f"Error querying hashes: {e}")
                return []

            # --- NEW FUNCTION FOR MACHINE LEARNING ---

    def get_songs_for_training(self) -> List[Tuple[str, str]]:
        """
        Retrieves all (file_path, artist) pairs for training the ML model.
        It only selects songs that have an artist label.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            query = "SELECT file_path, artist FROM songs WHERE artist IS NOT NULL AND artist != ''"
            cursor.execute(query)
            return cursor.fetchall()


if __name__ == "__main__":
    db = SongDatabase()
    songs = db.get_songs_by_artist("Blue Dot Sessions")
    print(songs)