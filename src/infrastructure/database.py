import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    status TEXT DEFAULT 'QUEUED',
                    target_langs TEXT,
                    source_lang TEXT,
                    file_size INTEGER,
                    video_duration REAL,
                    has_subtitles BOOLEAN,
                    started_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migration logic
            cols = [
                ("target_langs", "TEXT"),
                ("source_lang", "TEXT"),
                ("file_size", "INTEGER"),
                ("video_duration", "REAL"),
                ("has_subtitles", "BOOLEAN"),
                ("started_at", "TIMESTAMP"),
            ]
            for col_name, col_type in cols:
                try:
                    c.execute(f"ALTER TABLE tasks ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

    def add_task(self, path: str, meta: Dict):
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT status FROM tasks WHERE path = ?", (path,))
            row = c.fetchone()
            if row:
                if row["status"] not in ["QUEUED", "PROCESSING"]:
                    c.execute(
                        """
                        UPDATE tasks SET
                        status = 'QUEUED', updated_at = ?, target_langs = ?,
                        source_lang = ?, file_size = ?, video_duration = ?,
                        has_subtitles = ? WHERE path = ?
                    """,
                        (
                            datetime.now(),
                            meta.get("target_langs"),
                            meta.get("source_lang"),
                            meta.get("size"),
                            meta.get("duration"),
                            meta.get("has_subs"),
                            path,
                        ),
                    )
                    return "re-queued"
                return "ignored"
            else:
                c.execute(
                    """
                    INSERT INTO tasks
                    (path, target_langs, source_lang, file_size, video_duration, has_subtitles)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        path,
                        meta.get("target_langs"),
                        meta.get("source_lang"),
                        meta.get("size"),
                        meta.get("duration"),
                        meta.get("has_subs"),
                    ),
                )
                return "queued"

    def fetch_next_task(self) -> Optional[Dict]:
        with self._get_connection() as conn:
            c = conn.cursor()
            try:
                c.execute("BEGIN IMMEDIATE")
                c.execute("SELECT * FROM tasks WHERE status = 'QUEUED' ORDER BY created_at ASC LIMIT 1")
                row = c.fetchone()
                if row:
                    task = dict(row)
                    c.execute(
                        "UPDATE tasks SET status = 'PROCESSING', started_at = ?, updated_at = ? WHERE id = ?",
                        (datetime.now(), datetime.now(), task["id"]),
                    )
                    conn.commit()
                    return task
                conn.commit()
                return None
            except Exception:
                conn.rollback()
                raise

    def update_status(self, task_id: int, status: str):
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, datetime.now(), task_id),
            )
            conn.commit()

    def get_all_tasks(self) -> List[Dict]:
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM tasks ORDER BY created_at DESC")
            return [dict(row) for row in c.fetchall()]

    def reset_interrupted_tasks(self):
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("UPDATE tasks SET status = 'QUEUED' WHERE status = 'PROCESSING'")
            return c.rowcount

    def delete_task(self, task_id: int):
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()

    def retry_task(self, task_id: int):
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("UPDATE tasks SET status = 'QUEUED', updated_at = ? WHERE id = ?", (datetime.now(), task_id))
            conn.commit()
