"""SQLite telemetry store with an asynchronous write pipeline (REQ-10, REQ-14).

The engine thread only does queue.put() (microseconds); a writer thread owns the
write connection and commits in small batches. Readers (dashboard) open their
own read-only connections; WAL mode lets them read while the writer commits.
"""
from __future__ import annotations

import queue
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspection_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id VARCHAR(64) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    classification_status VARCHAR(16) CHECK(classification_status IN ('Good', 'Defective')),
    defect_area_pixels INTEGER DEFAULT 0,
    confidence_score FLOAT CHECK(confidence_score BETWEEN 0.0 AND 1.0),
    actuator_triggered BOOLEAN NOT NULL,
    image_frame_path TEXT,
    inference_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON inspection_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_status ON inspection_records(classification_status);
"""

INSERT = (
    "INSERT INTO inspection_records (product_id, classification_status, defect_area_pixels, "
    "confidence_score, actuator_triggered, image_frame_path, inference_ms) VALUES (?,?,?,?,?,?,?)"
)


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    con.commit()
    con.close()


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return con


class AsyncLogger:
    """Non-blocking inspection logger. Call .log(...) from the hot path."""

    def __init__(self, db_path: str | Path, queue_size: int = 2000):
        self.db_path = str(db_path)
        init_db(self.db_path)
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.dropped = 0
        self._t = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._t.start()

    def log(self, product_id, status, defect_area_px, confidence, actuated, frame_path, inference_ms):
        row = (product_id, status, int(defect_area_px), float(min(max(confidence, 0.0), 1.0)),
               bool(actuated), frame_path, float(inference_ms))
        try:
            self._q.put_nowait(row)
        except queue.Full:  # never block the inspection loop
            self.dropped += 1

    def _run(self) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        while not (self._stop.is_set() and self._q.empty()):
            try:
                rows = [self._q.get(timeout=0.2)]
            except queue.Empty:
                continue
            while len(rows) < 100:
                try:
                    rows.append(self._q.get_nowait())
                except queue.Empty:
                    break
            con.executemany(INSERT, rows)
            con.commit()
        con.close()

    def close(self) -> None:
        self._stop.set()
        self._t.join(timeout=5)
