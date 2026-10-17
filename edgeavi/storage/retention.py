"""Circular storage buffer: every N minutes, if disk usage > threshold, purge the
oldest records (and their saved frames) first-in-first-out."""
from __future__ import annotations

import os
import shutil
import sqlite3
import threading
from pathlib import Path


def disk_used_pct(path: str | Path) -> float:
    usage = shutil.disk_usage(path)
    return 100.0 * usage.used / usage.total


def purge_once(db_path, check_path, threshold_pct=85.0, batch=500, usage_fn=disk_used_pct) -> int:
    """Delete oldest records in batches until usage <= threshold. Returns rows deleted."""
    deleted = 0
    con = sqlite3.connect(db_path)
    try:
        while usage_fn(check_path) > threshold_pct:
            rows = con.execute(
                "SELECT record_id, image_frame_path FROM inspection_records "
                "ORDER BY record_id ASC LIMIT ?", (batch,)).fetchall()
            if not rows:
                break
            for _, frame in rows:
                if frame and os.path.exists(frame):
                    try:
                        os.remove(frame)
                    except OSError:
                        pass
            con.executemany("DELETE FROM inspection_records WHERE record_id=?", [(r[0],) for r in rows])
            con.commit()
            deleted += len(rows)
    finally:
        con.close()
    return deleted


class RetentionJob:
    def __init__(self, db_path, check_path, threshold_pct, interval_min, batch):
        self.args = (db_path, check_path, threshold_pct, batch)
        self.interval = interval_min * 60
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="retention", daemon=True)

    def start(self):
        self._t.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                n = purge_once(*self.args)
                if n:
                    print(f"[retention] purged {n} oldest records")
            except Exception as exc:  # never kill the engine because of housekeeping
                print(f"[retention] error: {exc}")

    def stop(self):
        self._stop.set()
