import os
import sqlite3
import time

from edgeavi.storage.database import AsyncLogger, connect_readonly
from edgeavi.storage.retention import purge_once


def test_async_logger_and_constraints(tmp_path):
    db = tmp_path / "t.db"
    lg = AsyncLogger(db)
    for i in range(50):
        lg.log(f"P{i}", "Defective" if i % 2 else "Good", 300 * (i % 2), 0.9, i % 2 == 1, None, 12.5)
    lg.close()
    con = connect_readonly(db)
    assert con.execute("SELECT COUNT(*) FROM inspection_records").fetchone()[0] == 50
    plan = con.execute("EXPLAIN QUERY PLAN SELECT * FROM inspection_records WHERE classification_status='Good'").fetchall()
    assert any("idx_status" in str(r[3]) for r in plan)          # indexed access (troubleshooting #3)
    con.close()
    raw = sqlite3.connect(db)
    try:
        raw.execute("INSERT INTO inspection_records (product_id, classification_status, actuator_triggered) VALUES ('x','Bad',0)")
        assert False, "CHECK constraint should reject unknown status"
    except sqlite3.IntegrityError:
        pass


def test_fifo_purge(tmp_path):
    db = tmp_path / "t.db"
    lg = AsyncLogger(db)
    frames = []
    for i in range(10):
        f = tmp_path / f"f{i}.jpg"
        f.write_bytes(b"x")
        frames.append(f)
        lg.log(f"P{i}", "Defective", 500, 0.9, True, str(f), 10.0)
    lg.close()
    calls = {"n": 0}

    def fake_usage(_):                       # 90% for the first batch, then below threshold
        calls["n"] += 1
        return 90.0 if calls["n"] == 1 else 50.0

    assert purge_once(db, tmp_path, 85.0, batch=4, usage_fn=fake_usage) == 4
    con = sqlite3.connect(db)
    assert [r[0] for r in con.execute("SELECT product_id FROM inspection_records ORDER BY record_id")][0] == "P4"
    assert not os.path.exists(frames[0]) and os.path.exists(frames[9])
