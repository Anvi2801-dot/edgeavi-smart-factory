"""End-to-end: fake controller -> TCP -> engine (dummy ONNX) -> DB + live files."""
import json
import socket
import sqlite3
import threading
import time

import numpy as np

from edgeavi.config import load_config
from edgeavi.engine.server import Engine
from edgeavi.ipc.protocol import recv_message, send_message


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_engine_end_to_end(tmp_path, dummy_model):
    cfg = load_config()
    for k, v in (("data_dir", "d"), ("live_dir", "d/live"), ("frames_dir", "d/frames")):
        cfg["paths"][k] = str(tmp_path / v)
    cfg["paths"]["db_path"] = str(tmp_path / "d" / "log.db")
    cfg["inference"]["providers"] = ["CPUExecutionProvider"]
    eng = Engine(cfg, dummy_model)
    port = free_port()
    threading.Thread(target=eng.serve, args=("127.0.0.1", port), daemon=True).start()
    time.sleep(0.5)

    good = np.full((480, 640, 3), 140, np.uint8)
    bad = good.copy(); bad[200:260, 300:360] = 5          # dark 60x60 patch
    with socket.create_connection(("127.0.0.1", port)) as s:
        for pid, frame in (("P1", good), ("P2", bad)):
            bgra = np.dstack([frame, np.full(frame.shape[:2], 255, np.uint8)])
            send_message(s, {"type": "frame", "product_id": pid, "w": 640, "h": 480, "c": 4, "fmt": "bgra"}, bgra.tobytes())
            res, _ = recv_message(s)
            print(res)
            assert res["status"] == ("Defective" if pid == "P2" else "Good")
            assert res["actuate"] == (pid == "P2") and res["latency_ms"] > 0
        send_message(s, {"type": "shutdown"})
    time.sleep(0.8)
    rows = sqlite3.connect(cfg["paths"]["db_path"]).execute(
        "SELECT product_id, classification_status, actuator_triggered FROM inspection_records ORDER BY record_id").fetchall()
    assert rows == [("P1", "Good", 0), ("P2", "Defective", 1)]
    status = json.loads((tmp_path / "d" / "live" / "status.json").read_text())
    assert status["total"] == 2 and status["defects"] == 1 and status["yield_pct"] == 50.0
    assert (tmp_path / "d" / "live" / "latest.jpg").exists()
    # REQ-20: dashboard-style threshold override is picked up live
    (tmp_path / "d" / "live" / "runtime_settings.json").write_text(json.dumps({"defect_area_threshold_px": 100000}))
    assert eng.threshold() == 100000


def test_int8_quantization_pipeline(tmp_path, dummy_model):
    import cv2
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from edgeavi.engine.inference import InferenceEngine
    from edgeavi.ml.quantize import FolderCalibrationReader
    from edgeavi.ml.synthetic import render_sample
    from pathlib import Path

    rng = np.random.default_rng(0)
    d = tmp_path / "cal"; d.mkdir()
    for i in range(8):
        cv2.imwrite(str(d / f"{i}.png"), render_sample(rng, bool(i % 2))[0])
    out = str(tmp_path / "int8.onnx")
    quantize_static(dummy_model, out, FolderCalibrationReader(d, 256, 320, 8),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    eng = InferenceEngine(out, ["CPUExecutionProvider"])
    assert eng.run(np.zeros((1, 3, 256, 320), np.float32)).shape == (256, 320)
