"""Inference engine process: TCP server that receives frames from the Webots
controller, runs INT8 inference, answers with a classification, and logs asynchronously.

    python -m edgeavi.engine.server            # uses models/model_int8.onnx
    python -m edgeavi.engine.server --model models/model_fp32.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from edgeavi.config import load_config
from edgeavi.engine.inference import InferenceEngine
from edgeavi.engine.postprocess import analyze, render_overlay
from edgeavi.engine.preprocess import preprocess
from edgeavi.ipc.protocol import recv_message, send_message, tune
from edgeavi.storage.database import AsyncLogger
from edgeavi.storage.retention import RetentionJob


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)          # readers never see a half-written file


class Stats:
    def __init__(self):
        self.total = self.defects = 0
        self.lat = deque(maxlen=300)

    def update(self, status: str, ms: float):
        self.total += 1
        self.defects += status == "Defective"
        self.lat.append(ms)

    def snapshot(self, budget_ms: float) -> dict:
        lat = np.array(self.lat) if self.lat else np.zeros(1)
        return {"updated": time.time(), "total": self.total, "defects": self.defects,
                "yield_pct": 100.0 * (self.total - self.defects) / self.total if self.total else 100.0,
                "latency_last_ms": float(lat[-1]), "latency_mean_ms": float(lat.mean()),
                "latency_p95_ms": float(np.percentile(lat, 95)), "latency_budget_ms": budget_ms}


class Engine:
    def __init__(self, cfg: dict, model_path: str):
        self.cfg = cfg
        inf = cfg["inference"]
        self.infer = InferenceEngine(model_path, inf["providers"])
        self.live = Path(cfg["paths"]["live_dir"]); self.live.mkdir(parents=True, exist_ok=True)
        self.frames = Path(cfg["paths"]["frames_dir"]); self.frames.mkdir(parents=True, exist_ok=True)
        self.logger = AsyncLogger(cfg["paths"]["db_path"], cfg["storage"]["queue_size"])
        self.stats = Stats()
        self.settings_path = self.live / "runtime_settings.json"
        self._thr = inf["defect_area_threshold_px"]; self._thr_mtime = 0.0
        self.saver_q: queue.Queue = queue.Queue(maxsize=64)
        threading.Thread(target=self._saver, name="frame-saver", daemon=True).start()
        self.retention = RetentionJob(cfg["paths"]["db_path"], cfg["paths"]["data_dir"],
                                      cfg["storage"]["disk_threshold_pct"],
                                      cfg["storage"]["purge_interval_min"], cfg["storage"]["purge_batch"])
        self.retention.start()

    def threshold(self) -> int:
        """REQ-20: dashboard writes runtime_settings.json; pick it up without a restart."""
        try:
            m = self.settings_path.stat().st_mtime
            if m != self._thr_mtime:
                self._thr_mtime = m
                self._thr = int(json.loads(self.settings_path.read_text())["defect_area_threshold_px"])
                print(f"[engine] defect area threshold -> {self._thr} px")
        except (FileNotFoundError, KeyError, ValueError):
            pass
        return self._thr

    def _saver(self):
        """Overlay rendering + JPEG encoding happen off the hot path."""
        while True:
            frame, mask, status, pid, ms, frame_path = self.saver_q.get()
            img = render_overlay(frame, mask, status, pid, ms)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                atomic_write(self.live / "latest.jpg", buf.tobytes())
                if frame_path:
                    Path(frame_path).write_bytes(buf.tobytes())

    def handle_frame(self, header: dict, payload: bytes) -> dict:
        t0 = time.perf_counter()
        frame = np.frombuffer(payload, np.uint8).reshape(header["h"], header["w"], header["c"])
        if header.get("fmt") == "bgra":
            frame = np.ascontiguousarray(frame[:, :, :3])
        tensor = preprocess(frame, self.infer.height, self.infer.width)
        logits = self.infer.run(tensor)
        res = analyze(logits, frame.shape[:2], self.cfg["inference"]["mask_threshold"], self.threshold())
        ms = (time.perf_counter() - t0) * 1000.0          # preprocess + inference + postprocess

        actuate = res.status == "Defective"
        pid = header["product_id"]
        frame_path = str(self.frames / f"{pid}.jpg") if (actuate and self.cfg["storage"]["save_defect_frames"]) else None
        self.logger.log(pid, res.status, res.defect_area_px, res.confidence, actuate, frame_path, ms)
        self.stats.update(res.status, ms)
        try:
            self.saver_q.put_nowait((frame.copy(), res.mask, res.status, pid, ms, frame_path))
        except queue.Full:
            pass
        atomic_write(self.live / "status.json", json.dumps(self.stats.snapshot(self.cfg["inference"]["latency_budget_ms"])).encode())
        return {"type": "result", "product_id": pid, "status": res.status, "defect_area_px": res.defect_area_px,
                "confidence": res.confidence, "actuate": actuate, "latency_ms": ms}

    def serve(self, host: str, port: int):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port)); srv.listen(1)
        print(f"[engine] listening on {host}:{port} - start the Webots simulation")
        try:
            while True:
                conn, addr = srv.accept()
                tune(conn)
                print("[engine] controller connected", addr)
                try:
                    while True:
                        header, payload = recv_message(conn)
                        if header["type"] == "frame":
                            send_message(conn, self.handle_frame(header, payload))
                        elif header["type"] == "shutdown":      # REQ-5
                            print("[engine] shutdown signal received")
                            break
                except (ConnectionError, OSError) as exc:
                    print("[engine] connection ended:", exc)
                finally:
                    conn.close()
        except KeyboardInterrupt:
            print("\n[engine] stopping")
        finally:
            self.retention.stop(); self.logger.close(); srv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="defaults to the INT8 model from the config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    Engine(cfg, a.model or cfg["model"]["int8_onnx"]).serve(cfg["ipc"]["host"], cfg["ipc"]["port"])


if __name__ == "__main__":
    main()
