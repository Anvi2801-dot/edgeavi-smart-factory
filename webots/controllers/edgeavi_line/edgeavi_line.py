"""Webots Supervisor controller for the EdgeAVI digital twin.

Simulates: product spawning -> belt motion -> camera capture at the inspection zone
(REQ-1/2) -> frame sent to the inference engine over TCP (REQ-4/9) -> divert paddle
(REQ-12/13/15). Products are moved kinematically by the supervisor (no physics), which
keeps the simulation deterministic and light.

Start the engine first (python -m edgeavi.engine.server); this controller retries for 30 s.
Set Webots > Preferences > Python command to the venv python that has numpy + pyyaml.
"""
import csv
import random
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np                                    # noqa: E402
from controller import Supervisor                     # noqa: E402

from edgeavi.config import load_config                # noqa: E402
from edgeavi.ipc.protocol import recv_message, send_message, tune  # noqa: E402

# ---- scene geometry (metres, ENU: x along belt, y across belt, z up) -------------------
BELT_SPEED = 0.25
SPAWN_X, INSPECT_X, SORT_X, END_X = -1.4, 0.0, 0.6, 1.45
SPAWN_EVERY_S = 2.5            # >= paddle cycle (~1.7 s) so the actuator is always free
BELT_TOP_Z, PRODUCT_H, PRODUCT_SIDE = 0.50, 0.06, 0.12
PADDLE_REST_Y, PADDLE_SPEED, PADDLE_HALF_T = 0.45, 1.0, 0.01
OFF_BELT_Y = -0.40
DEFECT_PROB = 0.4
LIGHT_RANGE = (0.6, 1.5)


def rgb_str(c):
    return f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}"


def product_vrml(pid, x, y, yaw, colour, defects):
    parts = [f"""Shape {{ appearance Appearance {{ material Material {{ diffuseColor {rgb_str(colour)} }} }}
              geometry Box {{ size {PRODUCT_SIDE} {PRODUCT_SIDE} {PRODUCT_H} }} }}"""]
    for dx, dy, dyaw, sx, sy in defects:           # thin dark patches on the top face
        parts.append(f"""Transform {{ translation {dx:.4f} {dy:.4f} {PRODUCT_H / 2 + 0.0005}
              rotation 0 0 1 {dyaw:.4f} children [ Shape {{
              appearance Appearance {{ material Material {{ diffuseColor 0.08 0.05 0.03 }} }}
              geometry Box {{ size {sx:.4f} {sy:.4f} 0.001 }} }} ] }}""")
    z = BELT_TOP_Z + PRODUCT_H / 2
    return (f'Solid {{ name "{pid}" translation {x} {y} {z} rotation 0 0 1 {yaw:.4f} '
            f'children [ {" ".join(parts)} ] }}')


def random_defects(rng):
    out = []
    for _ in range(rng.randint(1, 3)):
        dx, dy = rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04)
        if rng.random() < 0.5:                      # blob
            s = rng.uniform(0.016, 0.03)
            out.append((dx, dy, rng.uniform(0, 3.14), s, s))
        else:                                       # scratch
            out.append((dx, dy, rng.uniform(0, 3.14), rng.uniform(0.04, 0.07), rng.uniform(0.006, 0.01)))
    return out


class Product:
    def __init__(self, pid, node, x, y, defective):
        self.pid, self.node, self.x, self.y, self.gt_defective = pid, node, x, y, defective
        self.trans = node.getField("translation")
        self.z = BELT_TOP_Z + PRODUCT_H / 2
        self.lit = self.inspected = False
        self.decision = None            # result dict from the engine
        self.diverting = False


class Line:
    def __init__(self):
        self.cfg = load_config()
        self.robot = Supervisor()
        self.dt = int(self.robot.getBasicTimeStep())
        self.cam = self.robot.getDevice("inspection_cam")
        self.cam.enable(self.dt)
        self.sun = self.robot.getFromDef("SUN").getField("intensity")
        self.paddle = self.robot.getFromDef("PADDLE").getField("translation")
        self.children = self.robot.getRoot().getField("children")
        self.rng = random.Random(7)
        self.products: list[Product] = []
        self.n_spawned = 0
        self.next_spawn = 0.5
        self.paddle_y, self.paddle_state, self.paddle_target = PADDLE_REST_Y, "idle", None
        self.sock = self.connect()
        gt_path = Path(self.cfg["paths"]["data_dir"]) / "ground_truth.csv"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        self.gt_file = open(gt_path, "w", newline="")
        self.gt = csv.writer(self.gt_file)
        self.gt.writerow(["product_id", "gt_defective", "engine_status", "defect_area_px", "confidence", "latency_ms"])
        self.confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    # ---- IPC -------------------------------------------------------------------------
    def connect(self, timeout_s=30):
        host, port = self.cfg["ipc"]["host"], self.cfg["ipc"]["port"]
        t0 = time.time()
        while True:
            try:
                s = socket.create_connection((host, port), timeout=5)
                tune(s)
                s.settimeout(10)
                print(f"[webots] connected to engine {host}:{port}")
                return s
            except OSError:
                if time.time() - t0 > timeout_s:
                    raise RuntimeError("engine not reachable - start `python -m edgeavi.engine.server` first")
                time.sleep(1)

    def inspect(self, p: Product):
        img = self.cam.getImage()                   # BGRA bytes
        send_message(self.sock, {"type": "frame", "product_id": p.pid, "w": self.cam.getWidth(),
                                 "h": self.cam.getHeight(), "c": 4, "fmt": "bgra"}, img)
        header, _ = recv_message(self.sock)
        if header["type"] != "result":
            raise RuntimeError(f"unexpected engine reply: {header}")
        p.decision, p.inspected = header, True
        d = header["status"] == "Defective"
        key = ("TP" if d else "FN") if p.gt_defective else ("FP" if d else "TN")
        self.confusion[key] += 1
        self.gt.writerow([p.pid, int(p.gt_defective), header["status"], header["defect_area_px"],
                          f"{header['confidence']:.3f}", f"{header['latency_ms']:.1f}"])

    # ---- scene -----------------------------------------------------------------------
    def spawn(self):
        self.n_spawned += 1
        pid = f"P{self.n_spawned:06d}"
        defective = self.rng.random() < DEFECT_PROB
        colour = [self.rng.uniform(0.35, 0.9) for _ in range(3)]
        x, y = SPAWN_X, self.rng.uniform(-0.05, 0.05)
        defects = random_defects(self.rng) if defective else []
        self.children.importMFNodeFromString(-1, product_vrml(pid, x, y, self.rng.uniform(-0.2, 0.2), colour, defects))
        node = self.children.getMFNode(-1)
        self.products.append(Product(pid, node, x, y, defective))

    def remove(self, p: Product):
        p.trans.setSFVec3f([0, 0, -10])      # park below the floor instead of node.remove()
        self.products.remove(p)

    def step_paddle(self, dt):
        if self.paddle_state == "extend":
            self.paddle_y -= PADDLE_SPEED * dt
            for p in self.products:
                if p.diverting:                      # paddle front face pushes the product
                    p.y = min(p.y, self.paddle_y - PADDLE_HALF_T - PRODUCT_SIDE / 2)
            tgt = self.paddle_target
            if tgt is not None and tgt.y <= OFF_BELT_Y:
                self.remove(tgt)
                self.paddle_target, self.paddle_state = None, "retract"
        elif self.paddle_state == "retract":         # REQ-15: back to default position
            self.paddle_y = min(PADDLE_REST_Y, self.paddle_y + PADDLE_SPEED * dt)
            if self.paddle_y >= PADDLE_REST_Y:
                self.paddle_state = "idle"
        self.paddle.setSFVec3f([SORT_X + 0.15, self.paddle_y, BELT_TOP_Z + 0.03])

    def run(self):
        dt = self.dt / 1000.0
        try:
            while self.robot.step(self.dt) != -1:
                t = self.robot.getTime()
                if t >= self.next_spawn:
                    self.spawn()
                    self.next_spawn = t + SPAWN_EVERY_S
                for p in list(self.products):
                    p.x += BELT_SPEED * dt
                    if not p.lit and p.x >= INSPECT_X - 0.15:      # lighting varies per product
                        self.sun.setSFFloat(self.rng.uniform(*LIGHT_RANGE))
                        p.lit = True
                    if not p.inspected and p.x >= INSPECT_X:
                        self.inspect(p)                              # REQ-2 + REQ-4
                    if (p.inspected and not p.diverting and p.decision["actuate"]
                            and p.x >= SORT_X and self.paddle_state == "idle"):
                        p.diverting, self.paddle_state, self.paddle_target = True, "extend", p
                    p.trans.setSFVec3f([p.x, p.y, p.z])
                    if p.x >= END_X and not p.diverting:             # good products leave the belt
                        self.remove(p)
                self.step_paddle(dt)
        finally:
            self.shutdown()

    def shutdown(self):
        try:
            send_message(self.sock, {"type": "shutdown"})           # REQ-5
            self.sock.close()
        except OSError:
            pass
        self.gt_file.close()
        c = self.confusion
        n = sum(c.values()) or 1
        print(f"[webots] inspected={n} confusion={c} accuracy={100 * (c['TP'] + c['TN']) / n:.1f}%")


if __name__ == "__main__":
    Line().run()
