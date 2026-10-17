"""Streamlit monitoring dashboard (REQ-16..20).
    streamlit run edgeavi/dashboard/app.py
Reads: data/live/latest.jpg + status.json (written by the engine) and the SQLite log.
Writes: data/live/runtime_settings.json (operator threshold control).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pandas as pd
import streamlit as st

from edgeavi.config import load_config
from edgeavi.storage.database import connect_readonly

cfg = load_config()
LIVE = Path(cfg["paths"]["live_dir"])
DB = cfg["paths"]["db_path"]

st.set_page_config(page_title="EdgeAVI Quality Monitor", layout="wide")
st.title("Smart Factory Edge AI Quality Engine")

# ---- sidebar: role + threshold control -------------------------------------------------
role = st.sidebar.selectbox("Role", ["System Operator", "Quality Auditor"])
if role == "Quality Auditor" and os.environ.get("EDGEAVI_AUDITOR_PIN"):
    if st.sidebar.text_input("Auditor PIN", type="password") != os.environ["EDGEAVI_AUDITOR_PIN"]:
        st.sidebar.warning("Enter PIN to unlock auditor features")
        role = "System Operator"

settings_path = LIVE / "runtime_settings.json"
try:
    current_thr = int(json.loads(settings_path.read_text())["defect_area_threshold_px"])
except Exception:
    current_thr = cfg["inference"]["defect_area_threshold_px"]
new_thr = st.sidebar.slider("Defect area threshold (px, 640x480 frame)", 20, 2000, current_thr, 10)
if new_thr != current_thr:                                  # REQ-20
    LIVE.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"defect_area_threshold_px": new_thr}))
    st.sidebar.success(f"Threshold set to {new_thr} px")


def read_status() -> dict | None:
    try:
        return json.loads((LIVE / "status.json").read_text())
    except Exception:
        return None


def query(sql: str, params=()) -> pd.DataFrame:
    try:
        con = connect_readonly(DB)
        try:
            return pd.read_sql_query(sql, con, params=params)
        finally:
            con.close()
    except sqlite3.Error:
        return pd.DataFrame()


live_tab, history_tab = st.tabs(["Live", "History & Export"])

with live_tab:
    @st.fragment(run_every=1.0)                              # REQ-13/18: auto-refresh
    def live():
        s = read_status()
        online = bool(s) and time.time() - s["updated"] < 5
        st.markdown(f"**Engine:** {'🟢 online' if online else '🔴 offline / idle'}")
        left, right = st.columns([3, 2])
        with left:
            img = LIVE / "latest.jpg"
            if img.exists():
                st.image(img.read_bytes(), caption="Inspection zone (mask overlay on defects)", use_container_width=True)
            else:
                st.info("Waiting for the first inspected frame...")
        with right:
            if s:
                a, b = st.columns(2)
                a.metric("Inspected", s["total"])
                b.metric("Defective", s["defects"])
                a.metric("Yield", f"{s['yield_pct']:.1f}%")
                b.metric("Latency (last)", f"{s['latency_last_ms']:.1f} ms",
                         delta=f"{s['latency_last_ms'] - s['latency_budget_ms']:.1f} vs budget", delta_color="inverse")
                a.metric("Latency (mean)", f"{s['latency_mean_ms']:.1f} ms")
                b.metric("Latency (p95)", f"{s['latency_p95_ms']:.1f} ms")
        recent = query("SELECT product_id, timestamp, classification_status AS status, defect_area_pixels AS area_px, "
                       "ROUND(confidence_score,3) AS confidence, actuator_triggered AS actuated, "
                       "ROUND(inference_ms,1) AS ms FROM inspection_records ORDER BY record_id DESC LIMIT 12")
        st.dataframe(recent, use_container_width=True, hide_index=True)

    live()

with history_tab:                                            # REQ-19
    c1, c2 = st.columns(2)
    status = c1.selectbox("Status", ["All", "Defective", "Good"])
    limit = c2.number_input("Max rows", 50, 50000, 1000, 50)
    sql = ("SELECT record_id, product_id, timestamp, classification_status, defect_area_pixels, confidence_score, "
           "actuator_triggered, inference_ms, image_frame_path FROM inspection_records ")
    params: tuple = ()
    if status != "All":
        sql += "WHERE classification_status = ? "         # uses idx_status
        params = (status,)
    sql += "ORDER BY timestamp DESC, record_id DESC LIMIT ?"
    df = query(sql, params + (int(limit),))
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        per_min = (df.assign(minute=pd.to_datetime(df["timestamp"]).dt.floor("min"))
                     .groupby(["minute", "classification_status"]).size().unstack(fill_value=0))
        st.bar_chart(per_min)
    if role == "Quality Auditor":
        st.download_button("Export inspection log (CSV)", df.to_csv(index=False).encode(), "inspection_log.csv", "text/csv")
    else:
        st.caption("Switch to the Quality Auditor role to export logs.")
