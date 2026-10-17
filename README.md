# EdgeAVI - Smart Factory Edge AI Quality Engine

Webots digital twin -> TCP -> INT8 MobileNetV3-UNet (ONNX Runtime) -> divert paddle,
async SQLite log, Streamlit dashboard. Everything runs locally/offline.

```
Webots controller ──frames──▶ engine (edgeavi.engine.server) ──result──▶ controller → paddle
                                   │  ├─ SQLite (async writer thread)
                                   │  └─ data/live/{latest.jpg,status.json}
                                   ▼
                          Streamlit dashboard (reads DB + live files, writes runtime_settings.json)
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Install Webots R2023b+ and set **Preferences → Python command** to this venv's python.

## Pipeline (run from the repo root)
```bash
python -m edgeavi.ml.synthetic --out data/synthetic --train 1500 --val 300   # 1 data
python -m edgeavi.ml.train --epochs 15 --batch 16                            # 2 train FP32
python -m edgeavi.ml.export_onnx                                             # 3 ONNX (opset 13)
python -m edgeavi.ml.quantize --samples 200                                  # 4 INT8 PTQ
python -m edgeavi.ml.evaluate                                                # 5 FP32 vs INT8
python -m edgeavi.engine.benchmark                                           # 6 latency check
python -m edgeavi.engine.server                                              # 7 engine
streamlit run edgeavi/dashboard/app.py                                       # 8 dashboard
# 9: open webots/worlds/factory.wbt in Webots and press Play
python -m pytest -q tests                                                    # unit/integration tests
```
