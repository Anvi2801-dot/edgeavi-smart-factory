# EdgeAVI - Smart Factory Edge AI Quality Engine

An offline, edge-AI visual inspection system for a simulated factory line. A Webots digital twin sends camera frames of products on a conveyor belt to a local inference engine. An INT8 MobileNetV3-UNet segmentation model (ONNX Runtime) classifies each product as **Good** or **Defective**, and a sorting paddle diverts defective items. Every result is logged to SQLite and shown on a Streamlit dashboard. Nothing leaves the machine, and no cloud services or internet connection are needed.

## How it works

```
Webots controller --frames--> engine (edgeavi.engine.server) --result--> controller --> paddle
                                   |  +- SQLite (async writer thread)
                                   |  +- data/live/{latest.jpg, status.json}
                                   v
                          Streamlit dashboard
                          (reads the DB and live files, writes runtime_settings.json)
```

1. The Webots supervisor spawns products on the belt with random colours, lighting and defects.
2. When a product reaches the inspection zone, one frame is captured and sent to the engine over a local TCP socket.
3. The engine preprocesses the frame, runs the INT8 model, measures the defect area in pixels and applies the area threshold.
4. The result (status, defect area, confidence, latency) goes back to the controller. The paddle pushes defective products into the reject bin.
5. The engine logs each result asynchronously and updates the live files that the dashboard reads.

## Features

- Real-time defect segmentation with an INT8-quantized MobileNetV3-UNet
- Per-frame latency measured against a 40 ms budget
- Synthetic training data generator that imitates the simulator camera, with varied lighting, colour and defect shapes
- Safe failure: if the engine is unreachable, the controller stops with an error instead of sorting blindly
- Async SQLite logging with constraints, indexes and a FIFO retention job that purges the oldest records when disk usage passes a limit
- Streamlit dashboard with a live frame, counters, yield and latency, history filters, a per-minute chart, CSV export and a live threshold slider
- Ground-truth logging in the simulator, with a confusion-matrix report

## Repo structure

```
edgeavi-smart-factory/
├── configs/config.yaml            # paths, thresholds, ports, retention settings
├── edgeavi/
│   ├── config.py
│   ├── ipc/protocol.py            # length-prefixed TCP messages
│   ├── ml/                        # synthetic data, dataset, model, train, export, quantize, evaluate
│   ├── engine/                    # preprocess, inference, postprocess, server, benchmark
│   ├── storage/                   # async SQLite logger, retention job
│   └── dashboard/app.py           # Streamlit dashboard
├── webots/
│   ├── worlds/factory.wbt
│   └── controllers/edgeavi_line/edgeavi_line.py
├── tests/
├── confusion.py                   # confusion matrix from the newest ground-truth CSV
└── requirements.txt
```

`data/` and `models/` are not tracked by git. Generated data, logs and trained models stay local.

## Requirements

- Python 3.10 or newer
- [Webots](https://cyberbotics.com) R2023b or newer (developed with R2025a)
- A GPU is only needed for faster training. The engine runs on CPU.

## Setup

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

In Webots, open **Preferences** and set **Python command** to this venv's interpreter, for example `/path/to/edgeavi-smart-factory/venv/bin/python` (Windows: `...\venv\Scripts\python.exe`). The controller needs `numpy` and `pyyaml` from that environment. Restart Webots after changing it.

## Training the model

Run the pipeline from the repo root. A GPU makes training much faster, so Google Colab with a T4 runtime is a good option (see the next section).

```bash
python -m edgeavi.ml.synthetic --out data/synthetic --train 1500 --val 300   # 1. generate data
python -m edgeavi.ml.train --epochs 15 --batch 16                            # 2. train FP32
python -m edgeavi.ml.export_onnx                                             # 3. export ONNX (opset 13)
python -m edgeavi.ml.quantize --samples 200                                  # 4. INT8 quantization
python -m edgeavi.ml.evaluate                                                # 5. compare FP32 and INT8
```

This produces `models/best.pt`, `models/model_fp32.onnx` and `models/model_int8.onnx`.

### Training on Google Colab

1. Open a notebook and choose **Runtime, Change runtime type, T4 GPU**.
2. Run these cells:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/Anvi2801-dot/edgeavi-smart-factory.git
%cd /content/edgeavi-smart-factory
!pip install -q onnx onnxruntime pyyaml opencv-python-headless

!python -m edgeavi.ml.synthetic --out data/synthetic --train 1500 --val 300
!python -m edgeavi.ml.train --epochs 15 --batch 16
!python -m edgeavi.ml.export_onnx
!python -m edgeavi.ml.quantize --samples 200
!python -m edgeavi.ml.evaluate
!ls -lh models/

!mkdir -p /content/drive/MyDrive/edgeavi_models
!cp models/* /content/drive/MyDrive/edgeavi_models/
```

3. Download `model_int8.onnx` and `model_fp32.onnx` from Drive into your local `models/` folder.

Colab already includes `torch` and `torchvision`, so don't reinstall them. If you change `input_height` or `input_width` in `configs/config.yaml`, retrain, because the exported model has a fixed input size.

## Running the system

Start the pieces in this order, each in its own terminal, from the repo root with the venv active.

```bash
# 1. Check latency on this machine (optional)
python -m edgeavi.engine.benchmark

# 2. Inference engine. Wait for "listening on 127.0.0.1:5555"
python -m edgeavi.engine.server

# 3. Dashboard, opens at http://localhost:8501
streamlit run edgeavi/dashboard/app.py
```

4. In Webots, open `webots/worlds/factory.wbt` and press **Play**. The controller retries the engine connection for 30 seconds.

Engine options:

| Option | Meaning |
|---|---|
| `--model PATH` | Use another ONNX model (default is the INT8 model from the config) |
| `--config PATH` | Use another config file |
| `--fresh` | Delete previous logs and saved frames on startup |

Logs persist between runs by default. Use `--fresh` only for demos or a clean test session.

## Dashboard

- **Live tab:** engine online/offline status, the latest inspected frame with the defect mask, counters (inspected, defective, yield), latency (last, mean, p95 versus the 40 ms budget) and the 12 most recent records.
- **History & Export tab:** full inspection history with a status filter, a per-minute chart, and CSV export for the Quality Auditor role.
- **Threshold slider:** changes the defect-area threshold (pixels in the original 640x480 frame). The engine picks up the change on the next frame, with no restart.
- **Roles:** System Operator and Quality Auditor. Set the `EDGEAVI_AUDITOR_PIN` environment variable to require a PIN for the auditor role. This is an interface split, not real access control.

## Evaluating accuracy in simulation

The controller knows which products it made defective and records that next to the engine's decision. Each run writes one file, `data/ground_truth_<timestamp>.csv`, with a row per inspected product.

1. Start the engine and press Play. Let the simulation run for several minutes.
2. Press **Pause**, not Reload. Rows are saved as they are written.
3. Run the report:

```bash
python confusion.py
```

It reads the newest ground-truth file and prints the sample size and TP, TN, FP, FN and accuracy. Check that `n` is large enough (100 or more) before quoting an accuracy figure.

## Results

These numbers come from the simulation on the author's machine and are not guarantees:

- Per-frame latency during live runs stayed within the 40 ms budget (about 17 to 36 ms).
- A 5-minute simulated run of 189 products gave TP 83, TN 106, FP 0, FN 0 (100% accuracy).

The simulation uses a fixed random seed, so repeating the same run produces the same products. Change the seed in `edgeavi_line.py` to test other sequences. Perfect accuracy on this simulated set says the pipeline works end to end. It doesn't predict accuracy on real factory images.

## Configuration

Everything is in `configs/config.yaml`. The main settings:

| Key | Meaning |
|---|---|
| `model.input_height`, `model.input_width` | Model input size (divisible by 32) |
| `inference.defect_area_threshold_px` | Minimum flagged area, in original-frame pixels, to call a product Defective |
| `inference.mask_threshold` | Per-pixel probability cut-off |
| `inference.latency_budget_ms` | Latency budget shown on the dashboard |
| `ipc.host`, `ipc.port` | Engine address (default `127.0.0.1:5555`) |
| `storage.disk_threshold_pct`, `storage.purge_interval_min` | When the retention job purges the oldest records |
| `storage.save_defect_frames` | Save an overlay image for each defective product |

## Outputs

| Path | Content |
|---|---|
| `data/inspection_logs.db` | SQLite log of every inspection (timestamps stored in UTC) |
| `data/frames/` | Overlay images of defective products |
| `data/live/` | `latest.jpg`, `status.json` and `runtime_settings.json` for the dashboard |
| `data/ground_truth_*.csv` | Simulation ground truth, one file per run |

Query the database directly:

```bash
sqlite3 data/inspection_logs.db "SELECT * FROM inspection_records ORDER BY record_id DESC LIMIT 10;"
```

## Tests

```bash
python -m pytest -q tests
```

The suite covers the TCP protocol, the async database logger and its constraints, the FIFO retention logic, postprocessing and threshold behaviour, the synthetic data generator, INT8 quantization, and an end-to-end engine run with a stand-in model.

## Troubleshooting

| Problem | Fix |
|---|---|
| Webots console shows "engine not reachable" | Start `python -m edgeavi.engine.server` first, then **Reload** the world in Webots and press Play. |
| Engine fails to start | Check that `models/model_int8.onnx` exists. Copy the trained models from Colab or Drive. |
| Dashboard shows "offline / idle" | The engine or simulation isn't running, or no frame has been inspected in the last 5 seconds. |
| Product boxes appear on the belt before pressing Play | Product nodes were saved into `factory.wbt`. Remove them from the file. When Webots asks "Save changes?" after a run, choose **Don't Save**. |
| `ground_truth` CSV is missing or empty | Webots was reloaded before the first product was inspected. Press Pause (not Reload) when finished. |
| Latency above 40 ms | Reduce `input_height` and `input_width` in the config, retrain, and re-export. |

## Known limitations

- The model is trained on synthetic images that imitate the simulator camera. Real factory images would need new data and fine-tuning.
- Products are moved kinematically, without physics.
- A single sequential inference stream tops out near 25 products per second at 40 ms per image.
- Dashboard roles are an interface split, not a security boundary.
- The Python parts are covered by automated tests. The Webots world and controller are checked by running the simulation.

## Future work

Real industrial camera and conveyor integration, more product categories and defect types, fine-tuning on real images, and richer production analytics on the dashboard.