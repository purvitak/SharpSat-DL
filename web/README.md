# SharpSat — Web Dashboard

An end-to-end, hardware-optimized Deep Learning system for Super-Resolution Mapping (SRM) of satellite imagery (Sentinel-2 L2A). Built for Smart India Hackathon (SIH26142 / NTRO).

## Key Features

1. **Input → Output Super-Resolution Pipeline**: Upgrades medium-resolution satellite imagery (30m/40m) to native 10m high-resolution outputs using an Attention-RRDB generator (accuracy model) or its GAN-refined version (visual model).
2. **Spectral & Geographic Consistency**: Loss function incorporates **Charbonnier Pixel Loss**, **Spectral NDVI Consistency Loss**, and **Feature Representation Loss** to prevent hallucinations.
3. **Explicit Uncertainty Management**: **Monte Carlo Dropout** inference (N stochastic passes) computes pixel-wise standard deviation, producing an **Inferno Uncertainty Heatmap** highlighting inferred vs directly observed features.
4. **Remote Sensing Evaluation**: Calculates **PSNR**, **SSIM**, **SAM (Spectral Angle Mapper)**, and **ERGAS**.
5. **Persistent Upload History**: Automatically saves uploaded and processed images, parameters, thumbnails, and metrics into a persistent SQLite store. Past runs can be re-loaded into the inspector at any time.
6. **Interactive Web Studio**: Glassmorphism web dashboard featuring an interactive curtain split-viewer, view mode switches (RGB, Uncertainty, NDVI), and metric counters.

---

## Quick Start

The web app only runs inference. Models are trained from the repository root (see the root README), then copied in here.

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Add trained checkpoints and demo pairs
From the repository root, after `python -m src.train_sen2naip --stage both`:
```bash
mkdir -p web/checkpoints
cp checkpoints/best_model_attn_sen2naip.pt checkpoints/best_model_attn_gan_sen2naip.pt web/checkpoints/
python -m src.data.sen2naip --export_web 8     # 8 held-out test sites -> web/data/pairs/
```
Models are listed in `configs/config.yaml`; any whose checkpoint is missing appears greyed out in the UI. For the exported pairs, the model input is the real 10 m Sentinel-2 image and the ground truth is the 2.5 m aerial image; for other tiles and uploads, the input is made by blurring and shrinking the image ×4.

### 3. Launch the dashboard
```bash
python -m src.server
```
Open browser at: `http://localhost:8000`. If neither `data/pairs/` nor `data/raw_tiles/` has anything, a synthetic placeholder tile is created.

---

## Project Structure

```
web/
├── configs/
│   └── config.yaml             # Model registry (checkpoint paths and labels)
├── checkpoints/                # Trained .pt files (copied in, gitignored)
├── data/                       # Sample tiles + SQLite run history (gitignored)
├── src/
│   ├── data/                   # degrade (HR -> LR) and sample-tile loader
│   ├── models/                 # RRDBNet and Attention-RRDB
│   ├── losses/                 # NDVI computation
│   ├── uncertainty/            # MC-Dropout engine & heatmaps
│   ├── eval/                   # PSNR, SSIM, SAM, ERGAS metrics
│   ├── db/                     # SQLite persistent history store
│   ├── model_registry.py       # Loads and caches checkpoints, auto-detects architecture
│   └── server.py               # FastAPI backend web server
└── frontend/                   # Interactive web app
    ├── index.html
    ├── styles.css
    └── app.js
```
