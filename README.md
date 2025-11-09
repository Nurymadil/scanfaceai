# fWHR Scan

FastAPI application that estimates the facial width-to-height ratio (fWHR) for every face found in an uploaded photo. It uses TorchLM (FaceBoxesV2 + PIPNet-98) for detection/landmarks and serves a Tailwind-styled web UI with drag-and-drop uploads, webcam capture, automatic analysis, per-face visualizations, manual filtering, history, and a light/dark theme toggle.

## Features
- Instant analysis once a file is selected or a camera shot is captured.
- Supports multiple faces in one frame and returns mean/min/max fWHR metrics.
- Shows the original photo alongside grayscale landmark overlays for each detected face.
- Provides capture tips, error messaging, and a modern glassmorphism-inspired layout.

## Requirements
- Python 3.10+
- PyTorch with the desired backend (CPU, CUDA, or MPS).
- PIPNet checkpoint for the WFLW-98 landmark model (`pipnet_resnet.pth` already expected in the repo).

Install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** PyTorch/TorchLM wheels depend on your OS and accelerator. Consult [pytorch.org](https://pytorch.org/get-started/locally/) for the correct install command if the generic `pip install torch` does not match your environment.

## Running the App

```bash
uvicorn testo_app:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://127.0.0.1:8000/` in a browser. The UI offers two modes:

1. **Файл** – drag a JPEG/PNG into the dropzone (analysis runs automatically).
2. **Веб-камера** – grant permission, frame your face, click “Сделать снимок” (auto-analysis).

Results show:
- Count of detected faces with individual fWHR badges.
- Landmark overlays for each face.
- Aggregate statistics (mean/min/max fWHR), original uploaded image, manual face selection stats, and browser-side history with sparklines.

### Configuration

Settings can be controlled via environment variables (see `.env.example` for defaults). Key options:

- `FWHR_CHECKPOINT_PATH` – path to the PIPNet checkpoint.
- `FWHR_DEVICE_PREFERENCE` – `auto`, `cpu`, `cuda`, or `mps`.
- `FWHR_MAX_CONCURRENT_INFERENCE` – throttle concurrent analyses.
- `FWHR_HISTORY_LIMIT` – number of stored browser-side history entries.
- `FWHR_LOG_LEVEL` – logging verbosity (structured JSON logs).

Create a local `.env` when running outside Docker:

```bash
cp .env.example .env
```

## Project Structure

```
testo_app.py        # FastAPI app, inference pipeline, HTML template bindings
templates/
  └── index.html    # Tailwind-based UI rendered by Jinja2
requirements.txt    # Python dependencies
Dockerfile.cpu/gpu  # Container builds for CPU / CUDA runtimes
docker-compose.yml  # Local orchestration example
.github/workflows   # CI pipeline (tests + Docker builds to GHCR)
tests/              # pytest suite (fWHR math + API smoke tests)
pipnet_resnet.pth   # PIPNet-98 checkpoint for landmark detection
testo_model.pth     # (Optional) additional weights/sample data
testo_proto.py      # Experimental scripts/prototypes
```

## Customization Tips
- Update `CHECKPOINT_PATH` in `testo_app.py` if you relocate or swap the landmark weights.
- Adjust UI copy, Tailwind classes, or add analytics inside `templates/index.html`.
- Wrap the FastAPI app with reverse proxies (e.g., nginx) or package via Docker for deployment.
- Consider exporting the model to ONNX/TorchScript for faster CPU inference when deploying at scale.
- Extend the JS history widget to sync with a backend if you need cross-device comparisons, or hook Prometheus/OpenTelemetry into the Python metrics when deploying to production.

## Testing

```bash
pytest
```

Unit tests cover the fWHR helper math plus FastAPI smoke tests (the inference path is mocked to keep tests lightweight).

## Docker & CI

Build CPU image locally:

```bash
docker build -t fwhr-scan:cpu -f Dockerfile.cpu .
```

Or use `docker-compose up` to load `.env` and expose the service on port 8000.

GitHub Actions workflow (`.github/workflows/ci.yml`) runs pytest and builds/pushes both CPU and GPU images to GHCR on every push to `main`.

Happy experimenting! Feel free to extend the project with history tracking, authentication, or additional anthropometric metrics. 
