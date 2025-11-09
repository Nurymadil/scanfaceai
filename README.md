# fWHR Scan

FastAPI application that estimates the facial width-to-height ratio (fWHR) for every face found in an uploaded photo. It uses TorchLM (FaceBoxesV2 + PIPNet-98) for detection/landmarks and serves a Tailwind-styled web UI with drag-and-drop uploads, webcam capture, automatic analysis, and per-face visualizations.

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
pip install fastapi uvicorn python-multipart jinja2 numpy pillow matplotlib torch torchlm
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
- Aggregate statistics (mean/min/max fWHR) and the original uploaded image.

## Project Structure

```
testo_app.py        # FastAPI app, inference pipeline, HTML template bindings
templates/
  └── index.html    # Tailwind-based UI rendered by Jinja2
pipnet_resnet.pth   # PIPNet-98 checkpoint for landmark detection
testo_model.pth     # (Optional) additional weights/sample data
testo_proto.py      # Experimental scripts/prototypes
```

## Customization Tips
- Update `CHECKPOINT_PATH` in `testo_app.py` if you relocate or swap the landmark weights.
- Adjust UI copy, Tailwind classes, or add analytics inside `templates/index.html`.
- Wrap the FastAPI app with reverse proxies (e.g., nginx) or package via Docker for deployment.
- Consider exporting the model to ONNX/TorchScript for faster CPU inference when deploying at scale.

Happy experimenting! Feel free to extend the project with history tracking, authentication, or additional anthropometric metrics. 
