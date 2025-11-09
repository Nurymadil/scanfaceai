import asyncio
import base64
import io
import json
import logging
import logging.config
import time
import uuid
from threading import Lock
from datetime import datetime
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import torchlm
from torchlm.models import pipnet
from torchlm.tools import faceboxesv2

# ===============================
#   Настройки и вспомогательные функции
# ===============================


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FWHR_")

    checkpoint_path: str = Field(default="pipnet_resnet.pth")
    device_preference: str = Field(default="auto")
    max_concurrent_inference: int = Field(default=2, ge=1, le=8)
    log_level: str = Field(default="INFO")
    history_limit: int = Field(default=5, ge=1, le=12)
    enable_viz_cache: bool = Field(default=True)


settings = Settings()

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "json": {
            "format": "%(message)s",
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
        }
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        }
    },
    "root": {
        "handlers": ["default"],
        "level": settings.log_level.upper(),
    },
}

try:
    logging.config.dictConfig(LOGGING_CONFIG)
except Exception:
    logging.basicConfig(level=settings.log_level.upper())

logger = logging.getLogger("fwhr")

UI_TIPS = [
    {
        "title": "Хорошее освещение",
        "body": "Старайся снимать фото при мягком фронтальном свете — алгоритм точнее на равномерно освещённых лицах.",
    },
    {
        "title": "Смотри прямо",
        "body": "Нейтральная поза и взгляд в камеру помогают сетке landmark-ов правильно выстроить пропорции.",
    },
    {
        "title": "Кадрирование",
        "body": "Не обрезай часть лица и избегай сильных наклонов головы — это искажает fWHR.",
    },
]

app = FastAPI(title="fWHR Scan")
templates = Jinja2Templates(directory="templates")

_runtime_loaded = False
_runtime_device = None
_detector = None
_lm_model = None
_runtime_lock = Lock()
inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inference)
metrics_store = {
    "requests_total": 0,
    "faces_total": 0,
    "failures_total": 0,
}


def _record_metric(key: str, value: int):
    if key in metrics_store:
        metrics_store[key] += value


def get_device():
    preference = settings.device_preference.lower()
    if preference == "mps":
        return torch.device("mps")
    if preference == "cuda":
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_runtime():
    """
    Поднимаем пайплайн один раз и переиспользуем.
    """
    global _runtime_loaded, _runtime_device, _detector, _lm_model
    if _runtime_loaded:
        return _runtime_device

    with _runtime_lock:
        if _runtime_loaded:
            return _runtime_device

        device = get_device()
        logger.info(
            "Loading inference runtime",
            extra={"device": str(device), "checkpoint": settings.checkpoint_path},
        )

        _detector = faceboxesv2(device=str(device))
        torchlm.runtime.bind(_detector)

        _lm_model = pipnet(
            backbone="resnet18",
            pretrained=False,
            num_nb=10,
            num_lms=98,
            net_stride=32,
            input_size=256,
            meanface_type="wflw",
            map_location=str(device),
            checkpoint=settings.checkpoint_path,
        )
        torchlm.runtime.bind(_lm_model)

        _runtime_loaded = True
        _runtime_device = device
        logger.info("Runtime ready", extra={"device": str(device)})
        return device


def calculate_fwhr_from_landmarks(landmarks_xy: np.ndarray) -> float:
    """
    landmarks_xy: np.ndarray (N, 2) в пикселях.
    Грубая оценка fWHR.
    """
    xs = landmarks_xy[:, 0]
    ys = landmarks_xy[:, 1]

    face_width = xs.max() - xs.min()

    upper_y = np.percentile(ys, 20)
    lower_y = np.percentile(ys, 80)
    face_height = lower_y - upper_y

    if face_height <= 0:
        return 0.0

    return float(face_width / face_height)


def visualize_landmarks(pil_image: Image.Image, landmarks_xy: np.ndarray) -> io.BytesIO:
    """
    Ресайзим картинку до 224x224 и рисуем сетку точек.
    """
    w, h = pil_image.size
    target_size = 224
    scale_x = target_size / w
    scale_y = target_size / h

    landmarks_resized = landmarks_xy.copy()
    landmarks_resized[:, 0] *= scale_x
    landmarks_resized[:, 1] *= scale_y

    img_resized = pil_image.resize((target_size, target_size)).convert("L")

    fig, ax = plt.subplots()
    ax.imshow(np.array(img_resized), cmap="gray")
    ax.scatter(landmarks_resized[:, 0], landmarks_resized[:, 1], s=5)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf


def analyze_photo_bytes(image_bytes: bytes, device) -> Dict[str, List[Dict[str, str]]]:
    """
    Возвращает лендмарки и сводку метрик для каждого лица.
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image)

    landmarks_list, _ = torchlm.runtime.forward(image_np)

    if len(landmarks_list) == 0:
        raise RuntimeError("Не удалось найти лицо на фото. Попробуй другое фото.")

    results = []
    fwhr_values = []
    for idx, landmarks in enumerate(landmarks_list, start=1):
        landmarks_xy = np.array(landmarks)
        fwhr = calculate_fwhr_from_landmarks(landmarks_xy)
        fwhr_values.append(fwhr)

        viz_buf = visualize_landmarks(pil_image, landmarks_xy)
        viz_b64 = base64.b64encode(viz_buf.getvalue()).decode("utf-8")
        results.append(
            {
                "label": f"Лицо #{idx}",
                "fwhr": f"{fwhr:.2f}",
                "image_base64": viz_b64,
            }
        )

    stats = {
        "mean_fwhr": f"{np.mean(fwhr_values):.2f}",
        "min_fwhr": f"{np.min(fwhr_values):.2f}",
        "max_fwhr": f"{np.max(fwhr_values):.2f}",
    }

    return {"results": results, "stats": stats}


async def perform_analysis(image_bytes: bytes, request_id: str):
    device = ensure_runtime()
    started = time.perf_counter()
    await inference_semaphore.acquire()
    loop = asyncio.get_running_loop()
    try:
        analysis = await loop.run_in_executor(
            None, lambda: analyze_photo_bytes(image_bytes, device)
        )
        duration_ms = (time.perf_counter() - started) * 1000
        _record_metric("requests_total", 1)
        _record_metric("faces_total", len(analysis["results"]))
        logger.info(
            "Analysis finished",
            extra={
                "request_id": request_id,
                "faces": len(analysis["results"]),
                "duration_ms": round(duration_ms, 2),
            },
        )
        return analysis
    finally:
        inference_semaphore.release()


# ===============================
#   FastAPI endpoints + HTML UI
# ===============================


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.debug(
            "Request handled",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response
    except Exception as exc:
        _record_metric("failures_total", 1)
        logger.exception(
            "Request failed",
            extra={"request_id": request_id, "path": request.url.path},
        )
        raise exc


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "results": None,
            "error": None,
            "faces_count": 0,
            "stats": None,
            "source_preview": None,
            "tips": UI_TIPS,
            "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
            "metrics": metrics_store,
            "history_limit": settings.history_limit,
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_route(
    request: Request,
    image_file: UploadFile = File(None),
    camera_data: str = Form(None),
):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    context = {
        "request": request,
        "results": None,
        "error": None,
        "faces_count": 0,
        "stats": None,
        "source_preview": None,
        "tips": UI_TIPS,
        "request_id": request_id,
        "metrics": metrics_store,
        "history_limit": settings.history_limit,
        "analysis_timestamp": None,
    }
    image_bytes = None

    if image_file and image_file.filename:
        image_bytes = await image_file.read()
    elif camera_data:
        try:
            _, encoded = camera_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        except Exception:
            context["error"] = "Не удалось разобрать снимок с камеры."
            return templates.TemplateResponse(request, "index.html", context)
    else:
        context["error"] = "Добавь фото перед анализом."
        return templates.TemplateResponse(request, "index.html", context)

    try:
        analysis = await perform_analysis(image_bytes, request_id)
    except Exception as exc:
        context["error"] = str(exc)
        return templates.TemplateResponse(request, "index.html", context)

    context["results"] = analysis["results"]
    context["faces_count"] = len(analysis["results"])
    context["stats"] = analysis["stats"]
    context["source_preview"] = base64.b64encode(image_bytes).decode("utf-8")
    context["analysis_timestamp"] = datetime.utcnow().isoformat()

    return templates.TemplateResponse(request, "index.html", context)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("testo_app:app", host="0.0.0.0", port=8000, reload=True)
