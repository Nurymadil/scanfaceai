import base64
import io
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch

import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ===============================
#   Настройки и вспомогательные функции
# ===============================

CHECKPOINT_PATH = "pipnet_resnet.pth"

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


def get_device():
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

    device = get_device()

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
        checkpoint=CHECKPOINT_PATH,
    )
    torchlm.runtime.bind(_lm_model)

    _runtime_loaded = True
    _runtime_device = device
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


# ===============================
#   FastAPI endpoints + HTML UI
# ===============================


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": None,
            "error": None,
            "faces_count": 0,
            "stats": None,
            "source_preview": None,
            "tips": UI_TIPS,
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_route(
    request: Request,
    image_file: UploadFile = File(None),
    camera_data: str = Form(None),
):
    context = {
        "request": request,
        "results": None,
        "error": None,
        "faces_count": 0,
        "stats": None,
        "source_preview": None,
        "tips": UI_TIPS,
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
            return templates.TemplateResponse("index.html", context)
    else:
        context["error"] = "Добавь фото перед анализом."
        return templates.TemplateResponse("index.html", context)

    device = ensure_runtime()

    try:
        analysis = analyze_photo_bytes(image_bytes, device)
    except Exception as exc:
        context["error"] = str(exc)
        return templates.TemplateResponse("index.html", context)

    context["results"] = analysis["results"]
    context["faces_count"] = len(analysis["results"])
    context["stats"] = analysis["stats"]
    context["source_preview"] = base64.b64encode(image_bytes).decode("utf-8")
    return templates.TemplateResponse("index.html", context)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("testo_app:app", host="0.0.0.0", port=8000, reload=True)
