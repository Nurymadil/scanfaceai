import base64
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import testo_app

client = TestClient(testo_app.app)


def _fake_png_bytes(color=(255, 0, 0)):
    img = Image.fromarray(
        np.full((10, 10, 3), color, dtype=np.uint8),
        mode="RGB",
    )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_metrics():
    testo_app.metrics_store.update(
        {"requests_total": 0, "faces_total": 0, "failures_total": 0}
    )
    yield


@pytest.fixture
def mock_analysis(monkeypatch):
    async def _fake_analysis(image_bytes, request_id):
        return {
            "results": [
                {"label": "Лицо #1", "fwhr": "2.01", "image_base64": base64.b64encode(b"fake").decode()}
            ],
            "stats": {"mean_fwhr": "2.01", "min_fwhr": "2.01", "max_fwhr": "2.01"},
        }

    monkeypatch.setattr(testo_app, "perform_analysis", _fake_analysis)


def test_index_page_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "fWHR Scan" in response.text


def test_analyze_route_with_file(mock_analysis):
    png_bytes = _fake_png_bytes()
    response = client.post(
        "/analyze",
        files={"image_file": ("face.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert "Найдено лиц" in response.text
    assert "fWHR 2.01" in response.text


def test_analyze_route_requires_input():
    response = client.post("/analyze")
    assert response.status_code == 200
    assert "Добавь фото" in response.text or "Добавь фото перед анализом." in response.text
