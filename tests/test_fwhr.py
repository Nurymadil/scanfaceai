import numpy as np
import pytest

from testo_app import calculate_fwhr_from_landmarks


def test_fwhr_calculation_simple_rectangle():
    landmarks = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 2.0],
            [4.0, 2.0],
        ]
    )
    result = calculate_fwhr_from_landmarks(landmarks)
    assert result == pytest.approx(2.0, rel=1e-3)


def test_fwhr_returns_zero_when_height_invalid():
    landmarks = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    assert calculate_fwhr_from_landmarks(landmarks) == 0.0
