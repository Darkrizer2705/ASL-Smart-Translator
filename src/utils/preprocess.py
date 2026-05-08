from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def resize_image(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return image / 255.0


def preprocess_frame(frame: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    resized = resize_image(frame, size)
    return normalize_image(resized)


def to_bgr(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    raise TypeError("Expected a NumPy array image")
