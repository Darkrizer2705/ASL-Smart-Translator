from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def draw_label(frame: np.ndarray, label: str, position: Tuple[int, int] = (20, 40)) -> np.ndarray:
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def stack_frames(frames: Iterable[np.ndarray]) -> np.ndarray:
    frames = list(frames)
    if not frames:
        raise ValueError("No frames provided")
    return np.concatenate(frames, axis=1)
