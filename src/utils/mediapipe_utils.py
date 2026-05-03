from __future__ import annotations

import os
from typing import List, Optional
from urllib.request import urlretrieve

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode

from config import HAND_LANDMARKER_MODEL, HAND_LANDMARKER_MODEL_URL, MODEL_DIR

mp_hands = hand_landmarker
mp_draw = drawing_utils


def ensure_hand_landmarker_model() -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(HAND_LANDMARKER_MODEL):
        try:
            urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise RuntimeError(
                "MediaPipe hand landmarker model is missing and could not be downloaded. "
                f"Place it at {HAND_LANDMARKER_MODEL} or retry with network access."
            ) from exc
    return HAND_LANDMARKER_MODEL


def create_hands_detector(
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options_lib.BaseOptions(
            model_asset_path=ensure_hand_landmarker_model()
        ),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return hand_landmarker.HandLandmarker.create_from_options(options)


def extract_landmark_vector(hand_landmarks: Optional[object]) -> List[float]:
    if hand_landmarks is None:
        return []

    landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
    values: List[float] = []
    for landmark in landmarks:
        values.extend([landmark.x, landmark.y, landmark.z])
    return values


def frame_to_mp_image(frame: np.ndarray) -> image_lib.Image:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb_frame)


def draw_hand_landmarks(frame: np.ndarray, results: object) -> np.ndarray:
    hand_landmarks_list = getattr(results, "hand_landmarks", None)
    if not hand_landmarks_list:
        hand_landmarks_list = getattr(results, "multi_hand_landmarks", None)

    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HandLandmarksConnections.HAND_CONNECTIONS,
            )
    return frame
