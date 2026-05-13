from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode

try:
    from config import HAND_LANDMARKER_MODEL, HAND_LANDMARKER_MODEL_URL, MODEL_DIR
except ModuleNotFoundError:  # pragma: no cover - import path depends on entrypoint
    from src.data.config import HAND_LANDMARKER_MODEL, HAND_LANDMARKER_MODEL_URL, MODEL_DIR

mp_hands = hand_landmarker


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


def _normalized_landmark_vector(hand_landmarks: Optional[object]) -> List[float]:
    if hand_landmarks is None:
        return []

    landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
    if not landmarks:
        return []

    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z

    values: List[float] = []
    for landmark in landmarks:
        values.extend([
            landmark.x - base_x,
            landmark.y - base_y,
            landmark.z - base_z,
        ])
    return values


def extract_landmark_vector(hand_landmarks: Optional[object]) -> List[float]:
    return _normalized_landmark_vector(hand_landmarks)


def frame_to_mp_image(frame: np.ndarray) -> image_lib.Image:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb_frame)


def _landmark_list(hand_landmarks: object) -> Iterable[object]:
    return getattr(hand_landmarks, "landmark", hand_landmarks)


def get_hand_bbox(
    hand_landmarks: object,
    frame_width: int,
    frame_height: int,
    padding: int = 30,
) -> Optional[Tuple[int, int, int, int]]:
    landmarks = list(_landmark_list(hand_landmarks))
    if not landmarks:
        return None

    xs = [int(landmark.x * frame_width) for landmark in landmarks]
    ys = [int(landmark.y * frame_height) for landmark in landmarks]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(frame_width, max(xs) + padding)
    y2 = min(frame_height, max(ys) + padding)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_hand_landmarks(
    frame: np.ndarray,
    results: object,
    *,
    show_bbox: bool = True,
) -> np.ndarray:
    hand_landmarks_list = getattr(results, "hand_landmarks", None)
    if not hand_landmarks_list:
        hand_landmarks_list = getattr(results, "multi_hand_landmarks", None)

    if hand_landmarks_list:
        height, width = frame.shape[:2]
        for hand_index, hand_landmarks in enumerate(hand_landmarks_list):
            landmarks = list(_landmark_list(hand_landmarks))
            color = (0, 255, 0) if hand_index == 0 else (0, 180, 255)

            for connection in mp_hands.HandLandmarksConnections.HAND_CONNECTIONS:
                if hasattr(connection, "start") and hasattr(connection, "end"):
                    start = int(connection.start)
                    end = int(connection.end)
                else:
                    start = int(connection[0])
                    end = int(connection[1])
                if start >= len(landmarks) or end >= len(landmarks):
                    continue

                x1 = int(landmarks[start].x * width)
                y1 = int(landmarks[start].y * height)
                x2 = int(landmarks[end].x * width)
                y2 = int(landmarks[end].y * height)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)

            for landmark_id, landmark in enumerate(landmarks):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)
                cv2.circle(frame, (x, y), 3, color, -1)
                cv2.putText(
                    frame,
                    str(landmark_id),
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if show_bbox:
                bbox = get_hand_bbox(hand_landmarks, width, height)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"Hand {hand_index + 1}",
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
    return frame
