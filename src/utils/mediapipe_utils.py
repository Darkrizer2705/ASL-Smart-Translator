from __future__ import annotations

from typing import List, Optional

import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def create_hands_detector(
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_landmark_vector(hand_landmarks: Optional[object]) -> List[float]:
    if hand_landmarks is None:
        return []
    values: List[float] = []
    for landmark in hand_landmarks.landmark:
        values.extend([landmark.x, landmark.y, landmark.z])
    return values


def draw_hand_landmarks(frame: np.ndarray, results: object) -> np.ndarray:
    if getattr(results, "multi_hand_landmarks", None):
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame
