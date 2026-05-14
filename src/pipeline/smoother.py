# smoother.py
# -------------------------------------------------------
# What this does:
#   The camera model flickers a lot. It might say A, B, A, A, A very fast.
#   This file watches the last N predictions and only "accepts" one
#   when the same thing shows up enough times in a row.
#
# How to plug in your friend's model later:
#   Your friend's model gives: {"prediction": "A", "confidence": 0.95}
#   Just pass that into smoother.add(prediction, confidence)
#   Nothing else changes.
# -------------------------------------------------------

from collections import deque

class Smoother:
    def __init__(self, window_size=10, min_count=7, min_confidence=0.80):
        """
        window_size    : how many recent predictions to remember (default 10)
        min_count      : how many of those must match to accept it (default 7)
        min_confidence : ignore predictions below this confidence (default 0.80)
        """
        self.window_size     = window_size
        self.min_count       = min_count
        self.min_confidence  = min_confidence

        self.history = deque(maxlen=window_size)  # stores recent predictions
        self.last_accepted   = None               # avoid accepting same thing twice in a row

    def add(self, prediction: str, confidence: float):
        """
        Call this every frame with the model's output.
        Returns the accepted prediction (string) if stable, or None if still flickering.

        Example:
            result = smoother.add("A", 0.95)
            if result:
                print("Accepted:", result)
        """
        # Step 1: ignore low-confidence predictions
        if confidence < self.min_confidence:
            self.history.append("?")   # treat as noise
            return None

        # Step 2: ignore "nothing" — means no hand detected
        if prediction == "nothing":
            self.history.append("?")
            return None

        # Step 3: add to recent history
        self.history.append(prediction)

        # Step 4: count how many of the last N match this prediction
        count = self.history.count(prediction)

        # Step 5: accept only if it appears enough times AND is different from last accepted
        if count >= self.min_count and prediction != self.last_accepted:
            self.last_accepted = prediction
            return prediction

        return None  # still flickering, not ready yet

    def reset(self):
        """Call this when you want to start fresh (e.g. hand disappears for a while)"""
        self.history.clear()
        self.last_accepted = None
