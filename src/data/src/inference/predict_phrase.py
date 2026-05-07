from __future__ import annotations

from pathlib import Path


def predict_phrase(model_path: Path) -> str:
    raise NotImplementedError("Load your trained phrase model and return a prediction.")


if __name__ == "__main__":
    raise SystemExit("Import and call predict_phrase() from your app or capture loop.")
