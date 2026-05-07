from __future__ import annotations

from pathlib import Path


def predict_number(model_path: Path) -> str:
    raise NotImplementedError("Load your trained number model and return a prediction.")


if __name__ == "__main__":
    raise SystemExit("Import and call predict_number() from your app or capture loop.")
