from pathlib import Path

import pandas as pd

from config import COMBINED_CSV, GESTURE_CSV, PHRASE_CSV


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Not found: {path}")
        return None

    frame = pd.read_csv(path)
    print(f"{path.name}: {len(frame)} rows")
    print(f"   Columns: {list(frame.columns)}")
    return frame


def normalize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if "label" not in frame.columns:
        raise ValueError("Expected a 'label' column in the merged dataset.")

    normalized = frame.copy()
    normalized["label"] = normalized["label"].astype(str).str.strip().str.lower()
    return normalized


def main() -> None:
    gesture_path = Path(GESTURE_CSV)
    phrase_path = Path(PHRASE_CSV)
    combined_path = Path(COMBINED_CSV)

    frames: list[pd.DataFrame] = []

    gesture_df = load_csv(gesture_path)
    if gesture_df is not None:
        frames.append(normalize_labels(gesture_df))

    phrase_df = load_csv(phrase_path)
    if phrase_df is not None:
        frames.append(normalize_labels(phrase_df))

    if not frames:
        print("No data to merge!")
        return

    base_columns = list(frames[0].columns)
    for index, frame in enumerate(frames[1:], start=2):
        if list(frame.columns) != base_columns:
            print(f"Column mismatch in dataset {index}; reordering to match the first file.")
            missing = [column for column in base_columns if column not in frame.columns]
            if missing:
                raise ValueError(f"Dataset {index} is missing columns: {missing}")
            frames[index - 1] = frame[base_columns]

    combined = pd.concat(frames, ignore_index=True).dropna()
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)

    print(f"\nMerged dataset: {len(combined)} total rows")
    print(f"Saved to: {combined_path}")
    if "label" in combined.columns:
        print("\nSamples per label:")
        print(combined["label"].value_counts().to_string())


if __name__ == "__main__":
    main()