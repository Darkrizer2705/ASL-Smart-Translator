from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_landmark_csvs(source_csvs: list[Path], output_csv: Path) -> None:
    frames = [pd.read_csv(csv_path) for csv_path in source_csvs]
    if not frames:
        raise ValueError("No CSV files provided")
    pd.concat(frames, ignore_index=True).to_csv(output_csv, index=False)


if __name__ == "__main__":
    raise SystemExit("Import and call merge_landmark_csvs() with your landmark CSV paths.")
