# src/data/download_datasets.py
from pathlib import Path

import kaggle

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "datasets"

(DATASET_DIR / "alphabets").mkdir(parents=True, exist_ok=True)
(DATASET_DIR / "numbers").mkdir(parents=True, exist_ok=True)
(DATASET_DIR / "wlasl").mkdir(parents=True, exist_ok=True)

print("â¬‡ï¸ Downloading ASL Alphabets...")
kaggle.api.dataset_download_files(
    "grassknoted/asl-alphabet",
    path=str(DATASET_DIR / "alphabets"),
    unzip=True,
    quiet=False
)
print("âœ… Alphabets done")

print("â¬‡ï¸ Downloading ASL Numbers...")
kaggle.api.dataset_download_files(
    "lexset/synthetic-asl-numbers",
    path=str(DATASET_DIR / "numbers"),
    unzip=True,
    quiet=False
)
print("âœ… Numbers done")

print("â¬‡ï¸ Downloading WLASL...")
kaggle.api.dataset_download_files(
    "sttaseen/wlasl2000-resized",
    path=str(DATASET_DIR / "wlasl"),
    unzip=True,
    quiet=False
)
print("âœ… WLASL done")
print("ðŸŽ‰ All datasets downloaded!")
