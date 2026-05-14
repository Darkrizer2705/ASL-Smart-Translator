from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import ALPHABET_MODEL, ALPHABET_TRAIN_DIR, BATCH_SIZE, EPOCHS, IMG_SIZE

ALPHABET_LANDMARKS_CSV = ROOT_DIR / "datasets" / "alphabet_landmarks.csv"
ALPHABET_LANDMARK_MODEL = ROOT_DIR / "models" / "alphabet_landmark_classifier.pkl"

from src.utils.metrics import save_model_metrics
RESULTS_DIR = Path(__file__).resolve().parent / "metric"


def csv_has_data_rows(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False

    with csv_path.open("r", encoding="utf-8") as file_handle:
        next(file_handle, None)
        return next(file_handle, None) is not None


def resolve_training_mode(mode: str, image_data_dir: Path, landmarks_csv: Path) -> str:
    if mode != "auto":
        return mode

    if image_data_dir.exists():
        return "cnn"
    if csv_has_data_rows(landmarks_csv):
        return "landmarks"

    raise FileNotFoundError(
        "No alphabet training data found. Expected either image folders at "
        f"{image_data_dir} or a non-empty landmarks CSV at {landmarks_csv}."
    )


def train_alphabet_model(data_dir: Path, model_path: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Alphabet image dataset not found: {data_dir}. "
            "Use --mode landmarks, or pass --image-data-dir with the correct folder."
        )

    import cv2
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    import pickle

    print("Loading alphabet images...")
    X = []
    y = []
    
    classes_list = []
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            classes_list.append(class_dir.name)
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.resize(img, IMG_SIZE)
                        X.append(img.flatten() / 255.0)
                        y.append(class_dir.name)

    X = np.array(X)
    y = np.array(y)
    
    num_classes = len(classes_list)
    print(f"Classes found: {num_classes}")

    class_indices = {cls_name: i for i, cls_name in enumerate(classes_list)}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_index_path = model_path.with_name("alphabet_classes.json")
    with class_index_path.open("w", encoding="utf-8") as file_handle:
        json.dump(class_indices, file_handle, indent=2)
    print(f"Saved class indices to: {class_index_path}")
    
    y_encoded = np.array([class_indices[label] for label in y])

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    print("Training RandomForest classifier on images...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=classes_list))

    with model_path.open("wb") as file_handle:
        pickle.dump({"model": model, "classes": class_indices}, file_handle)
    
    print(f"Saved model to: {model_path}")

    print("\nEvaluating model to save metrics...")
    save_model_metrics(y_test, y_pred, classes_list, "alphabet_rf_image", RESULTS_DIR)


def train_alphabet_landmark_model(data_csv: Path, model_path: Path) -> None:
    if not data_csv.exists():
        raise FileNotFoundError(
            f"Alphabet landmarks CSV not found: {data_csv}. "
            "Use --mode cnn, or pass --landmarks-csv with the correct file."
        )

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    print("Loading alphabet landmarks...")
    df = pd.read_csv(data_csv)
    if df.empty:
        raise ValueError(
            f"Alphabet landmarks CSV has no samples: {data_csv}. "
            "Collect data first with: python src/data/collect_alphabet_landmarks.py"
        )
    if "label" not in df.columns:
        raise ValueError(f"Alphabet landmarks CSV is missing a label column: {data_csv}")

    df["label"] = df["label"].astype(str)

    print(f"Loaded {len(df)} rows | {df['label'].nunique()} classes")
    print(df["label"].value_counts().sort_index().to_string())

    x = df.drop("label", axis=1).values
    y = df["label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print(f"\nClasses: {list(encoder.classes_)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")

    print("\nTraining alphabet landmark classifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file_handle:
        pickle.dump({"model": model, "encoder": encoder}, file_handle)
    print(f"Saved model to: {model_path}")

    save_model_metrics(y_test, y_pred, list(encoder.classes_), "alphabet_landmarks", RESULTS_DIR)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train alphabet image CNN and/or landmark classifier."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "cnn", "landmarks", "both"),
        default="auto",
        help=(
            "Which alphabet model to train. Defaults to auto, which uses the "
            "image CNN data if present and otherwise trains landmarks."
        ),
    )
    parser.add_argument("--image-data-dir", type=Path, default=Path(ALPHABET_TRAIN_DIR))
    parser.add_argument("--cnn-model-path", type=Path, default=Path(ALPHABET_MODEL))
    parser.add_argument("--landmarks-csv", type=Path, default=ALPHABET_LANDMARKS_CSV)
    parser.add_argument("--landmark-model-path", type=Path, default=ALPHABET_LANDMARK_MODEL)
    args = parser.parse_args()

    mode = resolve_training_mode(args.mode, args.image_data_dir, args.landmarks_csv)
    print(f"Training mode: {mode}")

    if mode in ("cnn", "both"):
        train_alphabet_model(args.image_data_dir, args.cnn_model_path)

    if mode in ("landmarks", "both"):
        train_alphabet_landmark_model(args.landmarks_csv, args.landmark_model_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
