from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import ALPHABET_MODEL, ALPHABET_TRAIN_DIR, BATCH_SIZE, EPOCHS, IMG_SIZE

ALPHABET_LANDMARKS_CSV = ROOT_DIR / "datasets" / "alphabet_landmarks.csv"
ALPHABET_LANDMARK_MODEL = ROOT_DIR / "models" / "alphabet_landmark_classifier.pkl"


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

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("Loading alphabet images...")
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    train_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    num_classes = len(train_gen.class_indices)
    print(f"Classes found: {num_classes}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_index_path = model_path.with_name("alphabet_classes.json")
    with class_index_path.open("w", encoding="utf-8") as file_handle:
        json.dump(train_gen.class_indices, file_handle, indent=2)
    print(f"Saved class indices to: {class_index_path}")

    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    print("Training alphabet CNN...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(model_path), save_best_only=True, verbose=1),
        ],
    )

    print(f"Saved model to: {model_path}")


def train_alphabet_landmark_model(data_csv: Path, model_path: Path) -> None:
    if not data_csv.exists():
        raise FileNotFoundError(
            f"Alphabet landmarks CSV not found: {data_csv}. "
            "Use --mode cnn, or pass --landmarks-csv with the correct file."
        )

    import pandas as pd
    import xgboost as xgb
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
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        use_label_encoder=False,
        eval_metric="mlogloss",
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
