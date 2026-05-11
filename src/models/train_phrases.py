from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import COMBINED_CSV, PHRASE_MODEL


def train_phrase_model(data_csv: Path, model_path: Path) -> None:
    print("Loading combined dataset...")
    frame = pd.read_csv(data_csv)
    if "label" not in frame.columns:
        raise ValueError("Expected a 'label' column in the phrase dataset.")

    print(f"Rows: {len(frame)} | labels: {frame['label'].nunique()}")
    print(f"Samples per label:\n{frame['label'].value_counts().to_string()}\n")

    features = frame.drop(columns=["label"]).values
    labels = frame["label"].astype(str).values

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    print(f"Classes: {list(encoder.classes_)}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels,
    )
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")

    print("Training RandomForest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=encoder.classes_))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file_handle:
        pickle.dump({"model": model, "encoder": encoder}, file_handle)

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    train_phrase_model(Path(COMBINED_CSV), Path(PHRASE_MODEL))
