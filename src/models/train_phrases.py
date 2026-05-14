from __future__ import annotations

import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import COMBINED_CSV, PHRASE_MODEL


MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 1,
}


def load_phrase_dataset(data_csv: Path) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    print(f"Loading phrase dataset: {data_csv}")
    frame = pd.read_csv(data_csv)
    if "label" not in frame.columns:
        raise ValueError("Expected a 'label' column in the phrase dataset.")

    frame = frame.dropna()
    print(f"Rows: {len(frame)} | labels: {frame['label'].nunique()}")
    print(f"Samples per label:\n{frame['label'].value_counts().sort_index().to_string()}\n")

    features = frame.drop(columns=["label"]).values
    labels = frame["label"].astype(str).str.strip().values

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    print(f"Classes: {list(encoder.classes_)}")

    return features, encoded_labels, encoder


def train_phrase_model(
    data_csv: Path,
    model_path: Path,
) -> None:
    print("=" * 70)
    print("PHRASE MODEL TRAINING")
    print("=" * 70)

    features, labels, encoder = load_phrase_dataset(data_csv)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")

    print("Training XGBoost classifier...")
    model = xgb.XGBClassifier(**MODEL_PARAMS, use_label_encoder=False, eval_metric="mlogloss")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=encoder.classes_))

    print("Per-class accuracy:")
    for phrase_index, phrase in enumerate(encoder.classes_):
        mask = y_test == phrase_index
        if mask.sum() == 0:
            continue
        class_accuracy = accuracy_score(y_test[mask], predictions[mask])
        print(f"  {phrase:15} - {class_accuracy * 100:5.1f}% ({mask.sum()} samples)")

    probabilities = model.predict_proba(x_test)
    max_probs = np.max(probabilities, axis=1)
    print("Confidence distribution:")
    print(f"  >90%: {(max_probs >= 0.90).sum():4}")
    print(f"  70-90%: {((max_probs >= 0.70) & (max_probs < 0.90)).sum():4}")
    print(f"  50-70%: {((max_probs >= 0.50) & (max_probs < 0.70)).sum():4}")
    print(f"  <50%: {(max_probs < 0.50).sum():4}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file_handle:
        pickle.dump({"model": model, "encoder": encoder}, file_handle)

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    train_phrase_model(Path(COMBINED_CSV), Path(PHRASE_MODEL))
