# src/models/train_number_landmarks.py
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_CSV = "datasets/number_landmarks.csv"
OUTPUT_MODEL = "models/number_landmark_classifier.pkl"


def main():
    df = pd.read_csv(DATASET_CSV)
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

    print("\nTraining...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump({"model": model, "encoder": encoder}, f)
    print(f"Saved: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
