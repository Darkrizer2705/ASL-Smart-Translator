# src/models/train_number_landmarks.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ── Load ───────────────────────────────────────────
df = pd.read_csv("datasets/number_landmarks.csv")
print(f"✅ {len(df)} rows | {df['label'].nunique()} classes")
print(df['label'].value_counts().to_string())

X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}")

# ── Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train ──────────────────────────────────────────
print("\n🚀 Training...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save ───────────────────────────────────────────
with open("models/number_landmark_classifier.pkl", "wb") as f:
    pickle.dump({"model": model, "encoder": le}, f)
print("🎉 Saved: models/number_landmark_classifier.pkl")
