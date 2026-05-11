# src/models/train_phrases_mydata.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ── Load YOUR data only ────────────────────────────
CSV = "datasets/gesture_dataset.csv"

print("📂 Loading your personal gesture dataset...")
df = pd.read_csv(CSV)
print(f"✅ {len(df)} rows | {df['label'].nunique()} phrases")
print(f"\nSamples per phrase:\n{df['label'].value_counts().to_string()}\n")

X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Classes: {list(le.classes_)}")

# ── Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── Train ──────────────────────────────────────────
print("\n🚀 Training on YOUR hand data only...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save — overwrites old combined model ───────────
with open("models/phrase_classifier.pkl", "wb") as f:
    pickle.dump({"model": model, "encoder": le}, f)

print("\n🎉 Saved: models/phrase_classifier.pkl")
print("✅ Old model replaced with your personal model!")
