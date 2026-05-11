# fix_and_retrain.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── Step 1: Load your gesture_dataset ─────────────
print("📂 Loading gesture_dataset.csv...")
df = pd.read_csv("datasets/gesture_dataset.csv")

# ── Step 2: Fix uppercase labels ──────────────────
df["label"] = df["label"].astype(str).str.lower().str.strip()
print(f"✅ Labels normalized to lowercase")

# ── Step 3: Remove corrupt rows (all landmark coords zero) ───
# Some datasets may have x0==0 but valid landmarks; remove rows where
# the sum of absolute landmark coordinates is zero (i.e. all coords 0).
coords = [c for c in df.columns if c != "label"]
before = len(df)
row_sums = df[coords].abs().sum(axis=1)
# consider rows with near-zero sum as corrupt
df = df[row_sums > 1e-6].copy()
print(f"✅ Removed {before - len(df)} corrupt rows (all-zero landmarks)")

# ── Step 4: Remove low-sample phrases (<100) ──────
counts = df["label"].value_counts()
valid_labels = counts[counts >= 100].index
if len(valid_labels) < len(counts):
    print(f"Removing {len(counts) - len(valid_labels)} low-sample phrases (<100)")
df = df[df["label"].isin(valid_labels)]
print(f"✅ Kept {df['label'].nunique()} phrases with 100+ samples")

# ── Step 5: Show final distribution ───────────────
print(f"\nFinal dataset: {len(df)} rows")
print(f"Phrases: {df['label'].nunique()}")
print(f"\nSamples per phrase:")
print(df["label"].value_counts().to_string())

# ── Step 6: Train ──────────────────────────────────
X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

print("\n🚀 Training...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)

# ── Step 7: Evaluate ───────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Step 8: Save ───────────────────────────────────
with open("models/phrase_classifier.pkl", "wb") as f:
    pickle.dump({"model": model, "encoder": le}, f)
print("🎉 Saved: models/phrase_classifier.pkl")
print("✅ Model is now trained ONLY on your hand!")
