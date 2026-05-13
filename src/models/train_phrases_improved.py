# src/models/train_phrases_improved.py
"""
Improved phrase training with better hyperparameters and data analysis
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import GESTURE_CSV, PHRASE_MODEL

def train_improved_phrase_model(data_csv: Path, model_path: Path) -> None:
    print("=" * 70)
    print("IMPROVED PHRASE MODEL TRAINING")
    print("=" * 70)
    
    print("\n📂 Loading dataset...")
    df = pd.read_csv(data_csv)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the phrase dataset.")

    print(f"✓ Dataset: {len(df)} samples, {df['label'].nunique()} phrases")
    print(f"\nSamples per phrase:")
    for label, count in df['label'].value_counts().sort_index().items():
        print(f"  {label:15} : {count:4} samples")

    X = df.drop(columns=["label"]).values
    y = df["label"].astype(str).values

    # Create encoder
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)
    
    print(f"\n✓ {len(encoder.classes_)} classes encoded")

    # Split data with stratification to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, encoded_y,
        test_size=0.2,
        random_state=42,
        stratify=encoded_y,
    )
    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

    # Train with improved hyperparameters
    print("\n🚀 Training improved RandomForest...")
    model = RandomForestClassifier(
        n_estimators=300,          # More trees for better accuracy
        max_depth=25,              # Slightly deeper trees
        min_samples_split=4,       # Prevent overfitting
        min_samples_leaf=2,
        max_features='sqrt',       # Use sqrt(n_features) at each split
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

    # Per-class accuracy
    print("\n📊 Per-Class Accuracy:")
    correct_per_class = {}
    total_per_class = {}
    
    for phrase_idx, phrase in enumerate(encoder.classes_):
        mask = y_test == phrase_idx
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            correct_per_class[phrase] = (y_pred[mask] == y_test[mask]).sum()
            total_per_class[phrase] = mask.sum()
            status = "✓" if class_acc >= 0.95 else "⚠"
            print(f"  {status} {phrase:15} - {class_acc*100:5.1f}% ({mask.sum()} samples)")

    # Identify problem areas
    low_accuracy_phrases = [
        phrase for phrase, acc in zip(encoder.classes_, 
        [accuracy_score(y_test[y_test==i], y_pred[y_test==i]) if (y_test==i).sum() > 0 else 0 
         for i in range(len(encoder.classes_))])
        if acc < 0.92
    ]

    if low_accuracy_phrases:
        print(f"\n⚠️  Phrases needing improvement:")
        for phrase in low_accuracy_phrases:
            phrase_idx = encoder.transform([phrase])[0]
            mask = y_test == phrase_idx
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"   {phrase:15} ({class_acc*100:.1f}%)")

    # Confidence distribution
    max_probs = np.max(y_pred_proba, axis=1)
    print(f"\n🔍 Confidence Distribution:")
    print(f"   >90%: {(max_probs >= 0.90).sum():4} predictions ({(max_probs >= 0.90).sum()/len(max_probs)*100:.1f}%)")
    print(f"   70-90%: {((max_probs >= 0.70) & (max_probs < 0.90)).sum():4} predictions")
    print(f"   50-70%: {((max_probs >= 0.50) & (max_probs < 0.70)).sum():4} predictions")
    print(f"   <50%: {(max_probs < 0.50).sum():4} predictions")

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump({"model": model, "encoder": encoder}, f)
    
    print(f"\n💾 Model saved to: {model_path}")
    print("\nℹ️  Updated thresholds for predict_phrase.py:")
    print("   STABLE_THRESHOLD = 15  (was 20)")
    print("   CONFIDENCE_THRESHOLD = 0.60  (was 0.70)")
    print("   UNKNOWN_THRESHOLD = 0.50  (for Unknown classification)")


if __name__ == "__main__":
    train_improved_phrase_model(Path(GESTURE_CSV), Path(PHRASE_MODEL))
